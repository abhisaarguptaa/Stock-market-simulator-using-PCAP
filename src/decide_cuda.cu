// CUDA back end for the agent decision phase.
//
// One thread per agent. Agent data is structure-of-arrays (coalesced loads),
// agents are pre-sorted by strategy (less warp divergence), the per-instrument
// snapshots are tiny and read through the read-only cache, and the strategy
// parameters sit in constant memory.
//
// With compaction on, only agents that actually act are copied back: each
// acting thread reserves a slot with an atomic counter and writes (agent slot,
// intent). The host scatters them by agent slot, so the order in which threads
// win the atomic does not affect results.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>
#include <stdexcept>
#include <string>

#include "lob/sim.hpp"

namespace lob {

namespace {

#define CUDA_CHECK(x)                                                              \
  do {                                                                             \
    cudaError_t err_ = (x);                                                        \
    if (err_ != cudaSuccess)                                                       \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) + \
                               " at " __FILE__ ":" + std::to_string(__LINE__));     \
  } while (0)

// Params has default member initialisers, which constant memory does not
// allow, so it is stored as raw bytes.
__constant__ __align__(16) unsigned char c_params[sizeof(Params)];

struct Compact {
  uint32_t slot;
  Intent intent;
};

__global__ void decide_full(uint32_t n, const uint32_t* __restrict__ id,
                            const uint8_t* __restrict__ type,
                            const uint32_t* __restrict__ li,
                            const Snapshot* __restrict__ snaps, Intent* __restrict__ out,
                            int64_t tick, uint64_t seed) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const Params& P = *reinterpret_cast<const Params*>(c_params);
  const Snapshot s = snaps[__ldg(&li[i])];
  out[i] = decide(__ldg(&id[i]), __ldg(&type[i]), tick, seed, s, P);
}

__global__ void decide_compact(uint32_t n, const uint32_t* __restrict__ id,
                               const uint8_t* __restrict__ type,
                               const uint32_t* __restrict__ li,
                               const Snapshot* __restrict__ snaps,
                               Compact* __restrict__ out, uint32_t* __restrict__ count,
                               int64_t tick, uint64_t seed) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const Params& P = *reinterpret_cast<const Params*>(c_params);
  const Snapshot s = snaps[__ldg(&li[i])];
  const Intent it = decide(__ldg(&id[i]), __ldg(&type[i]), tick, seed, s, P);
  if (it.action == ACT_NONE) return;
  const uint32_t k = atomicAdd(count, 1u);
  out[k].slot = i;
  out[k].intent = it;
}

double wall() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

class CudaEngine : public DecideEngine {
 public:
  CudaEngine(const Config& c, const AgentArrays& a)
      : seed_(c.seed), n_(static_cast<uint32_t>(a.size())), block_(c.cuda_block),
        compact_(c.cuda_compact) {
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) throw std::runtime_error("no CUDA device found");
    nsnap_ = 0;
    for (uint32_t x : a.local_instr) nsnap_ = x + 1 > nsnap_ ? x + 1 : nsnap_;
    CUDA_CHECK(cudaMemcpyToSymbol(c_params, &c.params, sizeof(Params)));
    CUDA_CHECK(cudaMalloc(&d_id_, n_ * sizeof(uint32_t) + 1));
    CUDA_CHECK(cudaMalloc(&d_type_, n_ + 1));
    CUDA_CHECK(cudaMalloc(&d_li_, n_ * sizeof(uint32_t) + 1));
    CUDA_CHECK(cudaMalloc(&d_snaps_, nsnap_ * sizeof(Snapshot) + 1));
    CUDA_CHECK(cudaMemcpy(d_id_, a.id.data(), n_ * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_type_, a.type.data(), n_, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_li_, a.local_instr.data(), n_ * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    if (compact_) {
      CUDA_CHECK(cudaMalloc(&d_compact_, n_ * sizeof(Compact) + 1));
      CUDA_CHECK(cudaMalloc(&d_count_, sizeof(uint32_t)));
      CUDA_CHECK(cudaMallocHost(&h_compact_, n_ * sizeof(Compact) + 1));
      CUDA_CHECK(cudaMallocHost(&h_count_, sizeof(uint32_t)));
    } else {
      CUDA_CHECK(cudaMalloc(&d_out_, n_ * sizeof(Intent) + 1));
      CUDA_CHECK(cudaMallocHost(&h_out_, n_ * sizeof(Intent) + 1));
    }
    CUDA_CHECK(cudaEventCreate(&ev0_));
    CUDA_CHECK(cudaEventCreate(&ev1_));
  }

  ~CudaEngine() override {
    cudaFree(d_id_); cudaFree(d_type_); cudaFree(d_li_); cudaFree(d_snaps_);
    cudaFree(d_out_); cudaFree(d_compact_); cudaFree(d_count_);
    cudaFreeHost(h_out_); cudaFreeHost(h_compact_); cudaFreeHost(h_count_);
    cudaEventDestroy(ev0_); cudaEventDestroy(ev1_);
  }

  void decide(int64_t tick, const Snapshot* snaps, Intent* out, Timings& t) override {
    if (n_ == 0) return;
    const uint32_t grid = (n_ + block_ - 1) / block_;

    double t0 = wall();
    CUDA_CHECK(cudaMemcpy(d_snaps_, snaps, nsnap_ * sizeof(Snapshot), cudaMemcpyHostToDevice));
    if (compact_) CUDA_CHECK(cudaMemsetAsync(d_count_, 0, sizeof(uint32_t)));
    t.h2d += wall() - t0;

    CUDA_CHECK(cudaEventRecord(ev0_));
    if (compact_)
      decide_compact<<<grid, block_>>>(n_, d_id_, d_type_, d_li_, d_snaps_, d_compact_,
                                       d_count_, tick, seed_);
    else
      decide_full<<<grid, block_>>>(n_, d_id_, d_type_, d_li_, d_snaps_, d_out_, tick, seed_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev1_));
    CUDA_CHECK(cudaEventSynchronize(ev1_));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0_, ev1_));
    t.kernel += ms * 1e-3;

    t0 = wall();
    if (compact_) {
      CUDA_CHECK(cudaMemcpy(h_count_, d_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
      const uint32_t k = *h_count_;
      if (k) CUDA_CHECK(cudaMemcpy(h_compact_, d_compact_, k * sizeof(Compact),
                                   cudaMemcpyDeviceToHost));
      t.d2h += wall() - t0;
      // Clear last tick's intents, then scatter this tick's by agent slot.
      for (uint32_t s : touched_) out[s] = no_intent();
      touched_.resize(k);
      for (uint32_t j = 0; j < k; ++j) {
        out[h_compact_[j].slot] = h_compact_[j].intent;
        touched_[j] = h_compact_[j].slot;
      }
    } else {
      CUDA_CHECK(cudaMemcpy(h_out_, d_out_, n_ * sizeof(Intent), cudaMemcpyDeviceToHost));
      t.d2h += wall() - t0;
      std::memcpy(out, h_out_, n_ * sizeof(Intent));
    }
  }

 private:
  uint64_t seed_;
  uint32_t n_, nsnap_;
  int block_;
  bool compact_;
  uint32_t *d_id_ = nullptr, *d_li_ = nullptr, *d_count_ = nullptr, *h_count_ = nullptr;
  uint8_t* d_type_ = nullptr;
  Snapshot* d_snaps_ = nullptr;
  Intent *d_out_ = nullptr, *h_out_ = nullptr;
  Compact *d_compact_ = nullptr, *h_compact_ = nullptr;
  cudaEvent_t ev0_, ev1_;
  std::vector<uint32_t> touched_;
};

}  // namespace

std::unique_ptr<DecideEngine> make_cuda_engine(const Config& c, const AgentArrays& a) {
  return std::unique_ptr<DecideEngine>(new CudaEngine(c, a));
}

}  // namespace lob
