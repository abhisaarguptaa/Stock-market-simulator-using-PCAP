// The tick-synchronous simulator.
//
// Each tick has three phases, separated by barriers:
//   1. decide  - every agent reads its instrument's snapshot and emits an intent
//                (serial loop, OpenMP loop, or CUDA kernel)
//   2. match   - each instrument's intents are applied to its order book in a
//                deterministic arrival order (serial, or OpenMP across instruments)
//   3. publish - snapshots are rebuilt and the cross-instrument index is reduced
//                (MPI_Allreduce when instruments are split across ranks)
// All randomness is counter-based, so every back end produces the same trades.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lob/agents.hpp"
#include "lob/order_book.hpp"

namespace lob {

enum class DecideBackend { Serial, OpenMP, CUDA };
enum class MatchBackend { Serial, OpenMP };

struct Config {
  uint64_t seed = 42;
  uint32_t num_agents = 10000;
  uint32_t num_instruments = 8;
  uint32_t ticks = 1000;
  int64_t order_ttl = 100;  // resting orders expire after this many ticks (0 = never)
  uint32_t mix[kNumAgentTypes] = {65, 15, 10, 10};  // percent noise/momentum/meanrev/mm
  DecideBackend decide = DecideBackend::Serial;
  MatchBackend match = MatchBackend::Serial;
  int threads = 0;          // OpenMP threads (0 = runtime default)
  bool sort_by_type = true; // group agents by strategy (reduces GPU warp divergence)
  int cuda_block = 256;     // CUDA threads per block
  bool cuda_compact = true; // copy back only the agents that acted this tick
  Params params;
  std::string trades_path;  // optional CSV of every trade
  std::string prices_path;  // optional CSV of per-tick reference prices
};

// Per-phase wall-clock seconds, summed over all ticks.
struct Timings {
  double decide = 0, h2d = 0, kernel = 0, d2h = 0;  // h2d/kernel/d2h: CUDA only
  double match = 0, publish = 0, comm = 0, total = 0;
};

struct Result {
  uint64_t trade_hash = 0;   // order-sensitive hash of the full trade log
  uint64_t num_trades = 0;
  uint64_t volume = 0;
  Timings t;
};

// Computes intents for the agents owned by this process.
class DecideEngine {
 public:
  virtual ~DecideEngine() = default;
  // snaps is indexed by local instrument; out is indexed like the agent arrays.
  virtual void decide(int64_t tick, const Snapshot* snaps, Intent* out, Timings& t) = 0;
};

// Agents owned by this process, structure-of-arrays, sorted by type so that
// neighbouring GPU threads follow the same branch (less warp divergence).
struct AgentArrays {
  std::vector<uint32_t> id;
  std::vector<uint8_t> type;
  std::vector<uint32_t> local_instr;
  size_t size() const { return id.size(); }
};

std::unique_ptr<DecideEngine> make_cpu_engine(const Config& c, const AgentArrays& a,
                                              bool parallel);
#ifdef LOB_USE_CUDA
std::unique_ptr<DecideEngine> make_cuda_engine(const Config& c, const AgentArrays& a);
#endif

// Counters touched on every trade. Kept in their own array; each entry is
// padded to a cache line so threads matching neighbouring instruments do not
// false-share. Build with -DLOB_NO_PAD to measure the effect of removing it.
#ifdef LOB_NO_PAD
struct HotCounters {
#else
struct alignas(64) HotCounters {
#endif
  uint64_t hash = 1469598103934665603ull;  // FNV-1a offset basis
  uint64_t trades = 0;
  uint64_t volume = 0;
};

// Per-instrument state that is only touched by the thread matching it.
struct InstrState {
  OrderBook book;
  Price last = kInitPrice;
  int32_t ema_q8 = kInitPrice * 256;
  Price fundamental = kInitPrice;
  std::vector<Trade> log;      // filled only when trades_path is set
  std::vector<Price> history;  // filled only when prices_path is set
  explicit InstrState(int32_t instr, int64_t ttl) : book(instr, ttl) {}
};

class Simulator {
 public:
  // rank/nranks split instruments into contiguous blocks (1 process: 0/1).
  Simulator(const Config& c, int rank = 0, int nranks = 1);
  ~Simulator();
  Result run();

  // Exposed for tests.
  uint32_t first_instrument() const { return instr_lo_; }
  uint32_t local_instruments() const { return instr_hi_ - instr_lo_; }
  const AgentArrays& agents() const { return agents_; }

 private:
  void match_instrument(uint32_t li, int64_t tick);
  void publish(int64_t tick, Timings& t);

  Config cfg_;
  int rank_, nranks_;
  uint32_t instr_lo_, instr_hi_;
  AgentArrays agents_;
  std::vector<std::vector<uint32_t>> arrivals_;  // per local instrument: agent slots by id
  std::vector<InstrState> instr_;
  std::vector<HotCounters> hot_;
  std::vector<Snapshot> snaps_;
  std::vector<Intent> intents_;
  std::unique_ptr<DecideEngine> engine_;
  int64_t index_ema_q8_ = 0;
};

const char* to_string(DecideBackend b);
const char* to_string(MatchBackend b);

}  // namespace lob
