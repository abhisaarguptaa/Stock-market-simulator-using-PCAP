#include "lob/sim.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef LOB_USE_MPI
#include <mpi.h>
#endif

namespace lob {

namespace {

double now() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

inline uint64_t fnv_mix(uint64_t h, uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    h ^= (v >> (8 * i)) & 0xFFu;
    h *= 1099511628211ull;
  }
  return h;
}

uint64_t hash_trade(uint64_t h, const Trade& t) {
  h = fnv_mix(h, static_cast<uint64_t>(t.tick));
  h = fnv_mix(h, (static_cast<uint64_t>(static_cast<uint32_t>(t.price)) << 32) |
                     static_cast<uint32_t>(t.qty));
  h = fnv_mix(h, (static_cast<uint64_t>(t.buyer_agent) << 32) | t.seller_agent);
  h = fnv_mix(h, t.maker_id);
  h = fnv_mix(h, t.taker_id ^ (static_cast<uint64_t>(t.taker_side) << 63));
  return h;
}

// Updates the hot counters and the last price as trades happen.
class InstrSink : public TradeSink {
 public:
  InstrSink(HotCounters& hot, InstrState& st, bool keep_log) : hot_(hot), st_(st), keep_(keep_log) {}
  void on_trade(const Trade& t) override {
    hot_.hash = hash_trade(hot_.hash, t);
    hot_.trades += 1;
    hot_.volume += static_cast<uint64_t>(t.qty);
    st_.last = t.price;
    if (keep_) st_.log.push_back(t);
  }

 private:
  HotCounters& hot_;
  InstrState& st_;
  bool keep_;
};

class CpuEngine : public DecideEngine {
 public:
  CpuEngine(const Config& c, const AgentArrays& a, bool parallel)
      : cfg_(c), a_(a), parallel_(parallel) {}
  void decide(int64_t tick, const Snapshot* snaps, Intent* out, Timings&) override {
    const int64_t n = static_cast<int64_t>(a_.size());
    const uint32_t* id = a_.id.data();
    const uint8_t* type = a_.type.data();
    const uint32_t* li = a_.local_instr.data();
    const Params P = cfg_.params;
    const uint64_t seed = cfg_.seed;
#pragma omp parallel for schedule(static) if (parallel_)
    for (int64_t i = 0; i < n; ++i)
      out[i] = decide_one(id[i], type[i], tick, seed, snaps[li[i]], P);
  }

 private:
  static Intent decide_one(uint32_t id, uint8_t type, int64_t tick, uint64_t seed,
                           const Snapshot& s, const Params& P) {
    return lob::decide(id, type, tick, seed, s, P);
  }
  Config cfg_;
  const AgentArrays& a_;
  bool parallel_;
};

}  // namespace

std::unique_ptr<DecideEngine> make_cpu_engine(const Config& c, const AgentArrays& a,
                                              bool parallel) {
  return std::unique_ptr<DecideEngine>(new CpuEngine(c, a, parallel));
}

const char* to_string(DecideBackend b) {
  switch (b) {
    case DecideBackend::Serial: return "serial";
    case DecideBackend::OpenMP: return "omp";
    case DecideBackend::CUDA: return "cuda";
  }
  return "?";
}

const char* to_string(MatchBackend b) {
  return b == MatchBackend::Serial ? "serial" : "omp";
}

Simulator::Simulator(const Config& c, int rank, int nranks)
    : cfg_(c), rank_(rank), nranks_(nranks) {
  if (cfg_.num_instruments == 0 || cfg_.num_agents == 0)
    throw std::invalid_argument("need at least one agent and one instrument");
  const uint64_t n = cfg_.num_instruments;
  instr_lo_ = static_cast<uint32_t>(n * rank / nranks);
  instr_hi_ = static_cast<uint32_t>(n * (rank + 1) / nranks);
  const uint32_t nloc = instr_hi_ - instr_lo_;

  // Cumulative strategy mix in units of 1/65536.
  uint32_t total = 0;
  for (uint32_t m : cfg_.mix) total += m;
  if (total == 0) throw std::invalid_argument("agent mix sums to zero");
  uint32_t cum[kNumAgentTypes];
  uint32_t run = 0;
  for (int t = 0; t < kNumAgentTypes; ++t) {
    run += cfg_.mix[t];
    cum[t] = static_cast<uint32_t>(static_cast<uint64_t>(run) * 65536u / total);
  }

  // Agent a trades instrument a % num_instruments. Keep only local agents,
  // bucketed by type (ids stay ascending within each bucket).
  // With sort_by_type off, agents stay in id order (strategies interleaved).
  std::vector<uint32_t> bucket[kNumAgentTypes];
  std::vector<uint8_t> type_of;
  for (uint32_t a = 0; a < cfg_.num_agents; ++a) {
    const uint32_t instr = a % cfg_.num_instruments;
    if (instr < instr_lo_ || instr >= instr_hi_) continue;
    const uint8_t ty = assign_type(a, cfg_.seed, cum);
    bucket[cfg_.sort_by_type ? ty : 0].push_back(a);
    type_of.push_back(ty);
  }
  size_t k = 0;
  for (int t = 0; t < kNumAgentTypes; ++t)
    for (uint32_t a : bucket[t]) {
      agents_.id.push_back(a);
      agents_.local_instr.push_back(a % cfg_.num_instruments - instr_lo_);
      agents_.type.push_back(cfg_.sort_by_type ? static_cast<uint8_t>(t) : type_of[k]);
      ++k;
    }

  // Per instrument, the agent slots in ascending agent-id order.
  arrivals_.assign(nloc, {});
  for (uint32_t s = 0; s < agents_.size(); ++s) arrivals_[agents_.local_instr[s]].push_back(s);
  for (auto& v : arrivals_)
    std::sort(v.begin(), v.end(),
              [this](uint32_t x, uint32_t y) { return agents_.id[x] < agents_.id[y]; });

  instr_.reserve(nloc);
  for (uint32_t i = 0; i < nloc; ++i)
    instr_.emplace_back(static_cast<int32_t>(instr_lo_ + i), cfg_.order_ttl);
  hot_.assign(nloc, HotCounters{});

  Snapshot s0{};
  s0.bid = kNoBid;
  s0.ask = kNoAsk;
  s0.ref = kInitPrice;
  s0.fundamental = kInitPrice;
  snaps_.assign(nloc, s0);
  intents_.assign(agents_.size(), no_intent());
  index_ema_q8_ = static_cast<int64_t>(kInitPrice) * 256;

  switch (cfg_.decide) {
    case DecideBackend::Serial: engine_ = make_cpu_engine(cfg_, agents_, false); break;
    case DecideBackend::OpenMP: engine_ = make_cpu_engine(cfg_, agents_, true); break;
    case DecideBackend::CUDA:
#ifdef LOB_USE_CUDA
      engine_ = make_cuda_engine(cfg_, agents_);
      break;
#else
      throw std::runtime_error("built without CUDA (configure with -DLOB_USE_CUDA=ON)");
#endif
  }
}

Simulator::~Simulator() = default;

void Simulator::match_instrument(uint32_t li, int64_t tick) {
  InstrState& st = instr_[li];
  InstrSink sink(hot_[li], st, !cfg_.trades_path.empty());
  const uint32_t global = instr_lo_ + li;

  // Arrival order this tick: agent-id order rotated by a random offset, so no
  // agent is always first in the queue. Depends only on (seed, instrument, tick).
  const std::vector<uint32_t>& slots = arrivals_[li];
  const uint32_t n = static_cast<uint32_t>(slots.size());
  if (n > 0) {
    const uint32_t off = rand_below(
        rng_draw(cfg_.seed, kStreamArrival, global, static_cast<uint32_t>(tick)).v[0], n);
    for (uint32_t k = 0; k < n; ++k) {
      const uint32_t slot = slots[(off + k) % n];
      const Intent& in = intents_[slot];
      if (in.action == ACT_NONE) continue;
      const uint32_t agent = agents_.id[slot];
      const OrderId id = make_order_id(tick, agent);
      const Side side = static_cast<Side>(in.side);
      switch (in.action) {
        case ACT_LIMIT:
          st.book.submit_limit(id, agent, side, in.price, in.qty, tick, sink);
          break;
        case ACT_MARKET:
          st.book.submit_market(id, agent, side, in.qty, tick, sink);
          break;
        case ACT_REPLACE:
          if (in.cancel_id) st.book.cancel(in.cancel_id);
          st.book.submit_limit(id, agent, side, in.price, in.qty, tick, sink);
          break;
        default:
          break;
      }
    }
  }
  st.book.expire(tick);

  // Fundamental value: small random walk kept inside the middle of the grid.
  const uint32_t u = rand_below(
      rng_draw(cfg_.seed, kStreamFundamental, global, static_cast<uint32_t>(tick)).v[0], 100u);
  if (u < 5 && st.fundamental > kNumPrices / 4) st.fundamental -= 1;
  if (u >= 95 && st.fundamental < 3 * kNumPrices / 4) st.fundamental += 1;

  // Rebuild this instrument's snapshot (the index trend is filled in later).
  Snapshot& s = snaps_[li];
  s.bid = st.book.best_bid();
  s.ask = st.book.best_ask();
  const bool two_sided = s.bid != kNoBid && s.ask != kNoAsk;
  s.ref = two_sided ? (s.bid + s.ask) / 2 : st.last;
  s.spread = two_sided ? s.ask - s.bid : 0;
  s.fundamental = st.fundamental;
  st.ema_q8 += (s.ref * 256 - st.ema_q8) / 16;
  s.trend_q8 = s.ref * 256 - st.ema_q8;
  if (!cfg_.prices_path.empty()) st.history.push_back(s.ref);
}

void Simulator::publish(int64_t, Timings& t) {
  const uint32_t nloc = instr_hi_ - instr_lo_;
  int64_t local = 0;
  for (uint32_t i = 0; i < nloc; ++i) local += static_cast<int64_t>(snaps_[i].ref) * 256;
  int64_t global = local;
#ifdef LOB_USE_MPI
  if (nranks_ > 1) {
    const double c0 = now();
    MPI_Allreduce(&local, &global, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    t.comm += now() - c0;
  }
#else
  (void)t;
#endif
  const int64_t index_q8 = global / static_cast<int64_t>(cfg_.num_instruments);
  index_ema_q8_ += (index_q8 - index_ema_q8_) / 32;
  int64_t trend = index_q8 - index_ema_q8_;
  trend = std::max<int64_t>(-(1 << 30), std::min<int64_t>(1 << 30, trend));
  for (uint32_t i = 0; i < nloc; ++i) snaps_[i].index_trend_q8 = static_cast<int32_t>(trend);
}

Result Simulator::run() {
  Result r;
  Timings& t = r.t;
  const uint32_t nloc = instr_hi_ - instr_lo_;
  const bool par_match = cfg_.match == MatchBackend::OpenMP;
#ifdef LOB_USE_MPI
  if (nranks_ > 1) MPI_Barrier(MPI_COMM_WORLD);
#endif
  const double start = now();
  for (int64_t tick = 0; tick < static_cast<int64_t>(cfg_.ticks); ++tick) {
    double t0 = now();
    engine_->decide(tick, snaps_.data(), intents_.data(), t);
    double t1 = now();
    t.decide += t1 - t0;

#pragma omp parallel for schedule(dynamic, 1) if (par_match)
    for (int64_t i = 0; i < static_cast<int64_t>(nloc); ++i)
      match_instrument(static_cast<uint32_t>(i), tick);
    double t2 = now();
    t.match += t2 - t1;

    publish(tick, t);
    t.publish += now() - t2;
  }
  t.total = now() - start;

  // Combine per-instrument hashes in global instrument order. Instruments are
  // disjoint across ranks, so XOR-reducing zero-padded arrays gathers them.
  std::vector<uint64_t> hashes(cfg_.num_instruments, 0);
  uint64_t counts[2] = {0, 0};
  for (uint32_t i = 0; i < nloc; ++i) {
    hashes[instr_lo_ + i] = hot_[i].hash;
    counts[0] += hot_[i].trades;
    counts[1] += hot_[i].volume;
  }
#ifdef LOB_USE_MPI
  if (nranks_ > 1) {
    MPI_Allreduce(MPI_IN_PLACE, hashes.data(), static_cast<int>(hashes.size()),
                  MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  uint64_t h = 1469598103934665603ull;
  for (uint64_t x : hashes) h = fnv_mix(h, x);
  r.trade_hash = h;
  r.num_trades = counts[0];
  r.volume = counts[1];

  const std::string suffix = nranks_ > 1 ? ".rank" + std::to_string(rank_) : "";
  if (!cfg_.trades_path.empty()) {
    FILE* f = std::fopen((cfg_.trades_path + suffix).c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + cfg_.trades_path + suffix);
    std::fprintf(f, "tick,instrument,price,qty,taker_side,buyer,seller,maker_id,taker_id\n");
    for (const InstrState& st : instr_)
      for (const Trade& x : st.log)
        std::fprintf(f, "%lld,%d,%d,%d,%c,%u,%u,%llu,%llu\n", static_cast<long long>(x.tick),
                     x.instrument, x.price, x.qty, x.taker_side == BUY ? 'B' : 'S',
                     x.buyer_agent, x.seller_agent,
                     static_cast<unsigned long long>(x.maker_id),
                     static_cast<unsigned long long>(x.taker_id));
    std::fclose(f);
  }
  if (!cfg_.prices_path.empty()) {
    FILE* f = std::fopen((cfg_.prices_path + suffix).c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + cfg_.prices_path + suffix);
    std::fprintf(f, "tick,instrument,price\n");
    for (uint32_t i = 0; i < nloc; ++i)
      for (size_t k = 0; k < instr_[i].history.size(); ++k)
        std::fprintf(f, "%zu,%u,%d\n", k, instr_lo_ + i, instr_[i].history[k]);
    std::fclose(f);
  }
  return r;
}

}  // namespace lob
