// Unit tests. Self-contained (no test framework download needed).
// The order book tests are the correctness reference for every parallel
// variant; the simulator tests check that all back ends give identical trades.
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <vector>

#include "lob/agents.hpp"
#include "lob/order_book.hpp"
#include "lob/philox.hpp"
#include "lob/sim.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace lob;

static int g_failed = 0, g_checks = 0;
static std::vector<std::pair<std::string, std::function<void()>>>& registry() {
  static std::vector<std::pair<std::string, std::function<void()>>> r;
  return r;
}
struct Reg {
  Reg(const char* n, std::function<void()> f) { registry().emplace_back(n, f); }
};
#define TEST(name)                       \
  static void name();                    \
  static Reg reg_##name(#name, name);    \
  static void name()
#define CHECK(cond)                                                              \
  do {                                                                           \
    ++g_checks;                                                                  \
    if (!(cond)) {                                                               \
      ++g_failed;                                                                \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);              \
    }                                                                            \
  } while (0)
#define CHECK_EQ(a, b) CHECK((a) == (b))

// ---------------------------------------------------------------- RNG

TEST(philox_known_answers) {
  // Known-answer vectors from the Random123 distribution (kat_vectors).
  U32x4 c0{{0, 0, 0, 0}};
  U32x4 r = philox4x32_10(c0, 0, 0);
  CHECK_EQ(r.v[0], 0x6627e8d5u);
  CHECK_EQ(r.v[1], 0xe169c58du);
  CHECK_EQ(r.v[2], 0xbc57ac4cu);
  CHECK_EQ(r.v[3], 0x9b00dbd8u);
  U32x4 c1{{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}};
  r = philox4x32_10(c1, 0xffffffffu, 0xffffffffu);
  CHECK_EQ(r.v[0], 0x408f276du);
  CHECK_EQ(r.v[1], 0x41c83b0eu);
  CHECK_EQ(r.v[2], 0xa20bc7c6u);
  CHECK_EQ(r.v[3], 0x6d5451fdu);
  U32x4 c2{{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}};
  r = philox4x32_10(c2, 0xa4093822u, 0x299f31d0u);
  CHECK_EQ(r.v[0], 0xd16cfe09u);
  CHECK_EQ(r.v[1], 0x94fdccebu);
  CHECK_EQ(r.v[2], 0x5001e420u);
  CHECK_EQ(r.v[3], 0x24126ea1u);
}

TEST(rand_below_in_range) {
  for (uint32_t i = 0; i < 10000; ++i) {
    const uint32_t r = rng_draw(7, 1, i, 0).v[0];
    CHECK(rand_below(r, 13) < 13);
  }
}

// ---------------------------------------------------------------- order book

TEST(limit_order_rests_and_sets_best) {
  OrderBook b;
  VectorSink s;
  CHECK(b.submit_limit(1, 10, BUY, 100, 5, 0, s));
  CHECK(b.submit_limit(2, 11, SELL, 105, 3, 0, s));
  CHECK_EQ(b.best_bid(), 100);
  CHECK_EQ(b.best_ask(), 105);
  CHECK_EQ(b.depth_at(100), 5);
  CHECK_EQ(b.live_orders(), 2u);
  CHECK(s.trades.empty());
}

TEST(crossing_limit_trades_at_maker_price) {
  OrderBook b;
  VectorSink s;
  b.submit_limit(1, 10, SELL, 100, 5, 0, s);
  b.submit_limit(2, 11, BUY, 103, 2, 1, s);
  CHECK_EQ(s.trades.size(), 1u);
  CHECK_EQ(s.trades[0].price, 100);
  CHECK_EQ(s.trades[0].qty, 2);
  CHECK_EQ(s.trades[0].buyer_agent, 11u);
  CHECK_EQ(s.trades[0].seller_agent, 10u);
  CHECK_EQ(s.trades[0].maker_id, 1u);
  CHECK_EQ(s.trades[0].taker_id, 2u);
  CHECK_EQ(b.resting_qty(1), 3);
  CHECK(!b.contains(2));  // fully filled taker does not rest
}

TEST(price_time_priority) {
  OrderBook b;
  VectorSink s;
  b.submit_limit(1, 1, SELL, 101, 2, 0, s);  // worse price, earlier
  b.submit_limit(2, 2, SELL, 100, 2, 1, s);  // best price, first in time
  b.submit_limit(3, 3, SELL, 100, 2, 2, s);  // best price, second in time
  b.submit_market(4, 9, BUY, 5, 3, s);
  CHECK_EQ(s.trades.size(), 3u);
  CHECK_EQ(s.trades[0].maker_id, 2u);
  CHECK_EQ(s.trades[1].maker_id, 3u);
  CHECK_EQ(s.trades[2].maker_id, 1u);
  CHECK_EQ(s.trades[2].qty, 1);
  CHECK_EQ(b.resting_qty(1), 1);
}

TEST(limit_remainder_rests_after_sweeping) {
  OrderBook b;
  VectorSink s;
  b.submit_limit(1, 1, SELL, 100, 2, 0, s);
  b.submit_limit(2, 1, SELL, 101, 2, 0, s);
  b.submit_limit(3, 2, BUY, 101, 7, 1, s);
  CHECK_EQ(s.trades.size(), 2u);
  CHECK_EQ(b.best_bid(), 101);
  CHECK_EQ(b.best_ask(), kNoAsk);
  CHECK_EQ(b.resting_qty(3), 3);
}

TEST(market_order_remainder_dropped) {
  OrderBook b;
  VectorSink s;
  b.submit_limit(1, 1, BUY, 100, 2, 0, s);
  const Qty filled = b.submit_market(2, 2, SELL, 5, 1, s);
  CHECK_EQ(filled, 2);
  CHECK_EQ(b.best_bid(), kNoBid);
  CHECK(!b.contains(2));
  CHECK_EQ(b.submit_market(3, 2, SELL, 5, 1, s), 0);  // empty book
}

TEST(cancel_updates_best_price) {
  OrderBook b;
  VectorSink s;
  b.submit_limit(1, 1, BUY, 100, 2, 0, s);
  b.submit_limit(2, 1, BUY, 98, 2, 0, s);
  CHECK(b.cancel(1));
  CHECK_EQ(b.best_bid(), 98);
  CHECK(!b.cancel(1));  // already gone
  CHECK(b.cancel(2));
  CHECK_EQ(b.best_bid(), kNoBid);
  CHECK_EQ(b.live_orders(), 0u);
}

TEST(cancel_middle_of_queue) {
  OrderBook b;
  VectorSink s;
  b.submit_limit(1, 1, SELL, 100, 1, 0, s);
  b.submit_limit(2, 2, SELL, 100, 1, 0, s);
  b.submit_limit(3, 3, SELL, 100, 1, 0, s);
  CHECK(b.cancel(2));
  b.submit_market(4, 9, BUY, 2, 1, s);
  CHECK_EQ(s.trades.size(), 2u);
  CHECK_EQ(s.trades[0].maker_id, 1u);
  CHECK_EQ(s.trades[1].maker_id, 3u);
}

TEST(modify_reduce_keeps_priority) {
  OrderBook b;
  VectorSink s;
  b.submit_limit(1, 1, SELL, 100, 5, 0, s);
  b.submit_limit(2, 2, SELL, 100, 5, 0, s);
  CHECK(b.modify(1, 100, 3, 1, s));
  CHECK_EQ(b.depth_at(100), 8);
  b.submit_market(3, 9, BUY, 1, 2, s);
  CHECK_EQ(s.trades[0].maker_id, 1u);
}

TEST(modify_increase_loses_priority) {
  OrderBook b;
  VectorSink s;
  b.submit_limit(1, 1, SELL, 100, 5, 0, s);
  b.submit_limit(2, 2, SELL, 100, 5, 0, s);
  CHECK(b.modify(1, 100, 6, 1, s));
  b.submit_market(3, 9, BUY, 1, 2, s);
  CHECK_EQ(s.trades[0].maker_id, 2u);
}

TEST(modify_price_can_cross) {
  OrderBook b;
  VectorSink s;
  b.submit_limit(1, 1, BUY, 99, 4, 0, s);
  b.submit_limit(2, 2, SELL, 101, 4, 0, s);
  CHECK(b.modify(1, 101, 4, 1, s));
  CHECK_EQ(s.trades.size(), 1u);
  CHECK_EQ(s.trades[0].price, 101);
  CHECK_EQ(b.best_bid(), kNoBid);
  CHECK_EQ(b.best_ask(), kNoAsk);
  CHECK(!b.modify(77, 100, 1, 1, s));
}

TEST(modify_to_zero_cancels) {
  OrderBook b;
  VectorSink s;
  b.submit_limit(1, 1, BUY, 99, 4, 0, s);
  CHECK(b.modify(1, 99, 0, 1, s));
  CHECK_EQ(b.live_orders(), 0u);
}

TEST(rejects_bad_orders) {
  OrderBook b;
  VectorSink s;
  CHECK(!b.submit_limit(1, 1, BUY, 100, 0, 0, s));
  CHECK(!b.submit_limit(1, 1, BUY, -5, 1, 0, s));
  CHECK(!b.submit_limit(1, 1, BUY, kNumPrices, 1, 0, s));
  CHECK(b.submit_limit(1, 1, BUY, 100, 1, 0, s));
  CHECK(!b.submit_limit(1, 1, BUY, 100, 1, 0, s));  // duplicate live id
}

TEST(expiry_cancels_old_orders) {
  OrderBook b(0, 3);
  VectorSink s;
  b.submit_limit(1, 1, BUY, 100, 1, 0, s);
  b.submit_limit(2, 1, BUY, 101, 1, 2, s);
  b.expire(2);
  CHECK_EQ(b.live_orders(), 2u);
  b.expire(3);
  CHECK(!b.contains(1));
  CHECK(b.contains(2));
  b.expire(5);
  CHECK_EQ(b.live_orders(), 0u);
}

TEST(depth_over_levels) {
  OrderBook b;
  VectorSink s;
  b.submit_limit(1, 1, BUY, 100, 1, 0, s);
  b.submit_limit(2, 1, BUY, 99, 2, 0, s);
  b.submit_limit(3, 1, BUY, 97, 4, 0, s);
  CHECK_EQ(b.depth(BUY, 2), 3);
  CHECK_EQ(b.depth(BUY, 5), 7);
  CHECK_EQ(b.depth(SELL, 5), 0);
}

TEST(random_operations_keep_book_consistent) {
  // Random add/cancel/modify/market traffic; checks totals and best prices
  // against a brute-force recount after every step.
  OrderBook b;
  VectorSink s;
  std::vector<OrderId> ids;
  for (uint32_t step = 0; step < 20000; ++step) {
    const U32x4 r = rng_draw(99, 9, step, 0);
    const OrderId id = step + 1;
    const uint32_t op = r.v[0] % 10;
    const Side side = (r.v[1] & 1) ? SELL : BUY;
    const Price p = 1000 + static_cast<Price>(r.v[2] % 40);
    const Qty q = 1 + static_cast<Qty>(r.v[3] % 9);
    if (op < 6) {
      b.submit_limit(id, 1, side, p, q, step, s);
      ids.push_back(id);
    } else if (op < 7) {
      b.submit_market(id, 1, side, q, step, s);
    } else if (op < 9 && !ids.empty()) {
      b.cancel(ids[r.v[2] % ids.size()]);
    } else if (!ids.empty()) {
      b.modify(ids[r.v[2] % ids.size()], p, q, step, s);
    }
    // Book must never be crossed.
    CHECK(b.best_bid() == kNoBid || b.best_ask() == kNoAsk || b.best_bid() < b.best_ask());
    if (step % 500 == 0) {
      Price bb = kNoBid, ba = kNoAsk;
      Qty total = 0, live_total = 0;
      for (Price x = 0; x < kNumPrices; ++x) total += b.depth_at(x);
      for (OrderId x : ids) live_total += b.resting_qty(x);
      for (Price x = 0; x < kNumPrices; ++x)
        if (b.depth_at(x) > 0) {
          if (x <= b.best_bid()) bb = x;
          if (x >= b.best_ask() && ba == kNoAsk) ba = x;
        }
      CHECK_EQ(total, live_total);
      CHECK_EQ(bb, b.best_bid());
      CHECK_EQ(ba, b.best_ask());
    }
  }
  CHECK(!s.trades.empty());
}

// ---------------------------------------------------------------- agents

TEST(decide_is_pure_and_sane) {
  Params P;
  Snapshot s{};
  s.bid = 1000; s.ask = 1004; s.ref = 1002; s.fundamental = 1010; s.spread = 4;
  s.trend_q8 = 600; s.index_trend_q8 = 0;
  int acted[kNumAgentTypes] = {0, 0, 0, 0};
  for (uint32_t a = 0; a < 20000; ++a)
    for (uint8_t ty = 0; ty < kNumAgentTypes; ++ty) {
      const Intent x = decide(a, ty, 5, 1, s, P);
      const Intent y = decide(a, ty, 5, 1, s, P);
      CHECK_EQ(x.action, y.action);
      CHECK_EQ(x.price, y.price);
      if (x.action == ACT_NONE) continue;
      ++acted[ty];
      CHECK(x.qty >= 1 && x.qty <= 4 * P.max_qty);
      if (x.action != ACT_MARKET) CHECK(x.price >= 1 && x.price < kNumPrices - 1);
      const int32_t w = P.noise_width + (s.trend_q8 / 256) * P.noise_trend_gain / 100;
      if (ty == NOISE && x.action == ACT_LIMIT) CHECK(x.price >= s.ref - w && x.price <= s.ref + w);
      if (ty == MOMENTUM) CHECK(x.action == ACT_MARKET && x.side == BUY);       // trend up
      if (ty == MEAN_REVERT) CHECK(x.action == ACT_LIMIT && x.side == BUY);     // below fundamental
      if (ty == MARKET_MAKER) CHECK(x.action == ACT_REPLACE && x.cancel_id == make_order_id(3, a));
    }
  CHECK(acted[NOISE] > 1500 && acted[NOISE] < 2500);  // ~10%
  CHECK_EQ(acted[MARKET_MAKER], 20000);
}

// ---------------------------------------------------------------- simulator

static Result run_sim(Config c) { return Simulator(c).run(); }

static Config small_config() {
  Config c;
  c.num_agents = 3000;
  c.num_instruments = 6;
  c.ticks = 300;
  c.seed = 7;
  return c;
}

TEST(simulation_produces_trades) {
  const Result r = run_sim(small_config());
  CHECK(r.num_trades > 1000);
  CHECK(r.volume >= r.num_trades);
}

TEST(serial_and_openmp_trade_logs_identical) {
  Config c = small_config();
  const Result ref = run_sim(c);
  for (int threads : {1, 2, 3, 4, 8}) {
    c.threads = threads;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    for (auto d : {DecideBackend::Serial, DecideBackend::OpenMP})
      for (auto m : {MatchBackend::Serial, MatchBackend::OpenMP}) {
        c.decide = d;
        c.match = m;
        const Result r = run_sim(c);
        CHECK_EQ(r.trade_hash, ref.trade_hash);
        CHECK_EQ(r.num_trades, ref.num_trades);
      }
  }
}

TEST(agent_sorting_does_not_change_results) {
  Config c = small_config();
  const Result a = run_sim(c);
  c.sort_by_type = false;
  const Result b = run_sim(c);
  CHECK_EQ(a.trade_hash, b.trade_hash);
}

TEST(rank_split_preserves_agents) {
  // Splitting instruments over "ranks" must partition the agents exactly.
  Config c = small_config();
  size_t total = 0;
  for (int r = 0; r < 4; ++r) total += Simulator(c, r, 4).agents().size();
  CHECK_EQ(total, static_cast<size_t>(c.num_agents));
}

TEST(different_seeds_differ) {
  Config c = small_config();
  const Result a = run_sim(c);
  c.seed = 8;
  const Result b = run_sim(c);
  CHECK(a.trade_hash != b.trade_hash);
}

#ifdef LOB_USE_CUDA
TEST(cuda_trade_log_matches_cpu) {
  Config c = small_config();
  const Result ref = run_sim(c);
  c.decide = DecideBackend::CUDA;
  try {
    for (bool compact : {true, false})
      for (bool sorted : {true, false})
        for (int block : {64, 256}) {
          c.cuda_compact = compact;
          c.sort_by_type = sorted;
          c.cuda_block = block;
          const Result r = run_sim(c);
          CHECK_EQ(r.trade_hash, ref.trade_hash);
          CHECK_EQ(r.num_trades, ref.num_trades);
        }
  } catch (const std::exception& e) {
    std::printf("  SKIP (no usable GPU): %s\n", e.what());
  }
}
#endif

int main() {
  for (auto& t : registry()) {
    const int before = g_failed;
    t.second();
    std::printf("%s %s\n", g_failed == before ? "[ OK ]" : "[FAIL]", t.first.c_str());
  }
  std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
