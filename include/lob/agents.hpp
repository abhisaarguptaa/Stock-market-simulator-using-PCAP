// Agent decision logic, shared verbatim by the CPU and CUDA back ends.
//
// Every agent reads a read-only snapshot of its instrument's book and emits at
// most one intent per tick. Decisions use integer arithmetic only, so the CPU
// and GPU produce identical intents (no floating-point rounding differences).
#pragma once
#include "lob/common.hpp"
#include "lob/philox.hpp"

namespace lob {

enum AgentType : uint8_t {
  NOISE = 0,
  MOMENTUM = 1,
  MEAN_REVERT = 2,
  MARKET_MAKER = 3,
  kNumAgentTypes = 4
};

// What an agent sees at the start of a tick (32 bytes, one per instrument).
struct Snapshot {
  Price bid;            // best bid or kNoBid
  Price ask;            // best ask or kNoAsk
  Price ref;            // reference price: mid if both sides exist, else last trade
  Price fundamental;    // instrument's fundamental value (random walk)
  int32_t trend_q8;     // ref - EMA(ref), in 1/256 ticks
  int32_t index_trend_q8;  // same for the cross-instrument index
  int32_t spread;       // ask - bid, or 0 if one side is empty
  int32_t pad;
};

enum Action : uint8_t { ACT_NONE = 0, ACT_LIMIT = 1, ACT_MARKET = 2, ACT_REPLACE = 3 };

// One agent's output for one tick (24 bytes).
struct Intent {
  uint8_t action;
  uint8_t side;
  uint16_t pad;
  Price price;
  Qty qty;
  uint32_t pad2;
  OrderId cancel_id;  // for ACT_REPLACE: order to cancel before placing the new one
};

// Strategy parameters. Probabilities are in units of 1/65536.
struct Params {
  uint32_t p_act[kNumAgentTypes] = {6554, 3277, 2000, 65536};  // 10%, 5%, 3%, 100%
  int32_t noise_width = 12;       // max distance from ref for noise limit orders
  int32_t noise_trend_gain = 150;  // widen noise orders by gain% of |trend| (Preis et al. coupling)
  int32_t noise_market_pct = 8;   // share of noise orders sent as market orders
  int32_t max_qty = 5;
  int32_t mom_threshold_q8 = 256; // momentum acts when |trend| > 1 tick
  int32_t mom_scale = 64;         // extra qty = |signal_q8| / mom_scale (0 = off)
  int32_t mr_band = 2;            // mean-reverters act when |ref - fundamental| > band
  int32_t mm_half_spread = 2;
  int32_t mm_lean_div = 8;        // quotes move 1/div of the way to fundamental (0 = off)
};

LOB_HD Intent no_intent() {
  Intent it;
  it.action = ACT_NONE;
  it.side = 0;
  it.pad = 0;
  it.price = 0;
  it.qty = 0;
  it.pad2 = 0;
  it.cancel_id = 0;
  return it;
}

LOB_HD int32_t iabs(int32_t x) { return x < 0 ? -x : x; }

LOB_HD Intent decide(uint32_t agent, uint8_t type, int64_t tick, uint64_t seed,
                     const Snapshot& s, const Params& P) {
  const U32x4 r = rng_draw(seed, kStreamAgent, agent, static_cast<uint32_t>(tick),
                           static_cast<uint32_t>(tick >> 32));
  Intent it = no_intent();
  if (rand_below(r.v[0], 65536u) >= P.p_act[type]) return it;

  const Qty qty = 1 + static_cast<Qty>(rand_below(r.v[3], static_cast<uint32_t>(P.max_qty)));

  switch (type) {
    case NOISE: {
      const Side side = (r.v[1] & 1u) ? SELL : BUY;
      it.side = side;
      it.qty = qty;
      if (rand_below(r.v[1] >> 1, 100u) < static_cast<uint32_t>(P.noise_market_pct)) {
        it.action = ACT_MARKET;
      } else {
        // Distance from ref = min of two uniforms, so orders cluster near the touch.
        const int32_t extra = (iabs(s.trend_q8) / 256) * P.noise_trend_gain / 100;
        const uint32_t w = static_cast<uint32_t>(P.noise_width + extra);
        const uint32_t a = ((r.v[2] & 0xFFFFu) * w) >> 16;
        const uint32_t b = ((r.v[2] >> 16) * w) >> 16;
        const int32_t off = static_cast<int32_t>(a < b ? a : b);
        it.action = ACT_LIMIT;
        it.price = clamp_price(side == BUY ? s.ref - off : s.ref + off);
      }
      break;
    }
    case MOMENTUM: {
      const int32_t signal = s.trend_q8 + s.index_trend_q8 / 2;
      if (iabs(signal) <= P.mom_threshold_q8) break;
      it.action = ACT_MARKET;
      it.side = signal > 0 ? BUY : SELL;
      // Stronger trends -> bigger orders (capped), which feeds volatility back.
      it.qty = qty + (P.mom_scale ? iabs(signal) / P.mom_scale : 0);
      if (it.qty > 4 * P.max_qty) it.qty = 4 * P.max_qty;
      break;
    }
    case MEAN_REVERT: {
      const int32_t dev = s.ref - s.fundamental;
      if (iabs(dev) <= P.mr_band) break;
      // Price is too high: sell at the reference price; too low: buy there.
      it.action = ACT_LIMIT;
      it.side = dev > 0 ? SELL : BUY;
      it.price = clamp_price(s.ref);
      it.qty = qty;
      break;
    }
    case MARKET_MAKER: {
      // Requotes one side per tick, alternating, replacing its quote on that
      // side from two ticks ago. Quotes lean toward the fundamental value.
      const Side side = ((static_cast<uint32_t>(tick) + agent) & 1u) ? SELL : BUY;
      const int32_t half = P.mm_half_spread + static_cast<int32_t>(rand_below(r.v[2], 2u));
      const int32_t centre =
          s.ref + (P.mm_lean_div > 0 ? (s.fundamental - s.ref) / P.mm_lean_div : 0);
      it.action = ACT_REPLACE;
      it.side = side;
      it.price = clamp_price(side == BUY ? centre - half : centre + half);
      it.qty = qty;
      it.cancel_id = tick >= 2 ? make_order_id(tick - 2, agent) : 0;
      break;
    }
    default:
      break;
  }
  return it;
}

// Assigns an agent's strategy from the configured mix (fractions in 1/65536).
LOB_HD uint8_t assign_type(uint32_t agent, uint64_t seed, const uint32_t cum_mix[kNumAgentTypes]) {
  const U32x4 r = rng_draw(seed, kStreamAgentType, agent, 0);
  const uint32_t u = rand_below(r.v[0], 65536u);
  for (uint8_t t = 0; t < kNumAgentTypes; ++t)
    if (u < cum_mix[t]) return t;
  return NOISE;
}

}  // namespace lob
