// Basic types shared by host (C++) and device (CUDA) code.
#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define LOB_HD __host__ __device__ __forceinline__
#else
#define LOB_HD inline
#endif

namespace lob {

using Price = int32_t;    // integer price ticks
using Qty = int32_t;
using OrderId = uint64_t; // 0 means "no order"

enum Side : uint8_t { BUY = 0, SELL = 1 };

// Every instrument uses the same fixed price grid [0, kNumPrices).
constexpr Price kNumPrices = 4096;
constexpr Price kInitPrice = 2048;
constexpr Price kNoBid = -1;          // best_bid when the bid side is empty
constexpr Price kNoAsk = kNumPrices;  // best_ask when the ask side is empty

LOB_HD Price clamp_price(Price p) {
  return p < 1 ? 1 : (p > kNumPrices - 2 ? kNumPrices - 2 : p);
}

// Order ids are derived from (tick, agent) so the GPU can compute the id of an
// agent's earlier order without any feedback from the matching engine.
// Each agent submits at most one new order per tick.
LOB_HD OrderId make_order_id(int64_t tick, uint32_t agent) {
  return (static_cast<OrderId>(tick + 1) << 32) | agent;
}

}  // namespace lob
