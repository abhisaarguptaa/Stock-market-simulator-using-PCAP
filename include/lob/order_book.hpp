// Single-instrument limit order book with price-time priority.
//
// Layout: one FIFO queue per price tick on a fixed grid. Orders live in a pool
// and are linked into their level's queue, so add, cancel and modify are O(1)
// apart from moving the best-price pointer when a level empties.
//
// The book itself is strictly serial: all parallelism happens across books.
#pragma once
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lob/common.hpp"

namespace lob {

struct Trade {
  int64_t tick;
  int32_t instrument;
  Price price;
  Qty qty;
  Side taker_side;
  uint32_t buyer_agent;
  uint32_t seller_agent;
  OrderId maker_id;
  OrderId taker_id;
};

// Receives every trade the book produces. Kept as a tiny interface so the
// simulator can hash trades without storing them.
class TradeSink {
 public:
  virtual ~TradeSink() = default;
  virtual void on_trade(const Trade& t) = 0;
};

class VectorSink : public TradeSink {
 public:
  std::vector<Trade> trades;
  void on_trade(const Trade& t) override { trades.push_back(t); }
};

class OrderBook {
 public:
  explicit OrderBook(int32_t instrument = 0, int64_t ttl_ticks = 0);

  // Limit order: matches against the opposite side, then rests any remainder.
  // Returns false (and does nothing) if the id is already live or qty <= 0.
  bool submit_limit(OrderId id, uint32_t agent, Side side, Price price, Qty qty,
                    int64_t tick, TradeSink& sink);

  // Market order: matches immediately; any unfilled remainder is dropped.
  // Returns the filled quantity.
  Qty submit_market(OrderId id, uint32_t agent, Side side, Qty qty, int64_t tick,
                    TradeSink& sink);

  // Removes a resting order. Returns false if it is not in the book.
  bool cancel(OrderId id);

  // Changes a resting order. Reducing quantity at the same price keeps time
  // priority; any price change or quantity increase re-queues it at the back
  // (and may trade if the new price crosses). Returns false if not found.
  bool modify(OrderId id, Price new_price, Qty new_qty, int64_t tick, TradeSink& sink);

  // Cancels resting orders whose time-to-live has passed (no-op when ttl == 0).
  void expire(int64_t tick);

  Price best_bid() const { return best_bid_; }
  Price best_ask() const { return best_ask_; }
  Qty depth_at(Price p) const;
  Qty depth(Side side, int levels) const;  // total qty in the best `levels` levels
  size_t live_orders() const { return index_.size(); }
  bool contains(OrderId id) const { return index_.count(id) != 0; }
  Qty resting_qty(OrderId id) const;
  int32_t instrument() const { return instrument_; }

 private:
  struct Node {
    OrderId id;
    uint32_t agent;
    Price price;
    Qty qty;
    Side side;
    int32_t prev, next;
  };
  struct Level {
    int32_t head = -1, tail = -1;
    Qty total = 0;
  };

  Qty match(Side taker_side, Price limit, Qty qty, OrderId taker_id,
            uint32_t taker_agent, int64_t tick, TradeSink& sink);
  void rest(OrderId id, uint32_t agent, Side side, Price price, Qty qty, int64_t tick);
  void unlink(int32_t n);
  int32_t alloc_node();
  void refresh_best_bid(Price from);
  void refresh_best_ask(Price from);

  int32_t instrument_;
  int64_t ttl_;
  std::vector<Level> levels_;
  std::vector<Node> pool_;
  std::vector<int32_t> free_;
  std::unordered_map<OrderId, int32_t> index_;
  std::deque<std::pair<int64_t, OrderId>> expiry_;  // (expire tick, id), FIFO
  Price best_bid_ = kNoBid;
  Price best_ask_ = kNoAsk;
};

}  // namespace lob
