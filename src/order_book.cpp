#include "lob/order_book.hpp"

#include <algorithm>

namespace lob {

OrderBook::OrderBook(int32_t instrument, int64_t ttl_ticks)
    : instrument_(instrument), ttl_(ttl_ticks), levels_(kNumPrices) {
  pool_.reserve(1024);
  index_.reserve(1024);
}

int32_t OrderBook::alloc_node() {
  if (!free_.empty()) {
    const int32_t n = free_.back();
    free_.pop_back();
    return n;
  }
  pool_.push_back(Node{});
  return static_cast<int32_t>(pool_.size() - 1);
}

void OrderBook::refresh_best_bid(Price from) {
  Price p = from;
  while (p >= 0 && levels_[p].total == 0) --p;
  best_bid_ = p < 0 ? kNoBid : p;
}

void OrderBook::refresh_best_ask(Price from) {
  Price p = from;
  while (p < kNumPrices && levels_[p].total == 0) ++p;
  best_ask_ = p >= kNumPrices ? kNoAsk : p;
}

void OrderBook::unlink(int32_t n) {
  Node& node = pool_[n];
  Level& lv = levels_[node.price];
  if (node.prev >= 0) pool_[node.prev].next = node.next; else lv.head = node.next;
  if (node.next >= 0) pool_[node.next].prev = node.prev; else lv.tail = node.prev;
  lv.total -= node.qty;
  index_.erase(node.id);
  free_.push_back(n);
  if (lv.total == 0) {
    if (node.side == BUY && node.price == best_bid_) refresh_best_bid(node.price);
    if (node.side == SELL && node.price == best_ask_) refresh_best_ask(node.price);
  }
}

Qty OrderBook::match(Side taker_side, Price limit, Qty qty, OrderId taker_id,
                     uint32_t taker_agent, int64_t tick, TradeSink& sink) {
  while (qty > 0) {
    Price p;
    if (taker_side == BUY) {
      p = best_ask_;
      if (p == kNoAsk || p > limit) break;
    } else {
      p = best_bid_;
      if (p == kNoBid || p < limit) break;
    }
    Level& lv = levels_[p];
    const int32_t n = lv.head;  // oldest order at the best price
    Node& maker = pool_[n];
    const Qty fill = std::min(qty, maker.qty);
    Trade t;
    t.tick = tick;
    t.instrument = instrument_;
    t.price = p;
    t.qty = fill;
    t.taker_side = taker_side;
    t.buyer_agent = taker_side == BUY ? taker_agent : maker.agent;
    t.seller_agent = taker_side == BUY ? maker.agent : taker_agent;
    t.maker_id = maker.id;
    t.taker_id = taker_id;
    sink.on_trade(t);
    qty -= fill;
    if (fill == maker.qty) {
      unlink(n);
    } else {
      maker.qty -= fill;
      lv.total -= fill;
    }
  }
  return qty;
}

void OrderBook::rest(OrderId id, uint32_t agent, Side side, Price price, Qty qty,
                     int64_t tick) {
  const int32_t n = alloc_node();
  Level& lv = levels_[price];
  Node& node = pool_[n];
  node = Node{id, agent, price, qty, side, lv.tail, -1};
  if (lv.tail >= 0) pool_[lv.tail].next = n; else lv.head = n;
  lv.tail = n;
  lv.total += qty;
  index_.emplace(id, n);
  if (side == BUY && price > best_bid_) best_bid_ = price;
  if (side == SELL && price < best_ask_) best_ask_ = price;
  if (ttl_ > 0) expiry_.emplace_back(tick + ttl_, id);
}

bool OrderBook::submit_limit(OrderId id, uint32_t agent, Side side, Price price,
                             Qty qty, int64_t tick, TradeSink& sink) {
  if (qty <= 0 || price < 0 || price >= kNumPrices || index_.count(id)) return false;
  const Qty left = match(side, price, qty, id, agent, tick, sink);
  if (left > 0) rest(id, agent, side, price, left, tick);
  return true;
}

Qty OrderBook::submit_market(OrderId id, uint32_t agent, Side side, Qty qty,
                             int64_t tick, TradeSink& sink) {
  if (qty <= 0) return 0;
  const Price limit = side == BUY ? kNumPrices : -1;
  return qty - match(side, limit, qty, id, agent, tick, sink);
}

bool OrderBook::cancel(OrderId id) {
  auto it = index_.find(id);
  if (it == index_.end()) return false;
  unlink(it->second);
  return true;
}

bool OrderBook::modify(OrderId id, Price new_price, Qty new_qty, int64_t tick,
                       TradeSink& sink) {
  auto it = index_.find(id);
  if (it == index_.end()) return false;
  const int32_t n = it->second;
  Node& node = pool_[n];
  if (new_qty <= 0) {
    unlink(n);
    return true;
  }
  if (new_price == node.price && new_qty <= node.qty) {
    levels_[node.price].total -= node.qty - new_qty;
    node.qty = new_qty;
    return true;
  }
  const uint32_t agent = node.agent;
  const Side side = node.side;
  unlink(n);
  submit_limit(id, agent, side, new_price, new_qty, tick, sink);
  return true;
}

void OrderBook::expire(int64_t tick) {
  while (!expiry_.empty() && expiry_.front().first <= tick) {
    cancel(expiry_.front().second);  // no-op if already filled or cancelled
    expiry_.pop_front();
  }
}

Qty OrderBook::depth_at(Price p) const {
  return (p < 0 || p >= kNumPrices) ? 0 : levels_[p].total;
}

Qty OrderBook::depth(Side side, int levels) const {
  Qty total = 0;
  int seen = 0;
  if (side == BUY) {
    for (Price p = best_bid_; p >= 0 && seen < levels; --p)
      if (levels_[p].total > 0) { total += levels_[p].total; ++seen; }
  } else {
    for (Price p = best_ask_; p < kNumPrices && seen < levels; ++p)
      if (levels_[p].total > 0) { total += levels_[p].total; ++seen; }
  }
  return total;
}

Qty OrderBook::resting_qty(OrderId id) const {
  auto it = index_.find(id);
  return it == index_.end() ? 0 : pool_[it->second].qty;
}

}  // namespace lob
