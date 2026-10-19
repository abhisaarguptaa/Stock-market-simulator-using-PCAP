// Philox4x32-10 counter-based RNG (Salmon et al., SC'11), written as plain
// host/device code so the CPU and GPU produce bit-identical streams.
// A stream is identified by a 64-bit key; each counter value gives 4 random
// 32-bit words. No state is stored, so any (agent, tick) draw is independent
// of execution order, which is what makes parallel runs reproducible.
#pragma once
#include "lob/common.hpp"

namespace lob {

struct U32x4 {
  uint32_t v[4];
};

LOB_HD void philox_mulhilo(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) {
  const uint64_t p = static_cast<uint64_t>(a) * b;
  hi = static_cast<uint32_t>(p >> 32);
  lo = static_cast<uint32_t>(p);
}

LOB_HD U32x4 philox4x32_10(U32x4 c, uint32_t k0, uint32_t k1) {
  for (int round = 0; round < 10; ++round) {
    if (round > 0) {
      k0 += 0x9E3779B9u;
      k1 += 0xBB67AE85u;
    }
    uint32_t hi0, lo0, hi1, lo1;
    philox_mulhilo(0xD2511F53u, c.v[0], hi0, lo0);
    philox_mulhilo(0xCD9E8D57u, c.v[2], hi1, lo1);
    U32x4 n;
    n.v[0] = hi1 ^ c.v[1] ^ k0;
    n.v[1] = lo1;
    n.v[2] = hi0 ^ c.v[3] ^ k1;
    n.v[3] = lo0;
    c = n;
  }
  return c;
}

// Convenience: draw 4 words for (seed, stream, a, b, c).
LOB_HD U32x4 rng_draw(uint64_t seed, uint32_t stream, uint32_t a, uint32_t b,
                      uint32_t c = 0) {
  U32x4 ctr;
  ctr.v[0] = a;
  ctr.v[1] = b;
  ctr.v[2] = c;
  ctr.v[3] = stream;
  return philox4x32_10(ctr, static_cast<uint32_t>(seed),
                       static_cast<uint32_t>(seed >> 32));
}

// Unbiased-enough integer in [0, n) from one 32-bit word (Lemire's method,
// without the rejection step; bias is < n / 2^32).
LOB_HD uint32_t rand_below(uint32_t r, uint32_t n) {
  return static_cast<uint32_t>((static_cast<uint64_t>(r) * n) >> 32);
}

// Stream ids so different uses of the RNG never overlap.
enum RngStream : uint32_t {
  kStreamAgent = 1,
  kStreamAgentType = 2,
  kStreamFundamental = 3,
  kStreamArrival = 4,
};

}  // namespace lob
