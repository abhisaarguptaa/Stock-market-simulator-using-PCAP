# Design

## The tick

The simulator advances in discrete ticks. Every tick has three phases separated by barriers
(a conservative, barrier-synchronised scheme, see Fujimoto 1990):

```
            ┌──────────── parallel over agents ────────────┐
 snapshot → │ decide: each agent reads its instrument's     │ → intents
            │ snapshot, emits at most one order             │
            └───────────────────────────────────────────────┘
            ┌──── parallel over instruments, serial inside ─┐
 intents  → │ match: apply the instrument's intents to its  │ → trades
            │ order book in a deterministic arrival order   │
            └───────────────────────────────────────────────┘
            ┌──────────── reduction over instruments ───────┐
            │ publish: rebuild snapshots, reduce the        │ → snapshot (next tick)
            │ cross-instrument index (MPI_Allreduce)        │
            └───────────────────────────────────────────────┘
```

| Phase | Serial | OpenMP | CUDA | MPI |
|---|---|---|---|---|
| decide | loop | `parallel for` over agents | one thread per agent | each rank decides for its own agents |
| match | loop over instruments | `parallel for schedule(dynamic,1)` over instruments | (stays on CPU, OpenMP) | each rank owns a contiguous block of instruments |
| publish | sum of mid prices | same | same | `MPI_Allreduce` of the integer sum |

Matching inside one instrument is never parallelised: price-time priority makes it a total order.
That is the irreducible serial core the scaling study measures.

## Order book (`order_book.hpp`)

- Fixed price grid of 4096 integer ticks per instrument. One FIFO queue per price level.
- Orders live in a pool and are doubly linked into their level, with an `id -> node` hash map, so
  add, cancel and modify are O(1) except for moving the best-price pointer when a level empties.
- Limit orders match against the opposite side at the maker's price, then rest. Market orders match
  and drop any unfilled remainder. Modify keeps time priority only for a quantity decrease at the
  same price; anything else re-queues the order (and may trade).
- Resting orders expire after `--ttl` ticks (FIFO expiry queue), which keeps the book bounded.

## Agents (`agents.hpp`)

All four strategies are stateless functions of (agent id, tick, seed, snapshot), written once as
`__host__ __device__` code so the CPU loops and the CUDA kernel run literally the same logic:

| Strategy | Acts per tick | Behaviour |
|---|---|---|
| Noise | 10% | random side; mostly limit orders near the reference price (width grows with the trend, after Preis et al.), some market orders |
| Momentum | 5% | market order in the direction of the instrument trend plus half the index trend; bigger orders for stronger trends |
| Mean-reverting | 3% | limit order back toward the fundamental value when the price strays beyond a band |
| Market maker | every tick | re-quotes one side per tick (alternating), cancelling its own quote from two ticks earlier |

Decisions use integer arithmetic only. Floating point on CPU and GPU can round differently (fused
multiply-add, different math libraries), which would break exact trade-log comparison.

Order ids are `(tick + 1) << 32 | agent`, so a market maker on the GPU can compute the id of the
quote it wants to cancel without any feedback from the matching engine.

## Determinism

The trade log is identical across all back ends, thread counts, block sizes and MPI rank counts
for a given seed, and this is checked by `scripts/check_determinism.sh` and the unit tests via an
order-sensitive FNV-1a hash of every trade. It works because:

1. **Counter-based RNG.** Philox4x32-10 maps (seed, stream, agent, tick) to random numbers with no
   shared state, so the order in which threads run cannot change any draw.
2. **Integer-only decisions** (see above).
3. **Deterministic arrival order.** Within an instrument, intents are applied in agent-id order rotated
   by a random offset drawn from (seed, instrument, tick), so no agent is always first in the queue.
4. **Integer reductions.** The cross-instrument index is a sum of integers, so `MPI_Allreduce` gives the
   same answer whatever the reduction order.
5. **Order-independent GPU compaction.** The kernel copies back only agents that acted, reserving
   output slots with `atomicAdd`. The slot order varies run to run, but each entry carries its agent
   index and the host scatters by index, so the result does not depend on it.

The costs of these choices (the extra scatter, integer-only maths, the barrier per tick) are part of
what the study measures.

## GPU kernel choices (`decide_cuda.cu`)

- **Structure of arrays** for agent id, type and instrument: consecutive threads load consecutive
  words (coalesced).
- **Agents sorted by strategy** (`--no-sort` turns this off): threads in a warp mostly take the same
  branch, reducing divergence. Measure with Nsight Compute's "threads per warp instruction".
- **Snapshots** are 32 bytes per instrument, read through the read-only cache (`__ldg`).
- **Parameters** in constant memory.
- **Compaction** (`--no-compact` turns it off): only about 20% of agents act per tick, so copying back
  only their intents cuts device-to-host traffic. This is the CPU–GPU boundary cost.

## Measurement

`lobsim` times each phase per tick and reports totals (for MPI, the slowest rank). CUDA timings are
split into host-to-device copy, kernel (CUDA events) and device-to-host copy. The scripts in
`scripts/` run the experiments; `analysis/scaling.py` turns them into plots and fits Amdahl's law to
the strong-scaling speedups to estimate the serial fraction, and also reports Karp–Flatt values.
`analysis/roofline.py` places the kernel on a T4 integer roofline from Nsight Compute counters.

## Known limitations

- Agents do not track inventory or cash; fills are not fed back to them. This keeps the GPU phase free
  of per-tick feedback from the matching engine.
- Colab provides one GPU and 2 CPU threads, so multi-node MPI and high thread counts need other
  hardware; on one machine, MPI ranks measure the communication cost but not network latency.
- The price grid is fixed at 4096 ticks; prices are clamped at its edges.
- Strategy parameters were tuned by hand so that returns show heavy tails and volatility clustering.
  Results vary by seed; `analysis/stylized_facts.py` reports them for any run.
