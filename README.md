# Stock-market-simulator-using-PCAP

**Heterogeneous parallelization of an agent-based limit order book simulator using CUDA, OpenMP and MPI.**
Mini project for Parallel Computer Architecture and Programming (CSE-3129), MIT Manipal.

Thousands of trading agents decide what to do in parallel (GPU), their orders are matched under strict
price-time priority per instrument (serial per instrument, parallel across instruments with OpenMP),
and instruments are split across processes (MPI). The point of the project is to measure where this
deliberately imperfect workload stops scaling, not to pick an easy embarrassingly parallel problem.

See [docs/DESIGN.md](docs/DESIGN.md) for how it works and how each objective is measured.

## Repository layout

| Path | What it is |
|---|---|
| `include/lob/order_book.hpp`, `src/order_book.cpp` | Serial matching engine: add, cancel, modify, market orders, price-time priority |
| `include/lob/agents.hpp` | Noise, momentum, mean-reverting and market-making strategies (shared by CPU and GPU) |
| `include/lob/philox.hpp` | Philox4x32-10 counter-based RNG (identical results on CPU and GPU) |
| `include/lob/sim.hpp`, `src/sim.cpp` | Tick loop, OpenMP matching across instruments, MPI index reduction |
| `src/decide_cuda.cu` | CUDA agent-decision kernel |
| `src/main.cpp` | `lobsim` command-line program |
| `tests/test_main.cpp` | Unit tests (order book reference + cross-back-end determinism) |
| `scripts/` | Determinism check, benchmark experiments, Nsight profiling |
| `analysis/` | Stylised facts, scaling plots with Amdahl fit, roofline |
| `notebooks/colab_run.ipynb` | One-click run of everything GPU-related on Google Colab |

## Build

Requirements: C++17 compiler with OpenMP, CMake 3.18+. Optional: CUDA toolkit (for the GPU back end),
Open MPI or MPICH (for multi-process runs), Python 3 with `pip install -r requirements.txt` (analysis).

```bash
# CPU only (laptop, WSL2 Ubuntu, lab machine)
cmake -S . -B build && cmake --build build -j
# with MPI
cmake -S . -B build -DLOB_USE_MPI=ON && cmake --build build -j
# with CUDA + MPI (Colab T4 = arch 75; RTX 30xx = 86, RTX 40xx = 89)
cmake -S . -B build -DLOB_USE_CUDA=ON -DLOB_USE_MPI=ON -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build -j

./build/lob_tests            # unit tests
```

On Windows, use WSL2 (`wsl --install -d Ubuntu-24.04`), then inside Ubuntu:
`sudo apt install -y build-essential cmake git openmpi-bin libopenmpi-dev python3-pip`.

## Run

```bash
./build/lobsim --header --agents 100000 --instruments 32 --ticks 1000 --decide omp --match omp
./build/lobsim --header --agents 1000000 --instruments 128 --ticks 500 --decide cuda --match omp
mpirun -np 4 --bind-to none ./build/lobsim --header --agents 400000 --instruments 64 --decide omp --match omp --threads 2
```

Each run prints one CSV row: configuration, trade count, a hash of the full trade log, and per-phase
times (`t_decide`, `t_h2d`, `t_kernel`, `t_d2h`, `t_match`, `t_publish`, `t_comm`). `./build/lobsim --help`
lists all options, including `--set NAME=VALUE` for strategy parameters, `--prices` and `--trades`
for CSV output.

## Live demo

`bash scripts/demo.sh build` runs a short demo on any machine: the unit tests, identical trade logs
across serial, OpenMP, MPI (and CUDA when the build and machine have it), and the serial matching core.
For the GPU, open `notebooks/colab_demo.ipynb` in Colab on a T4 runtime and run all cells.

## Reproducing the study

| Objective | Command | Output |
|---|---|---|
| 1. Correct serial matching engine | `./build/lob_tests` | unit tests incl. a randomized consistency test |
| 2. Agent population + serial baseline | `lobsim --decide serial --match serial` | baseline for all speedups |
| 3. CUDA agent phase | `scripts/bench.sh cuda` | block size, type sorting, compaction |
| 4. OpenMP matching | `scripts/bench.sh strong`, `instruments`, `falseshare` | thread scaling, false sharing |
| 5. MPI distribution | `scripts/bench.sh mpi` | communication share of tick time |
| 6. Validation | `scripts/check_determinism.sh build`, `analysis/stylized_facts.py` | identical trade hashes; heavy tails, volatility clustering |
| 7. Scaling study | `scripts/bench.sh all`, `scripts/profile.sh`, `analysis/scaling.py`, `analysis/roofline.py` | strong/weak scaling, Amdahl fit, roofline, Nsight metrics |
| 8. CPU vs GPU threshold | `scripts/bench.sh crossover` then `analysis/scaling.py` | agent count where CUDA overtakes OpenMP |

Suggested split of machines:

- **GPU experiments (Colab):** open `notebooks/colab_run.ipynb` in Colab (File > Open notebook > GitHub),
  select a T4 GPU and run all cells. It builds, tests, checks determinism, runs `bench.sh gpu`, profiles
  and downloads `results.zip`.
- **CPU experiments (a laptop with more cores):** Colab only has 2 CPU threads, so run
  `scripts/bench.sh cpu build` on a multi-core machine. For the false-sharing comparison also build
  `cmake -S . -B build-nopad -DLOB_NO_PAD=ON && cmake --build build-nopad -j`.
- Put both sets of CSVs in one `results/` folder and run `python analysis/scaling.py results`.

## Team

Abhisaar Gupta, Devesh Khaitan, Sohan Kar, Pratham Dobhal.
