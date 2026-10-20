# Heterogeneous Parallelization of an Agent-Based Limit Order Book Simulator Using CUDA, OpenMP, and MPI

Mini project for **Parallel Computer Architecture and Programming (CSE-3129)**, Manipal Institute of Technology.
Team: Abhisaar Gupta, Devesh Khaitan, Sohan Kar, Pratham Dobhal.

This program simulates a stock exchange. Thousands of computer traders ("agents") decide every round
whether to buy or sell, and a matching engine turns their orders into trades, first come first served
at each price, like a real exchange. The agents' decisions run in parallel on a GPU (CUDA) or on CPU
cores (OpenMP), the matching runs on CPU cores with one stock per core, and stocks can be split across
several processes (MPI). Every version produces exactly the same trades.

The final report is in [`report/`](report/) (IEEE and Springer formats). How the code works is in
[`docs/DESIGN.md`](docs/DESIGN.md).

---

## Quick start: run it on any laptop (no GPU needed)

You need Ubuntu. On Windows, Ubuntu runs inside Windows through WSL.

**Step 1. Get Ubuntu (Windows only, one time).**
Right-click the Start button, open **Terminal (Admin)** and type:

```
wsl --install -d Ubuntu-24.04
```

Restart when asked. An Ubuntu window opens and asks you to choose a username and password.
Next time, open it by pressing the Windows key and typing **Ubuntu**, or by typing `wsl` in a terminal.
(On a Linux computer, just open a terminal.)

**Step 2. Download the project.** In the Ubuntu window, type these lines one at a time and press Enter after each:

```
cd ~
sudo apt install -y git
git clone https://github.com/abhisaarguptaa/Stock-market-simulator-using-PCAP.git lob
cd lob
```

**Step 3. Install, build and test, in one command.**

```
bash scripts/setup_ubuntu.sh
```

It asks for your Ubuntu password once, installs the compiler, CMake and MPI, builds the program and
runs the tests. The last line should say `259597 checks, 0 failed`.

**Step 4. Run the demo.**

```
bash scripts/demo.sh build
```

It takes a few minutes and prints three results (explained in the next section).

> **Tip:** if a long command gets split in two when you paste it, type it by hand instead.
> If you see `command not found`, the previous step probably did not finish: run it again.

## What the demo shows

1. **The tests pass.** About 260,000 automatic checks of the order book, the random number generator,
   and every parallel version against the serial one.
2. **Every version makes the same trades.** Serial, OpenMP, MPI (and CUDA on a GPU) print the same
   *trade-log fingerprint*, a code computed from every single trade. Same fingerprint = same trades.
   The time column shows how much faster the parallel versions are.
3. **Matching is the bottleneck.** With 1 stock, more cores give no speedup, because one stock's orders
   must be matched one at a time. With 64 stocks, different cores match different stocks, so it gets
   several times faster.

## Run it on a GPU (free, in the browser with Google Colab)

1. Open the demo notebook:
   [colab_demo.ipynb](https://colab.research.google.com/github/abhisaarguptaa/Stock-market-simulator-using-PCAP/blob/main/notebooks/colab_demo.ipynb)
2. Click **Runtime > Change runtime type**, choose **T4 GPU**, click **Save**.
3. Click **Runtime > Run all**. If Colab warns that the notebook is not from Google, click **Run anyway**.
4. Wait for the first cell to build the code (a few minutes). The second cell runs the same demo with a
   fourth line, **CUDA GPU**, and the same fingerprint. The last cell draws a chart of simulated prices.

To repeat every GPU experiment in the report (takes much longer, profiles with Nsight and downloads all
results), use [colab_run.ipynb](https://colab.research.google.com/github/abhisaarguptaa/Stock-market-simulator-using-PCAP/blob/main/notebooks/colab_run.ipynb)
the same way.

## Run your own simulation

```
./build/lobsim --header --agents 100000 --instruments 32 --ticks 500 --decide omp --match omp
```

| Option | Meaning |
|---|---|
| `--agents N` | number of traders (default 10000) |
| `--instruments M` | number of stocks (default 8) |
| `--ticks T` | number of rounds to simulate (default 1000) |
| `--decide serial\|omp\|cuda` | where agents decide: one core, all cores, or the GPU |
| `--match serial\|omp` | matching on one core or one stock per core |
| `--threads K` | number of CPU threads |
| `--seed S` | random seed; the same seed always gives the same trades |
| `--prices FILE` / `--trades FILE` | save prices or every trade to a CSV file |

The program prints one CSV line: the settings, the number of trades, the trade-log fingerprint
(`hash`) and the time spent in each phase (`t_decide`, `t_match`, ...). `./build/lobsim --help` lists
every option. With MPI: `mpirun -np 2 --bind-to none ./build/lobsim --agents 100000 --decide omp --match omp`.

## Reproduce every result in the report

| Objective | Command | Output |
|---|---|---|
| 1. Correct serial matching engine | `./build/lob_tests` | unit tests incl. a randomized consistency test |
| 2. Agent population + serial baseline | `./build/lobsim --decide serial --match serial` | baseline for all speedups |
| 3. CUDA agent phase | `bash scripts/bench.sh cuda build` (on a GPU) | block size, type sorting, compaction |
| 4. OpenMP matching | `bash scripts/bench.sh strong build` (also `instruments`, `falseshare`) | thread scaling, false sharing |
| 5. MPI distribution | `bash scripts/bench.sh mpi build` | communication share of tick time |
| 6. Validation | `bash scripts/check_determinism.sh build`, `python3 analysis/stylized_facts.py` | identical trade hashes; heavy tails, volatility clustering |
| 7. Scaling study | `bash scripts/bench.sh all build`, `scripts/profile.sh`, `analysis/scaling.py`, `analysis/roofline.py` | strong/weak scaling, Amdahl fit, roofline, Nsight metrics |
| 8. CPU vs GPU threshold | `bash scripts/bench.sh crossover build` (on a GPU), then `analysis/scaling.py` | agent count where CUDA overtakes OpenMP |

- The CSVs we measured are in [`results/`](results/): `colab-t4/` (GPU experiments on a Tesla T4) and
  `laptop/` (CPU experiments on a 16-thread laptop). Plots: `pip install -r requirements.txt`, then
  `python3 analysis/scaling.py results/laptop`.
- For the false-sharing test, also build without padding:
  `cmake -S . -B build-nopad -DLOB_NO_PAD=ON` and then `cmake --build build-nopad -j`.
- For fair timings: plug the laptop in, set the power mode to best performance, close other apps.

## Repository layout

| Path | What it is |
|---|---|
| `include/lob/order_book.hpp`, `src/order_book.cpp` | Serial matching engine: add, cancel, modify, market orders, price-time priority |
| `include/lob/agents.hpp` | Noise, momentum, mean-reverting and market-making strategies (shared by CPU and GPU) |
| `include/lob/philox.hpp` | Philox4x32-10 counter-based random numbers (identical on CPU and GPU) |
| `include/lob/sim.hpp`, `src/sim.cpp` | Tick loop, OpenMP matching across instruments, MPI reduction |
| `src/decide_cuda.cu` | CUDA agent-decision kernel |
| `src/main.cpp` | `lobsim` command-line program |
| `tests/test_main.cpp` | Unit tests |
| `scripts/` | Setup, demo, determinism check, benchmarks, Nsight profiling |
| `analysis/` | Plots: scaling with Amdahl fit, roofline, stylised facts |
| `notebooks/` | Google Colab notebooks for the GPU |
| `results/` | Measured CSVs used in the report |
| `report/` | Final report (IEEE and Springer LaTeX sources and PDFs) |

## Troubleshooting

| Problem | Fix |
|---|---|
| `command not found` | The previous step did not finish. Run `bash scripts/setup_ubuntu.sh` again. |
| `No such file or directory` | You are in the wrong folder. Type `cd ~/lob`. |
| A pasted command breaks into two lines | Type it by hand. |
| The MPI line is missing in the demo | MPI is not installed or not built in. Run `bash scripts/setup_ubuntu.sh`. |
| `CUDA GPU` is skipped | Normal on a computer without an NVIDIA GPU. Use the Colab notebook for the GPU. |
| Times jump around between runs | Plug the laptop in, choose best performance power mode, close other apps. |

## Building by hand (optional)

```
cmake -S . -B build -DLOB_USE_MPI=ON
cmake --build build -j
```

With an NVIDIA GPU and the CUDA toolkit, add `-DLOB_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75`
(75 = Tesla T4; RTX 30xx = 86, RTX 40xx = 89).
