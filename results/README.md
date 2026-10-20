# Measured results

These are the raw measurements behind every table and plot in the report. Each row is one run of
`lobsim`; the columns are the settings, the number of trades, the trade-log fingerprint (`hash`) and
the time in seconds spent in each phase (`t_decide`, `t_kernel`, `t_d2h`, `t_match`, `t_comm`, ...).

## `colab-t4/`: Google Colab, NVIDIA Tesla T4, 2 CPU threads (2026-10-19)

Made with `notebooks/colab_run.ipynb`. Each configuration ran 3 times.

| File | Experiment | Report section |
|---|---|---|
| `crossover.csv` | serial vs OpenMP vs CUDA, 1k to 2M agents | CPU vs GPU crossover |
| `cuda_block.csv` | CUDA block size 32 to 1024 | kernel experiments |
| `cuda_ablation.csv` | type sorting and compaction on/off | kernel experiments |
| `gustafson.csv` | 32k to 2M agents at 32 instruments | Gustafson's law |
| `profile/` | Nsight Compute and Nsight Systems summaries | divergence, occupancy, roofline |
| `stylized_facts.json` | heavy tails, volatility clustering | model realism |

## `laptop/`: 16-thread laptop, Windows 11 with WSL2 Ubuntu (2026-10-19)

Made with `scripts/bench.sh`.

| File | Experiment | Runs |
|---|---|---|
| `strong.csv` | strong scaling, 1 to 16 threads | 5 rounds, interleaved, plugged in, best performance mode |
| `weak.csv` | weak scaling, 50k agents and 8 instruments per thread | 5 rounds, interleaved |
| `instruments.csv` | 1 to 128 instruments, serial vs OpenMP matching | 3 runs, serial and OpenMP back to back |
| `mpi.csv` | 1, 2 and 4 MPI ranks | 3 runs |
| `falseshare.csv` | padded vs unpadded counters (`build-nopad`) | 3 runs |

A first strong and weak scaling run on the laptop was not interleaved and gave unreliable times
(serial 22 s against 36 s for 1-thread OpenMP doing the same work), so it was replaced by the files
above.

## Make the plots again

```
pip install -r requirements.txt
python3 analysis/scaling.py results/laptop
python3 analysis/scaling.py results/colab-t4
```
