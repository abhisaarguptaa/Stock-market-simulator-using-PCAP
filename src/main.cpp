// Command-line driver. Prints one CSV row of results (and a header with
// --header), so benchmark scripts can append runs to a single file.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "lob/sim.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef LOB_USE_MPI
#include <mpi.h>
#endif

using namespace lob;

namespace {

void usage() {
  std::printf(
      "usage: lobsim [options]\n"
      "  --agents N          number of agents (default 10000)\n"
      "  --instruments M     number of instruments (default 8)\n"
      "  --ticks T           number of ticks (default 1000)\n"
      "  --seed S            RNG seed (default 42)\n"
      "  --decide B          serial | omp | cuda   (agent decision phase)\n"
      "  --match B           serial | omp          (matching phase)\n"
      "  --threads K         OpenMP threads (default: OMP_NUM_THREADS)\n"
      "  --ttl K             resting order lifetime in ticks, 0 = forever (default 100)\n"
      "  --mix a,b,c,d       percent noise,momentum,meanrev,marketmaker (default 65,15,10,10)\n"
      "  --no-sort           do not group agents by strategy\n"
      "  --block K           CUDA block size (default 256)\n"
      "  --no-compact        CUDA: copy back every agent's intent, not just active ones\n"
      "  --trades FILE       write every trade to a CSV\n"
      "  --prices FILE       write per-tick reference prices to a CSV\n"
      "  --set NAME=VALUE    override a strategy parameter (see Params in agents.hpp):\n"
      "                      p_noise p_mom p_mr p_mm (per 65536), noise_width,\n"
      "                      noise_market_pct, max_qty, mom_threshold_q8, mr_band,\n"
      "                      mm_half_spread, mm_lean_div, noise_trend_gain, mom_scale\n"
      "  --label TEXT        free-text label copied into the output row\n"
      "  --header            print the CSV header first\n");
}

DecideBackend parse_decide(const std::string& s) {
  if (s == "serial") return DecideBackend::Serial;
  if (s == "omp") return DecideBackend::OpenMP;
  if (s == "cuda") return DecideBackend::CUDA;
  throw std::invalid_argument("unknown --decide " + s);
}

MatchBackend parse_match(const std::string& s) {
  if (s == "serial") return MatchBackend::Serial;
  if (s == "omp") return MatchBackend::OpenMP;
  throw std::invalid_argument("unknown --match " + s);
}

void set_param(Params& P, const std::string& kv) {
  const size_t eq = kv.find('=');
  if (eq == std::string::npos) throw std::invalid_argument("--set needs NAME=VALUE");
  const std::string k = kv.substr(0, eq);
  const long v = std::stol(kv.substr(eq + 1));
  if (k == "p_noise") P.p_act[NOISE] = static_cast<uint32_t>(v);
  else if (k == "p_mom") P.p_act[MOMENTUM] = static_cast<uint32_t>(v);
  else if (k == "p_mr") P.p_act[MEAN_REVERT] = static_cast<uint32_t>(v);
  else if (k == "p_mm") P.p_act[MARKET_MAKER] = static_cast<uint32_t>(v);
  else if (k == "noise_width") P.noise_width = static_cast<int32_t>(v);
  else if (k == "noise_market_pct") P.noise_market_pct = static_cast<int32_t>(v);
  else if (k == "max_qty") P.max_qty = static_cast<int32_t>(v);
  else if (k == "mom_threshold_q8") P.mom_threshold_q8 = static_cast<int32_t>(v);
  else if (k == "mr_band") P.mr_band = static_cast<int32_t>(v);
  else if (k == "mm_half_spread") P.mm_half_spread = static_cast<int32_t>(v);
  else if (k == "noise_trend_gain") P.noise_trend_gain = static_cast<int32_t>(v);
  else if (k == "mom_scale") P.mom_scale = static_cast<int32_t>(v);
  else if (k == "mm_lean_div") P.mm_lean_div = static_cast<int32_t>(v);
  else throw std::invalid_argument("unknown parameter " + k);
  if (P.noise_width < 1 || P.max_qty < 1) throw std::invalid_argument("noise_width and max_qty must be >= 1");
}

}  // namespace

int main(int argc, char** argv) {
  int rank = 0, nranks = 1;
#ifdef LOB_USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif
  int code = 0;
  try {
    Config c;
    std::string label;
    bool header = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
        return argv[++i];
      };
      if (a == "--agents") c.num_agents = std::stoul(next());
      else if (a == "--instruments") c.num_instruments = std::stoul(next());
      else if (a == "--ticks") c.ticks = std::stoul(next());
      else if (a == "--seed") c.seed = std::stoull(next());
      else if (a == "--decide") c.decide = parse_decide(next());
      else if (a == "--match") c.match = parse_match(next());
      else if (a == "--threads") c.threads = std::stoi(next());
      else if (a == "--ttl") c.order_ttl = std::stoll(next());
      else if (a == "--mix") {
        const std::string v = next();
        if (std::sscanf(v.c_str(), "%u,%u,%u,%u", &c.mix[0], &c.mix[1], &c.mix[2], &c.mix[3]) != 4)
          throw std::invalid_argument("--mix needs four comma-separated numbers");
      } else if (a == "--no-sort") c.sort_by_type = false;
      else if (a == "--block") c.cuda_block = std::stoi(next());
      else if (a == "--no-compact") c.cuda_compact = false;
      else if (a == "--trades") c.trades_path = next();
      else if (a == "--prices") c.prices_path = next();
      else if (a == "--set") set_param(c.params, next());
      else if (a == "--label") label = next();
      else if (a == "--header") header = true;
      else if (a == "--help" || a == "-h") { if (rank == 0) usage(); throw 0; }
      else throw std::invalid_argument("unknown option " + a);
    }
#ifdef _OPENMP
    if (c.threads > 0) omp_set_num_threads(c.threads);
    const int threads = omp_get_max_threads();
#else
    const int threads = 1;
#endif

    Simulator sim(c, rank, nranks);
    const Result r = sim.run();

    // Report the slowest rank's phase times (that is what bounds a tick).
    Timings t = r.t;
#ifdef LOB_USE_MPI
    if (nranks > 1) {
      double v[8] = {t.decide, t.h2d, t.kernel, t.d2h, t.match, t.publish, t.comm, t.total};
      MPI_Allreduce(MPI_IN_PLACE, v, 8, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      t = Timings{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
    }
#endif
    if (rank == 0) {
      if (header)
        std::printf("label,decide,match,ranks,threads,agents,instruments,ticks,seed,sorted,"
                    "block,compact,trades,volume,hash,t_total,t_decide,t_h2d,t_kernel,t_d2h,"
                    "t_match,t_publish,t_comm,ticks_per_sec,agent_updates_per_sec\n");
      const double tps = c.ticks / t.total;
      std::printf("%s,%s,%s,%d,%d,%u,%u,%u,%llu,%d,%d,%d,%llu,%llu,%016llx,"
                  "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f,%.4e\n",
                  label.c_str(), to_string(c.decide), to_string(c.match), nranks, threads,
                  c.num_agents, c.num_instruments, c.ticks,
                  static_cast<unsigned long long>(c.seed), c.sort_by_type ? 1 : 0,
                  c.cuda_block, c.cuda_compact ? 1 : 0,
                  static_cast<unsigned long long>(r.num_trades),
                  static_cast<unsigned long long>(r.volume),
                  static_cast<unsigned long long>(r.trade_hash), t.total, t.decide, t.h2d,
                  t.kernel, t.d2h, t.match, t.publish, t.comm, tps,
                  tps * static_cast<double>(c.num_agents));
    }
  } catch (int) {
  } catch (const std::exception& e) {
    std::fprintf(stderr, "lobsim: %s\n", e.what());
    code = 1;
  }
#ifdef LOB_USE_MPI
  if (code != 0) MPI_Abort(MPI_COMM_WORLD, code);
  MPI_Finalize();
#endif
  return code;
}
