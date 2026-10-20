#!/usr/bin/env bash
# One-step setup on Ubuntu (or Ubuntu inside WSL on Windows):
# installs the tools, builds the simulator with OpenMP and MPI, and runs the
# unit tests. Run it from the repository folder:
#
#   bash scripts/setup_ubuntu.sh
#
# It asks for your password once (for installing packages).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3 Installing compiler, CMake and MPI (asks for your password)"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential cmake git zip openmpi-bin libopenmpi-dev python3 python3-pip > /dev/null
echo "   done"

echo "== 2/3 Building (this takes about a minute)"
cmake -S . -B build -DLOB_USE_MPI=ON -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build -j "$(nproc)" > /dev/null
echo "   built build/lobsim and build/lob_tests"

echo "== 3/3 Running the unit tests"
./build/lob_tests | tail -1

echo
echo "All set. Next, run the demo:   bash scripts/demo.sh build"
