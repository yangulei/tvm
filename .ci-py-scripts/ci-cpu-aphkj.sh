set -eux

sccache --start-server
./tests/scripts/task_config_build_cpu.sh build-cpu
./tests/scripts/task_build.py --build-dir build-cpu
./tests/scripts/task_python_unittest.sh
./tests/scripts/task_python_vta_fsim.sh
./tests/scripts/task_python_vta_tsim.sh
