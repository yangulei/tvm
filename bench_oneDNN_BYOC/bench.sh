#!/bin/bash
export OMP_NUM_THREADS=24
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

<<COMMENT
COMMENT
echo "benchmarking tvm ..."
#--profiling \
TVM_LIBRARY_PATH=${TVM_HOME}/build_release_native \
numactl --physcpubind=0-23 --membind=0 \
python ${TVM_HOME}/bench_oneDNN_BYOC/bench_tvm.py \
 --network=resnet50\
 --batch-size=1 \
 --warmup=20 \
 --repeat=5 \
 --steps=20 \
 1>bench_tvm.log \
 2>bench_tvm_debug.log

echo "benchmarking byoc ..."
#DNNL_VERBOSE=1 \
TVM_LIBRARY_PATH=${TVM_HOME}/build_release_gnu \
numactl --physcpubind=0-23 --membind=0 \
python ${TVM_HOME}/bench_oneDNN_BYOC/bench_byoc.py \
 --network=resnet50\
 --batch-size=1 \
 --warmup=20 \
 --repeat=5 \
 --steps=20 \
 1>bench_byoc.log \
 2>bench_byoc_debug.log
