
# Build and install oneDNN:
    wget https://github.com/oneapi-src/oneDNN/archive/refs/tags/v2.5.1.tar.gz    # get v2.5.1 source code
    tar -xzvf v2.5.1.tar.gz && cd oneDNN-2.5.1/ # extract the source code
    mkdir build && cd "$_"    # setup a build dir
    cmake -DCMAKE_INSTALL_PREFIX=../install .. && make -j && make install    # build and install
    export oneDNN_HOME=`pwd`/../install
    export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${oneDNN_HOME}/include
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${oneDNN_HOME}/lib64

# build TVM
## get tvm
    git clone --recursive https://github.com/yangulei/tvm -b dev_byoc
    cd tvm
## build for tvm benchmark
    mkdir build_release_native && cd "$_"
    cp ../cmake/config.cmake .
    sed -i '/USE_LLVM/ s/OFF/ON/g' config.cmake    # Enable LLVM
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
## build for byoc benchmark
    cd .. && mkdir build_release_gnu && cd "$_"
    cp ../cmake/config.cmake .
    sed -i '/USE_LLVM/ s/OFF/ON/g' config.cmake    # Enable LLVM
    sed -i '/USE_DNNL_CODEGEN/ s/OFF/ON/g' config.cmake    # Enable oneDNN
    sed -i '/USE_OPENMP/ s/none/gnu/g' config.cmake    # needed for oneDNN-BYOC, but slow for native TVM codegen
    cmake .. -DCMAKE_BUILD_TYPE=Release -DEXTERN_LIBRARY_DNNL=${oneDNN_HOME}/lib64/libdnnl.so
    make -j && cd ..
## setup env
    export TVM_HOME=`pwd`
    export PYTHONPATH=${TVM_HOME}/python:${PYTHONPATH}

# Setup python environment
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh && source ~/.bashrc
    conda create -n tvm python=3.8
    conda activate tvm
    pip install numpy decorator attrs tornado psutil xgboost cloudpickle pytest
    pip install mxnet==1.8.0.post0 gluoncv

# benchmark
    cd bench_oneDNN_BYOC/
    bash bench.sh