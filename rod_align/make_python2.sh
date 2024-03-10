#!/usr/bin/env bash
cd src
echo "Compiling rod_align kernels by nvcc..."

# Specify the architecture of your NV card below.
# -arch=sm_75 is compatible with the following NV GPU cards,
# GeForce RTX 2080 Ti, RTX 2080, RTX 2070    Quadro RTX 8000, Quadro RTX 6000, Quadro RTX 5000 Tesla T4
# See more at https://raw.githubusercontent.com/stereolabs/zed-yolo/master/libdarknet/Makefile
nvcc -c -o rod_align_kernel.cu.o rod_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_75

cd ../
# Export CUDA_HOME. And build and install the library.
export CUDA_HOME=/usr/local/cuda-10.0 && python setup.py install

