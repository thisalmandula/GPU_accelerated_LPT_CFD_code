To run Fortran on local computer with nvidia graphics card, install the latest version of: NVIDIA HPC SDK (https://developer.nvidia.com/hpc-sdk)

To run nvfortran (Linux), use following commands to add to path (might have to do after restarting PC)

NVHPC=/opt/nvidia/hpc_sdk
PATH=$NVHPC/Linux_x86_64/24.11/compilers/bin:$PATH
LD_LIBRARY_PATH=$NVHPC/Linux_x86_64/24.11/compilers/lib:$LD_LIBRARY_PATH


After installing CUDA Fortran, run following commands to compile

nvfortran -cuda -gpu=rdc -c biofouling.cuf
nvfortran -cuda -gpu=rdc lpt.cuf biofouling.cuf -o gpu

To run the code:
./gpu