# GPU_accelerated_LPT_CFD_code
This repository contains GPU accelerated version of the particle tracking model developed by Merel Kooi for biofouled microplastic particles ( available at: https://pubs.acs.org/doi/10.1021/acs.est.6b04702) written in CUDA Fortran and CUDA Python. This repository is intended as a learning tool for GPU programming and developed during the TRA220 GPU-accelerated Computational Methods using Python and CUDA offered at Chalmers University of Technology.

The performance of an LPT code developed in Python and Fortran was evaluated on both CPU and GPU architectures. This study aimed to quantify the performance enhancements achieved by running LPT simulations on GPUs and to compare the efficiencies of CUDA implementations in Python versus Fortran. The number of particles was varied from 4 (2×2) to 16,384 (128×128) to assess scalability and computational efficiency. The following simulations were conducted on an NVIDIA GeForce RTX 3070 GPU. The results are illustrated in the figure below.

![fig2](https://github.com/user-attachments/assets/a78b6095-9bee-4f24-89c9-c2c22d3968e1)

For small particle numbers, the Fortran implementation running on the CPU outperformed all other configurations by a significant margin. This superior performance is primarily attributed to Fortran being a native language, which allows it to efficiently utilize CPU resources when handling a limited number of particles. However, as the number of particles increases, the Fortran CPU performance scales linearly on a logarithmic scale, resulting in drastically reduced efficiency for large particle counts. In contrast, the Python implementation on the CPU exhibited the poorest performance across all particle numbers. Nevertheless, when the Python code was ported to CUDA, substantial performance improvements were observed. Both GPU-accelerated codes, Python with CUDA and Fortran with CUDA, demonstrated modest performance for small particle numbers but exhibited excellent scalability, showing minimal performance degradation as the number of particles increased. Overall, CUDA-Fortran outperformed CUDA-Python, though the difference was nearly negligible for smaller particle numbers.

The next figure presents a performance benchmark of the GPUs available in the Chalmers C3SE Vera supercomputing cluster in comparison to Nvidia GeForce RTX 3070.

![fig3](https://github.com/user-attachments/assets/f1661caa-c777-45a8-8ca1-41dfdc9f135d)

The first observation is that the LPT code demonstrates superior performance on V100 and A100 GPUs compared to T4 and A40 models. Furthermore, the CUDA Python implementation significantly outperforms the RTX 3070 when deployed on V100 and A100 GPUs, whereas the Fortran version achieves only marginal improvements on the same hardware. Overall, CUDA Fortran runs approximately twice as fast as CUDA Python across all benchmarked GPUs, except for the A40, where both implementations demonstrate comparable performance.
