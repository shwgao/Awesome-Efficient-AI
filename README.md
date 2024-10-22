# Awesome Efficient AI
I am sharing awesome papers related to the HPC, Efficient AI and AI 4 Science etc..

Topics:
- [AI Compilers](#ai-compilers)
- [CUDA](#cuda)
- [NNFusion](#nnfusion)
- [Efficient AI Algorithms](#efficient-ai-algorithms)
- [Memory Management](#memory-management)

## AI Compilers
| Paper | Conference & Year |
|-------|-------------------|
|[Welder: Scheduling Deep Learning Memory Access via Tile-graph](./docs/AI%20Compilers/Welder/welder.md)|OSDI, 2023 |
|[TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](./docs/AI%20Compilers/TVM/TVM.md)| OSDI, 2018 |
|[Pytorch 2.0 compiler: torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#id3)|          |

## CUDA
| Paper | Conference & Year |
|-------|-------------------|
|[CUDA basic knowledge](./docs/CUDA/CUDA_basic_knowledge/%20CUDA_basics.md)||
|[Mirage: A Multi-Level Superoptimizer for Tensor Programs <br>(Generating Fast GPU Kernels without Programming in CUDA/Triton)](./docs/CUDA/Mirage/Mirage.md)|Arxiv, 2024|

## NNFusion
| Paper | Conference & Year |
|-------|-------------------|

## Efficient AI Algorithms
| Paper | Conference & Year |
|-------|-------------------|
|[G10: Enabling An Efficient Unified GPU Memory and Storage Architecture with Smart Tensor Migrations](./docs/Efficient%20AI%20Algorithems/G10/G10.md)|MICRO, 2023|
|[SELF-ATTENTION DOES NOT NEED $O(n^{2})$ MEMORY](./docs/Efficient%20AI%20Algorithems/efficient-attention-memory.md)|arXiv, 2021|
|[Enable Simultaneous DNN Services Based on Deterministic Operator Overlap and Precise Latency Prediction](./docs/Efficient%20AI%20Algorithems/Abacus/Abacus.md)|SC, 2021|

## Memory Management
### Existing methods to break the "memory wall" in AI training.
1. Distributed training: 
    - Data parallelism: Each worker holds a copy of the model and a subset of the data.
    - Model parallelism: Each worker holds a subset of the model.
2. Novel network architectures reduce the number of parameters.
    - Neural architecture search (NAS) and AutoML.
3. Model compression:
    - Quantization.
    - Pruning.
    - Distillation.
4. Training with reduced precision.
5. In-memory tensor compression.
6. Rematerialization: Recompute intermediate tensors instead of storing them.
7. Paging, swapping, and spilling to other memory pool.
8. Other methods may be benificial for memory management in AI training:
    - Operator fusion.
    - Compiler optimizations.

| Paper | Conference & Year |
|-------|-------------------|
|[MODeL: Memory Optimizations for Deep Learning](./docs/Memory%20Management/Peak%20Memory%20Minimization/MODel.md)|ICML, 2023|
|[ROAM: memory-efficient large DNN training via optimized operator ordering and memory layout](.)|arXiv, 2023|
|[Efficient Memory Management for Large Language Model Serving with PagedAttention](.)|SOSP, 2023|
|[MegTaiChi: Dynamic Tensor-based Memory Management Optimization for DNN Training](.)|ICS, 2022|
|[Dapple:A pipelined data parallel approach for training large models.](.)|PPoPP, 2021|
|[Zero-offload: Democratizing billion-scale modeltraining](.)|USENIX, 2021|
|[Memory-efficient pipeline-parallel dnn training](.)|PMLR, 2021|
|[Zero:Memory optimizations toward training trillion parameter models.](.)|SC20, 2020|
|[Efficient Memory Management for Deep Neural Net Inference](.)| arXiv, 2020|
|[Gpipe: Efficient training of giant neural networks using pipeline parallelism](.)|NeurIPS, 2019|