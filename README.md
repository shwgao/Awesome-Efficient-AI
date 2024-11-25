# Awesome Efficient AI
I am sharing awesome papers related to the HPC, Efficient AI and AI 4 Science etc..

Topics:
- [AI Compilers](#ai-compilers)
- [CUDA](#cuda)
- [NNFusion](#nnfusion)
- [Efficient AI Algorithms](#efficient-ai-algorithms)
- [Break the "memory wall"](#existing-methods-to-break-the-memory-wall-in-ai-training)
    - [Offloading](#distributed-training-or-inference)
    - [Memory Management](#memory-management)
    - [LLM Training and Inference](#llm-training-and-inference)
- [Parallelism and Pipeline](#parallelism-and-pipeline)
- [Some other topics](#some-other-topics)

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
|[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning]()| arXiv, 2023|
|[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](./docs/Efficient%20AI%20Algorithems/FlashAttention/FlashAttention.md)| NeurIPS, 2022|
|[SELF-ATTENTION DOES NOT NEED $O(n^{2})$ MEMORY](./docs/Efficient%20AI%20Algorithems/efficient-attention-memory.md)|arXiv, 2021|
|[Enable Simultaneous DNN Services Based on Deterministic Operator Overlap and Precise Latency Prediction](./docs/Efficient%20AI%20Algorithems/Abacus/Abacus.md)|SC, 2021|


## Offloading
| Paper | Conference & Year |
|-------|-------------------|
|[Zero-offload: Democratizing billion-scale modeltraining](./docs/Memory%20Management/Zero-offload/zero-offload.md)|USENIX, 2021|


### Existing methods to break the "memory wall" in AI training.
1. [Offloading](#offloading): 
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
8. [Memory Management](#memory-management)
9. Other methods may be benificial for memory management in AI training:
    - Operator fusion.
    - Compiler optimizations.

## Memory Management
| Paper | Conference & Year |
|-------|-------------------|
|[MAGIS: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN](.)|ASPLOS, 2024
|[ROAM: memory-efficient large DNN training via optimized operator ordering and memory layout](.)|arXiv, 2023|
|[MODeL: Memory Optimizations for Deep Learning](./docs/Memory%20Management/Peak%20Memory%20Minimization/MODel.md)|ICML, 2023|
|[Efficient Memory Management for Large Language Model Serving with PagedAttention](.)|SOSP, 2023|
|[MegTaiChi: Dynamic Tensor-based Memory Management Optimization for DNN Training](.)|ICS, 2022|
|[Zero:Memory optimizations toward training trillion parameter models.](.)|SC20, 2020|
|[Efficient Memory Management for Deep Neural Net Inference](.)| arXiv, 2020|
|[A scalable concurrent malloc(3) implementation for freebsd.](.)| ..., 2006|
|[Tcmalloc: Thread-caching malloc.](.)| ..., 2005|
|[Hoard: A scalable memory allocator for multithreaded applications.](.)| ASPLOS, 2000|

## LLM Training and Inference
| Paper | Conference & Year |
|-------|-------------------|
|[FlexLLMGen: High-throughput Generative Inference of Large Language Models with a Single GPU](.)|ICML, 2023|
|[DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale](.)|SC, 2022|

## Parallelism and Pipeline 
| Paper | Conference & Year |
|-------|-------------------|
|[H3T: Efficient Integration of Memory Optimization and Parallelism for High-Throughput Transformer Training](.)|NeurIPS, 2024|
|[Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](./docs/Parallism/Alpa/Alpa.md)|OSDI, 2022|
|[Dapple:A pipelined data parallel approach for training large models.](.)|PPoPP, 2021|
|[Memory-efficient pipeline-parallel dnn training](.)|PMLR, 2021|
|[GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](./docs/Parallism/GPipe/Gpipe.md)|NeurIPS, 2019|

## Some other topics
| Notes | References |
|-------|-------------------|
|Pytorch Internals|[notes](./docs/Some%20other%20topics/Pytorch/pytorch_internals.md)|
|Performance optimization for deep learning|[tutorial/slides](https://docs.google.com/presentation/d/1vikeOOHF2ig15af2qQxtUG3KRDu9T973/edit#slide=id.p2)|