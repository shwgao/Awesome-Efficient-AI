# Awesome Efficient AI
I am sharing awesome papers related to the HPC, Efficient AI and AI 4 Science etc..

Topics:
- [AI Compilers](#ai-compilers)
- [CUDA](#cuda)
- [NNFusion](#nnfusion)
- [Efficient AI Algorithms](#efficient-ai-algorithms)
- [Break the "Memory Wall"](./docs/Memory%20Management/Break_memory_wall.md)
- [Offloading](#offloading)
- [Memory Management](#memory-management)
- [Parallelism and Pipeline](#parallelism-and-pipeline)
- [LLM Training and Inference](#llm-training-and-inference)
- [LLM Inference Scaling Laws](#llm-inference-scaling-laws)
- [LLM Serving](#llm-serving)
- [Some other topics](#some-other-topics)


## AI Compilers
| Paper | Conference & Year |
|-------|-------------------|
|[Welder: Scheduling Deep Learning Memory Access via Tile-graph](./docs/AI%20Compilers/Welder/welder.md)|OSDI, 2023 |
|[TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](./docs/AI%20Compilers/TVM/TVM.md)| OSDI, 2018 |
|[Pytorch 2.0 compiler: torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#id3)|          |
|[MLIR](./docs/AI%20Compilers/MLIR/MLIR.md)|Arxiv, 2024|

## CUDA
| Paper | Conference & Year |
|-------|-------------------|
|[CUDA basic knowledge](./docs/CUDA/CUDA_basic_knowledge/%20CUDA_basics.md)||
|[Mirage: A Multi-Level Superoptimizer for Tensor Programs <br>(Generating Fast GPU Kernels without Programming in CUDA/Triton)](./docs/CUDA/Mirage/Mirage.md)|Arxiv, 2024|
|[Accurate and Convenient Energy Measurements for GPUs: A Detailed Study of NVIDIA GPU’s Built-In Power Sensor](./docs/CUDA/GPU_Power.md)|SC24, 2024|

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




## Memory Management
| Paper | Conference & Year |
|-------|-------------------|
|[MAGIS: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN](./docs/Memory%20Management/MAGIS/MAGIS.md)|ASPLOS, 2024
|[ROAM: memory-efficient large DNN training via optimized operator ordering and memory layout](./docs/Memory%20Management/ROAM/ROAM.md)|arXiv, 2023|
|[TelaMalloc: Efficient On-Chip Memory Allocation for Production Machine Learning Accelerators](./docs/Memory%20Management/TelaMalloc/TelaMalloc.md)|ASPLOS, 2023|
|[MODeL: Memory Optimizations for Deep Learning(vLLM)](./docs/Memory%20Management/Peak%20Memory%20Minimization/MODel.md)|ICML, 2023|
|[Efficient Memory Management for Large Language Model Serving with PagedAttention](.)|SOSP, 2023|
|[MegTaiChi: Dynamic Tensor-based Memory Management Optimization for DNN Training](.)|ICS, 2022|
|[Dynamic Tensor Rematerialization](./docs/Memory%20Management/DTR/DTR.md)|ICLR, 2021|
|[Zero:Memory optimizations toward training trillion parameter models.](.)|SC20, 2020|
|[Efficient Memory Management for Deep Neural Net Inference](.)| arXiv, 2020|
|[A scalable concurrent malloc(3) implementation for freebsd.](.)| ..., 2006|
|[Tcmalloc: Thread-caching malloc.](.)| ..., 2005|
|[Hoard: A scalable memory allocator for multithreaded applications.](.)| ASPLOS, 2000|


## Parallelism and Pipeline 
| Paper | Conference & Year |
|-------|-------------------|
|[GraphPipe: Improving Performance and Scalability of DNN Training with Graph Pipeline Parallelism](./docs/Parallism/Graphpipe/graphpipe.md)|ASPLOS, 2025|
|[H3T: Efficient Integration of Memory Optimization and Parallelism for High-Throughput Transformer Training](.)|NeurIPS, 2024|
|[Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](./docs/Parallism/Alpa/Alpa.md)|OSDI, 2022|
|[Dapple:A pipelined data parallel approach for training large models.](.)|PPoPP, 2021|
|[Memory-efficient pipeline-parallel dnn training](.)|PMLR, 2021|
|[GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](./docs/Parallism/GPipe/Gpipe.md)|NeurIPS, 2019|

## LLM Training and Inference
| Paper | Conference & Year |
|-------|-------------------|
|[FlexLLMGen: High-throughput Generative Inference of Large Language Models with a Single GPU](.)|ICML, 2023|
|[DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale](.)|SC22, 2022|

## LLM Inference Scaling Laws
| Paper | Conference & Year |
|-------|-------------------|
|[Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving](.)|ICLR, 2025|
|[s1: Simple test-time scaling](https://aipapersacademy.com/s1/)|ICLR, 2025|

## LLM Serving
| Paper | Conference & Year |
|-------|-------------------|
|[SGLang](https://lmsys.org/blog/2024-07-25-sglang-llama3/)| , |
|[vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](./docs/LLM%20serving/vllm/vllm.md)|SOSP, 2023|

## Some other topics
| Notes | References |
|-------|-------------------|
|CMU Machine Learning System, CMU, 2023|[Course](https://www.cs.cmu.edu/~zhihaoj2/15-849/schedule.html)|
|Pytorch Internals|[notes](./docs/Some%20other%20topics/Pytorch/pytorch_internals.md)|
|Performance optimization for deep learning|[tutorial/slides](https://docs.google.com/presentation/d/1vikeOOHF2ig15af2qQxtUG3KRDu9T973/edit#slide=id.p2)|
|Good reading materials about transformer|[Github Markdown](https://github.com/feuyeux/hello-ai/tree/7f29a2cb90f58e1c91dbefa4002fb9be090b9db1/background)|
|CS267, Applications of Parallel Computers, UC Berkeley|[Course](https://sites.google.com/lbl.gov/cs267-spr2024)|


