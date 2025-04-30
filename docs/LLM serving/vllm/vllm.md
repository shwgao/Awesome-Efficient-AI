# Efficient Memory Management for Large Language Model Serving with PagedAttention

## Basic Information
* **Title:** Efficient Memory Management for Large Language Model Serving with PagedAttention
* **Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
* **Publication Year:** 2023
* **Journal/Conference:** SOSP '23
* **External Resources:**
    * [Detailed vLLM Source Code Analysis (Zhihu)](https://zhuanlan.zhihu.com/p/661360117)

## Abstract Summary
LLM serving faces memory inefficiency due to the large, dynamic KV cache. Existing systems waste memory through fragmentation and redundancy, limiting throughput. This paper introduces PagedAttention, inspired by OS paging, allowing non-contiguous KV cache storage. The vLLM system, built on PagedAttention, minimizes memory waste and enables flexible cache sharing. Evaluations show vLLM boosts throughput by 2-4x over systems like FasterTransformer and Orca, especially with long sequences and complex decoding.

## Introduction / Background
LLMs are powerful but costly to serve. High throughput is key for cost-effectiveness. Autoregressive generation relies heavily on the KV cache, making memory a bottleneck.

* **Generation Phases:**
    * **Prompt processing**: Parallelizable (matrix-matrix multiplication) as all input tokens are known.
    * **Autoregressive generation**: Sequential (matrix-vector multiplication), less efficient due to token dependencies.
* **Memory Challenges:**
    * Large KV cache size per request.
    * Complex memory patterns needed for different decoding algorithms (e.g., beam search).
    * Variable, unknown input and output lengths requiring dynamic memory management.
* **Problem:** Traditional contiguous KV cache allocation causes significant internal and external fragmentation, wasting memory and limiting batch sizes. This also prevents memory sharing.
* **Batching:** Processing multiple requests improves efficiency (e.g., cellular batching, iterative batching), but memory limits the batch size.

## Related Work
This work builds upon general model serving systems, specialized Transformer serving systems (like FasterTransformer), and memory optimization techniques. It's most closely related to Orca, which uses iteration-level scheduling for throughput. vLLM is complementary, focusing on memory efficiency via PagedAttention to enable larger batches. PagedAttention addresses memory challenges exacerbated by fine-grained scheduling.

## Methodology / Model (Techniques)
The core idea is **PagedAttention** within the **vLLM** system.

* **PagedAttention Algorithm:**
    * Partitions KV cache into fixed-size blocks stored non-contiguously in physical memory.
    * Inspired by OS virtual memory and paging.
    * Adapts attention computation to work over these scattered blocks.

* **vLLM System:**
    * Uses a KV Cache Manager with logical-to-physical block mapping via block tables.
    * Allocates physical memory blocks on demand, minimizing waste.
    * Supports distributed execution with tensor parallelism.

* **Decoding & Sharing:**
    * Handles parallel sampling and beam search efficiently using block sharing and **copy-on-write (CoW)**.
    * Supports caching KV blocks for shared prefixes (e.g., system prompts).
    * Manages mixed decoding methods transparently via the block mapping layer.

* **Scheduling and Preemption:**
    * Uses **First-Come, First-Served (FCFS)** scheduling.
    * **Preemption**: If memory is full, evicts sequence groups (all-or-nothing policy), prioritizing the latest requests for preemption.
    * Recovery Mechanisms:
        * **Swapping:** Evicted KV blocks are moved to CPU RAM (swap space).
        * **Recomputation:** Regenerate KV cache for preempted sequences upon rescheduling. (Paper notes recomputation is faster for small block sizes, while swapping can be faster for larger block sizes).

* **Implementation:**
    * Python and C++/CUDA. Uses PyTorch, Transformers, NCCL.
    * Custom CUDA kernels optimize PagedAttention operations (fused read/write/attention/copy).

## Experimental Setup
* **Models:** OPT (13B, 66B, 175B), LLAMA (13B).
* **Hardware:** NVIDIA A100 GPUs.
* **Workloads:** Synthesized from ShareGPT (long/variable sequences) and Alpaca (short sequences) datasets. Poisson arrival rates.
* **Baselines:** FasterTransformer (latency-optimized library) and Orca (throughput-optimized system, re-implemented with Oracle, Pow2, Max reservation strategies).
* **Metrics:** Throughput measured via normalized latency vs. request rate.

## Results and Analysis
* **Basic Sampling:** vLLM significantly outperforms baselines (1.7x-8x Orca, up to 22x FT) due to higher memory efficiency enabling larger batches.
* **Parallel Sampling & Beam Search:** vLLM's advantage increases with more parallelism/beam width due to effective memory sharing.
* **Shared Prefix:** vLLM shows substantial throughput gains (1.7x-3.6x over Orca) by caching prefix KV states.
* **Chatbot:** vLLM achieves 2x higher throughput than Orca, handling long prompts efficiently.
* **Ablations:** PagedAttention kernel latency is slightly higher, but end-to-end performance is vastly better. Block size 16 offers a good balance. Recomputation vs. swapping performance depends on block size.

## Discussion and Limitations
PagedAttention is highly effective for memory-bound LLM serving but might add overhead to compute-bound or static workloads. vLLM adapts OS concepts with LLM-specific optimizations. Kernel overhead exists but is outweighed by system-level gains. Optimal block size and preemption strategy depend on hardware/workload.

## Conclusion and Future Work
PagedAttention and vLLM successfully adapt OS paging concepts to LLM serving, drastically improving memory efficiency and enabling flexible sharing. This leads to 2-4x throughput gains over state-of-the-art systems.

## Novelty and Contribution
* Quantified memory inefficiency in existing LLM serving.
* Proposed **PagedAttention** for non-contiguous KV cache storage.
* Developed **vLLM** system demonstrating near-zero waste and efficient sharing (CoW).
* Achieved state-of-the-art throughput via memory optimization.
* Showcased effective adaptation of OS memory management principles.