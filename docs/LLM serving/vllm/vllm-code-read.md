
## modules
- Entrypoint: LLM, API server
- Engine: LLMengine
- Scheduler: continuous batching
- KV cache manager: PagedAttentio, LMCache
- Evictor
    - prefix caching(use case: same prefix of a request is cached)
    - lru caching(use case: same request is cached)
    - CacheBlend(What if prefix doesn't match?)
    - DeepSeek (MLA)
- Worker
    - workerbase class
    - worker class
- Model executor (Model runner)
    - llama.py (line 265)
- Modelling: transfer huggingface model to vllm format model
- Attention backend

## Distributed
### Infra side
- Communication device:
    - NVLink: direct communication between GPUs
    - Infinity Band: High-speed communication between nodes
    - RDMA: Remote Direct Memory Access
        - RDMA NIC
        - Software solution
        - Key advantage: bypassing operating system / zero copy
        - RoCE
    - NVSwitch: 

- Communication libarary: `vllm/distributed/device_communicator`
    * `PyNccl`: communication for NVIDIA.
    * `shared memory`: OS.
    * `custom allreduce`: A kernel just for all reduce operation.
    * `torch.distributed`: Provide wide supoort to a list of communication libraries.

### Algorithm side
First step of distributed LLM is to read: 
`vllm/model_executor/models/llama.py`. It includes TP, PP, and DP.

If you are interested in the **attention** backend, you should read: `vllm/attenion/backends/flash_attn.py`.

### Pipeline parallel
- Much less requirement on device to device connection hardware.
- Cost: not improve latency, while Tensor parallel directly improve latency.
- Every worker incharge of a subset of layers.
    - Details: `vllm/model_executor/models/llama.py`
    - `self.start_layer` --> `self.end_layer`
    - between workers: communicate `IntermediateTensor`. Using `get_pp_group()`(in file `vllm/worker/model_runner.py`) to communicate.

### Expert parallel & data parallel (advanced)
- Why expert parallel?
    - Mistral /Mixtral / Deepseek model: MoE.
        - Normal model: all weights participant in computation.
        - MoE: expert as granularity, a small subset of experts participate in computation, this subset of experts may be different for different requests.
    - Place different experts onto different GPUs --> expert parallel.
    - **Algorithm**:
        - Expert parallel:
            - Shuffle (deepep communication kernel)
            - Forward
            - Shuffle back
    - TP is for attention and linear layers and attention is the challenging part, EP is for linear layers.
    - Shared expert will have high load --> duplicate shared expert.
- DP (data parallel):
    - max TP << EP needed.
    - TP < # attention heads.
    - basic linear layer "degree of paralleism" >> basic attention layer TP degree of paralleism, so use DP to parallelize request to raise attention degree of paralleism.
    - Difficult to implement in practice:
        - request padding to avoid deadlock.


### PD disaggregation
- P: prefill, D: decode
- Why PD disaggregation?
    prefill will stop other request's decode

- Key challenge in PD:
    - How to transfer KV cache:
        - 2 modes: pooling mode, p2p mode
        - LMCache, MoonCake, MIXL
    - How to extract (and inject) KV cache from (to) vLLM?
        - connector API
        - called in model_runner.py
            - before forward: try receive KV cache (inject KV cache into vLLM's paged memory)
            - after forward: extract KV cache from vLLM's paged memory and send it to outside.
        - When to send the request to P and D node?
            - first P then D
            - first D then P


### Speculative decoding
- LLM inference is GPU-memory bound. So we need to find a way to increase amount of computation but does not significantly increase the amount of GPU memory access.

- Solution: Generate token --> Guess multiple tokens and verify
    - In terms of token generation per iteration
        - Guess 3 tokens, acceptence rate: 2/3, if 2 tokens of guessing is correct, LLM inference will generate a new token --> 3 tokens.
    - Iteration time
        - Computation: (1 + 3)x
        - Memory:
            - w/o spec: Model parameters (8x2 GB) + KV caches (n * 100 KB)
            - w/ spec: Model parameters (8x2 GB) + KV caches ((n+3) * 100 KB)
        - Iteration time almost unchanged.
    


## HaHa moment:
- *A process* is a running program with its own isolated memory space and resources. 
- *A thread* is a lighter-weight unit of execution within a process, sharing its parent process's memory and resources. 


### How vllm compute the maximum number of tokens can be stored in the KV cache?
vLLM computes the maximum number of tokens that can be stored in the KV cache based on available GPU memory after loading model weights and reserving space for activations. It divides the remaining memory by the per-token KV cache size (which depends on model architecture, number of layers, heads, head size, and dtype), then allocates as many blocks as fit, each block holding a fixed number of tokens (block_size). The total tokens = num_blocks × block_size. A step-by-step breakdown:
1. Profile Memory Usage: vLLM loads the model weights and runs a dummy forward pass to profile the peak memory usage for activations and other non-KV-cache allocations (gpu_worker.py).

2. Calculate Available Memory: It subtracts the memory used for model weights and activations from the total GPU memory, then applies the gpu_memory_utilization factor to determine the memory available for the KV cache (kv_cache_utils.py).

3. Determine Per-Block and Per-Token Size: The per-token KV cache size is determined by model parameters (layers, heads, head size, dtype). Each block holds a fixed number of tokens (block_size), and the per-block size is calculated accordingly.

4. Compute Number of Blocks: The available KV cache memory is divided by the per-block size to get the number of blocks that can be allocated.

5. Compute Maximum Tokens: The total number of tokens that can be stored in the KV cache is num_blocks * block_size.


###