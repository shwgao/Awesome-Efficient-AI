
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

## HaHa moment:
- *A process* is a running program with its own isolated memory space and resources. 
- *A thread* is a lighter-weight unit of execution within a process, sharing its parent process's memory and resources. 

###