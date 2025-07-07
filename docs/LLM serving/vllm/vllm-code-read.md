# Reference
- ZhiHu: @猛猿
- ZhiHu: @BoLi2001
- ChatGPT
- vLLM official document

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
    

### How vllm compute the maximum number of tokens can be stored in the KV cache?
vLLM computes the maximum number of tokens that can be stored in the KV cache based on available GPU memory after loading model weights and reserving space for activations. It divides the remaining memory by the per-token KV cache size (which depends on model architecture, number of layers, heads, head size, and dtype), then allocates as many blocks as fit, each block holding a fixed number of tokens (block_size). The total tokens = num_blocks × block_size. A step-by-step breakdown:
1. Profile Memory Usage: vLLM loads the model weights and runs a dummy forward pass to profile the peak memory usage for activations and other non-KV-cache allocations (gpu_worker.py).

2. Calculate Available Memory: It subtracts the memory used for model weights and activations from the total GPU memory, then applies the gpu_memory_utilization factor to determine the memory available for the KV cache (kv_cache_utils.py).

3. Determine Per-Block and Per-Token Size: The per-token KV cache size is determined by model parameters (layers, heads, head size, dtype). Each block holds a fixed number of tokens (block_size), and the per-block size is calculated accordingly.

4. Compute Number of Blocks: The available KV cache memory is divided by the per-block size to get the number of blocks that can be allocated.

5. Compute Maximum Tokens: The total number of tokens that can be stored in the KV cache is num_blocks * block_size.


### EngineCore Architecture

In previous versions of vLLM, the entire process of accepting user requests, preprocessing requests, tokenization, multimodal input processing, request scheduling, GPU inference execution, result detokenization, and finally returning results to users was executed in a single process. However, this involved significant CPU overhead, as all steps except the GPU inference were executed on the CPU and were serial in nature. This overhead was particularly evident for smaller models.

In previous vLLM inference versions, we often found that the detokenization step (token ID → text) would become a bottleneck, especially when dealing with long sequences. During the process of waiting for sequences to be detokenized, the GPU would be idle. Now, through this decomposition, what was originally serial is parallelized.

Therefore, vLLM v1 adopted a multi-process architecture, separating the API server and LLM core functionality into two processes, with inter-process communication using ZeroMQ:

- The API server is responsible for receiving user requests, preprocessing requests, tokenization, detokenizing inference results, and returning them to users
- The other process, EngineCore, is responsible for the core LLM functionality, mainly scheduling requests and performing GPU inference in a loop

Simply put, **the request pre-processing and output result post-processing are separated from the actual inference process into 2 different processes** (process0, process1). The Client is responsible for request pre-processing and output result post-processing, while EngineCore is responsible for the actual inference process, with different processes communicating data using ZMQ.

This **fully realizes the overlap of CPU and GPU, hiding CPU overhead and improving throughput**.

For offline batching and online serving, they select different types of Clients for operation, but their EngineCore operation is basically consistent.

#### Offline Batching

The specific structure is:

<img src="llm%20serving.assets/vllm1%20(1).png" width="800"  alt="vLLM v1 Architecture">

<img src="llm%20serving.assets/v2-0d7b3741144d2293ac6d615f2d5f3d01_1440w.jpg" width="600"  alt="vLLM v1 Structure">

When constructing the LLM, an LLMEngine member is constructed inside the LLM object:

<img src="llm%20serving.assets/image-20250317105547742.png" width="600" alt="LLM Construction">

<img src="llm%20serving.assets/image-20250317105612948.png" width="600"  alt="LLMEngine Construction">

When constructing the LLMEngine object, an EngineCore member is constructed inside it.

When constructing EngineCore, if the multi-process mode (multiprocess_mode) is not selected, an `EngineCore` object is directly constructed in the current process, which is the v0 mode. You can directly call the add_request and step methods on the EngineCore object in a single process. Then calling the add_request method of self.engine_core can directly add requests to the Scheduler, and calling the step method can also directly obtain output from EngineCore (actually, the local EngineCore object is wrapped as InprocClient here, because it needs to maintain consistency with other MPClient interfaces):

<img src="llm%20serving.assets/image-20250315180455553.png" width="600"  alt="EngineCore Construction">

<img src="llm%20serving.assets/image-20250315181726396.png" width="600"  alt="InprocClient">

<img src="llm%20serving.assets/image-20250315183759311.png" width="600"  alt="EngineCore Step">

If multi-process mode is selected, only an MPClient is constructed. Specifically, different MPClients are constructed based on whether the parameter is asynchronous or synchronous (asyncio_mode), but both have MPClient as their base class. **The so-called MPClient is to construct EngineCore in another process, and itself is a client of EngineCore in the current process**.

Specifically: Assuming it's synchronous mode (asyncio_mode=False), a `SyncMPClient` is constructed. Before constructing it, its base class MPClient is constructed first.

During MPClient construction, **a background process named EngineCore is created through the BackgroundProcHandle object, and the main thread of this process calls the static member function run_engine_core in the EngineCoreProc class**:

<img src="llm%20serving.assets/image-20250315174756932.png" width="600"  alt="Background Process Creation">

<img src="llm%20serving.assets/image-20250315175022310.png" width="600"  alt="EngineCoreProc Run">

In the run_engine_core method, an `EngineCoreProc` object is first constructed, then its `run_busy_loop` method is called:

<img src="llm%20serving.assets/image-20250315161716147.png" width="600"  alt="EngineCoreProc Run Busy Loop">

The EngineCoreProc class is a subclass of the EngineCore class, which wraps EngineCore to run on an independent process, **mainly used to handle communication between EngineCore and the main process**. Specifically, the EngineCoreProc object has two queues, one for receiving requests sent by LLMEngine, and one for storing its own processed results. It also creates two threads to handle these two queues respectively:

<img src="llm%20serving.assets/image-20250322112258947.png" width="600"  alt="EngineCoreProc Queues">

After the EngineCoreProc object is constructed, the run_engine_core method **executes a busy loop in the main thread of this EngineCore process**: it continuously polls whether there are new incoming requests in the input_queue of EngineCoreProc, and if there are, it takes them out.

Then it determines what type of request this is. If it's an add request, it calls the add_request method of EngineCore to add it to the Scheduler. **At this point, the request is really added to the waiting queue of the Scheduler**, and the mapping of its id and req object is added to the requests table of the Scheduler.

Then it repeatedly executes the above process, adding all requests in the input_queue of EngineCoreProc to the Scheduler until the input_queue is empty, then executes one step. After execution, the inference results are saved in the output_queue of EngineCoreProc, then the above process is repeated, continuing to check if there are new requests in the Scheduler.

<img src="llm%20serving.assets/image-20250323100515300.png" width="600"  alt="EngineCore Main Thread">

The above is **what the main thread of EngineCore does. In summary, it gets requests from input_queue, adds them to the Scheduler's queue, then calls step to save the execution results in output_queue**.

The other two threads created during EngineCoreProc construction are responsible for:

- One thread reads request data sent from the main process from the socket by calling `socket.recv_multipart`, deserializes it, and saves it in input_queue
- One thread gets data computed by the main thread of this background process from output_queue, serializes it, and sends it out through the socket using `socket.send_multipart`

These two threads are also continuously polling.

The above is the MPClient construction process, then the subclass SyncMPClient is constructed.

In this client, there is an output_queue used to save the results computed by the EngineCore process. During SyncMPClient construction, a separate thread (named `EngineCoreOutputQueueThread`) is created to specifically handle the output_queue, specifically reading data from the socket, deserializing it, and saving it in output_queue.

<img src="llm%20serving.assets/image-20250315182429000.png" width="600"  alt="SyncMPClient Output Queue">

Then the upper layer gets data from output_queue through the get_output method in SyncMPClient. The upper layer also sends requests by calling the add_request method of SyncMPClient, that is, sending req to the EngineCoreProc of another process through the socket in SyncMPClient.

So the data flow process of the multi-process EngineCore is:

- The main thread A of the main process calls add_request for each prompt through the generate method, with the call stack as shown below:

  <img src="llm%20serving.assets/image-20250315161306565.png" width="600"  alt="Generate Call Stack">

  Finally, in the `add_request` method of `llm_engine`, the add_request method of the engine_core member variable of LLMEngine is called:

  <img src="llm%20serving.assets/image-20250315192952776.png" width="600"  alt="LLMEngine Add Request">

- Then the `add_request` method of `SyncMPClient` is actually called, sending data to the EngineCore process through the socket

- Thread B in the EngineCore process is responsible for reading request data sent from the main process from the socket, deserializing it, and saving it in input_queue

- The main thread C of the EngineCore process runs `run_busy_loop` and is responsible for reading data from input_queue and saving it to the waiting queue of the Scheduler member of the EngineCore object

- Then the main thread C calls step once more, saving the computed data in output_queue

- Thread D of the EngineCore process serializes the data in output_queue and sends it out through the socket

- Then thread E of the main process deserializes the data from the socket and saves it in the output_queue of `SyncMPClient`

- At this point, for the main thread A, it has just finished calling the `_validate_and_add_requests` method in the generate method to send the prompt out, then calls the `_run_engine` method, where it repeatedly calls the step method of llm_engine in a while loop. The step method of LLMEngine in v1 version directly calls get_output from EngineCore (in this case, actually calling get_output on MPClient), getting the data from the output_queue of SyncMPClient.

<img src="llm%20serving.assets/vllm1 (1)-1742635141038-8.png" width="800"  alt="vLLM v1 Step Method">

**For the llm_engine of v1 version, the step method is very simple and doesn't need to do any scheduling or computation operations, because the main work is done by the step method of EngineCore**, whether it's a multi-process EngineCore or a local EngineCore. The step method of llm_engine only needs to call the get_output method of engine_core:

- For InprocClient, its get_output method needs to call the step of the local EngineCore, then wait for the step to complete and return the result. **In this case, add_request and scheduling are serial with GPU execution. If the model is relatively small and the execution time on GPU is less, then the CPU overhead will be very large**.

  <img src="llm%20serving.assets/image-20250315203746892.png" width="600"  alt="InprocClient Get Output">

- For MPClient, the get_output method directly gets output from the queue. Because step is executed by the EngineCore of another process, this **achieves the overlap of add_request (including accepting user requests, preprocessing requests, tokenization) and scheduling with GPU execution**. This is why when debugging vLLM v1, even though several prompts are added at once through add_request, only one req is shown in the waiting queue during scheduling.

#### Online Serving