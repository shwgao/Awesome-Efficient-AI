
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

