# CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion

## Introduction
- Large language models can provide higher quality answers with more relevant information. A typical example is RAG. 
- In RAG, a user query will be prepended by multiple **text chunks** from the knowledge base to form the LLM input.
- The context text chunks will significantly slow down the LLM **prefill**, specifically **the time to first token(TTFT)**.
- Speed up the prefill of LLM: re-use the stored KV caches.
- Limitations of existing solutions:
    - Prefix caching: only stores and reuses the KV cache of the prefix of the LLM input.
    - Full KV reuse: When a reused text is not at the input prefix, it still reuses the KV cache by adjusting its positional embedding so that the LLM generation will produce meaningful output. However, this method approximates the cross-attention within the text chunk.
- CacheBlend: Selectively recomputing the KV cache of a small fraction of tokens, based on the preceding texts in the specific LLM input.

## Motivation
**Goal**: When an LLM input includes multiple re-used text chunks, how to quickly update the pre-computed KV cache, such that the forward attention matrix (and subsequently the output text) has **minimum difference** with the one produced by full KV recompute.

## Design

### Selctively recomputing KV cache
- It first applies a mask on the input of each layer 𝑖 to reduce it to a subset of selected tokens. 
- It then transforms the reduced input into the 𝑄𝑖 , 𝐾𝑖 and 𝑉𝑖 vectors will also be restricted to the selected tokens. 
- It then expands the 𝐾𝑖 vector and 𝑉𝑖 vector by reusing the KV cache entries associated with the un-selected tokens on layer 𝑖, so that the attention matrix includes attention between selected tokens and all other tokens. 
- Finally, it runs the same attention module to produce the input of the next layer.

### Selecting which tokens to recompute

