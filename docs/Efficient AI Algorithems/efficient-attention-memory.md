# SELF-ATTENTION DOES NOT NEED $O(n^{2})$ MEMORY

>We present a very simple algorithm for attention that requires $O(1)$ memory with respect to sequence length and an extension to self-attention that requires $O(log n)$ memory. This is in contrast with the frequently stated belief that self-attention requires $O(n^2)$ memory. While the time complexity is still $O(n^2)$, device memory rather than compute capability is often the limiting factor on modern accelerators.

[github](https://github.com/google-research/google-research/tree/master/memory_efficient_attention)