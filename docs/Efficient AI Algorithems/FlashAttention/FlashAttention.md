# FlashAttention

## FlashAttention-1: Fast and Memory-Efficient Exact Attention with IO-Awareness ([GitHub](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file)).

![alt text](image.png)

1. Problem Context: Traditional self-attention in Transformers is memory-intensive and slow, particularly for long sequences. Existing approximate attention solutions lower computational demands but fail to provide consistent speed improvements and often compromise model quality.

2. FlashAttention's Approach: The proposed method incorporates tiling and recomputation techniques. Tiling breaks down attention computations into blocks that fit within SRAM, reducing the need to store and read large matrices from HBM. This process allows FlashAttention to avoid materializing the full attention matrix, leading to fewer memory accesses and thus higher speed and efficiency.

3. Benefits and Performance: FlashAttention achieves significant speedups over conventional attention methods. In specific benchmarks, such as training BERT-large and GPT-2, FlashAttention provides up to a 3× speed boost. It also enables models to handle longer sequences effectively, allowing for new applications in long-context tasks.

4. Extension to Block-Sparse Attention: FlashAttention extends to block-sparse attention, which is particularly efficient for longer sequences with inherent sparsity. This variant shows further speed gains, outperforming existing sparse and approximate attention mechanisms.