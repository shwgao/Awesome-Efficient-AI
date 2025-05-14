### Existing methods to break the "memory wall" in AI training.
1. Offloading: 
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
8. Memory Management
9. Other methods may be benificial for memory management in AI training:
    - Operator fusion.
    - Compiler optimizations.