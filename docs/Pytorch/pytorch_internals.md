## [Pytorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals)
- Tensor concepts
- Autograd


### Tensor

```
Tensor :
sizes (D, H, W)
strides (H*W, W, 1) # contiguous memory
dtype float
device CPU
layout torch.strided
```
