# Coop: Memory is not a Commodity

## Motivation: 
The existing tensor rematerialization techniques overlook the memory system in deep learning frameworks and implicitly assume that free memory blocks at different addresses are identical.

## Co-optimization: 
1. tensor allocation
2. tensor rematerialization

## Sliding Window: 
Remove objects that are consistent in the memory pool to reduce the memory fragmentation.

## Cheap tensor: 
Group tensors with the same mangitude and put them in the same memory block.

## Recomputable in-place

## heuristic: 

h(t) = c(t) / s(t) 
c(t): computational cost
s(t): staleness