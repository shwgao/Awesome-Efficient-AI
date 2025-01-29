
## [Scaling law](https://www.kth.se/blogs/pdc/2018/11/scalability-strong-and-weak-scaling/)
### Strong scaling law
speedup = 1 / (s + p / N)

where s is the proportion of execution time spent on the serial part, p is the proportion of execution time spent on the part that can be parallelized, and N is the number of processors. Amdahl’s law gives the upper limit of speedup for a problem of fixed size. 

### Weak scaling law
scaled speedup = s + p × N

where s, p and N have the same meaning as in Amdahl’s law. This is called weak scaling, where the scaled speedup is calculated based on the amount of work done for a scaled problem size (in contrast to Amdahl’s law which focuses on fixed problem size).