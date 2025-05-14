# Accurate and Convenient Energy Measurements for GPUs: A Detailed Study of NVIDIA GPU’s Built-In Power Sensor

## Abstract
GPU has emerged as the go-to accelerator for HPC workloads, however its power consumption has become a major limiting factor for further scaling HPC systems. An accurate understanding of GPU power consumption is essential for further improving its energy efficiency, and consequently reducing the associated carbon footprint. Despite the limited documentation and lack of understanding, NVIDIA GPUs’ built-in power sensor is widely used in energy-efficient computing research. Our study seeks to elucidate the internal mechanisms of the power readings provided by nvidia-smi and assess the accuracy of the measurements. We evaluated over 70 different GPUs across 12 architectural generations, and identified several unforeseen problems that can lead to drastic under/overestimation of energy consumed, for example on the A100 and H100 GPUs only 25% of the runtime is sampled. We proposed several mitigations that could reduce the energy measurement error by an average of 35% in the test cases we present.

## Contributions
1. The error in nvidia-smi’s power draw is ±5% as opposed to ±5W claimed by NVIDIA. On modern GPUs capable of drawing 700W this could lead to a ±30W of over/underestimation. For a data centre with 10,000 GPUs, this would lead to an extra $1 million in electricity cost yearly. 
2. On the A100 and H100 GPUs only 25% of the runtime is sampled for power consumption, during the other 75% of the time, the GPU can be using drastically different power and nvidia-smi and results presented by it are unaware of this. The situation with the Grace Hopper Superchip is even more pronounced, with the GPU sampling 20%, and the CPU sampling merely 10% of the runtime. 
3. Naively measure energy consumption using nvidia-smi could underestimate by on average 39.3% and up to 68.6%. We proposed several mitigations that could bring this error down to the 5% intrinsic power measurement electronic component error.

## Proposed Measurement and good practices
1. Execute the target program for 32 consecutive iterations or until a minimum runtime of 5 seconds is reached. If information loss exists due to a small averaging window, insert 8 controlled delays evenly spaced within the repetitions. 
2. Perform separate trials with randomised delay in between. 
3. Post process the data by discarding repetitions executed during rise time, and shift data to sync with GPU activity.


## Benchmark code:

https://github.com/JimZeyuYang/GPU_Power_Benchmark