# A Summary of Neural Network Quantization

Quantization is a model optimization technique that involves converting the continuous, high-precision floating-point numbers (typically 32-bit float, FP32) within a neural network into lower-precision numbers, most commonly 8-bit integers (INT8).

The primary goals of quantization are:
1.  **Model Compression**: To significantly reduce the model's file size, making it easier to store and deploy on devices with limited storage.
2.  **Inference Acceleration**: To leverage specialized hardware that can perform integer arithmetic much faster than floating-point arithmetic, resulting in lower latency.

## What Parts of a Model Are Quantized?

For effective acceleration, quantization is applied to most of the data involved in the computation.

### 1. Model Weights

Weights are the parameters learned during training and are almost always quantized.

* **Why**: They constitute the bulk of a model's size. Converting weights from FP32 to INT8 reduces the model's size by approximately **75%**. This also reduces memory bandwidth requirements during inference.
* **How**: For each layer (or each channel), a **scale factor** and **zero-point** are calculated. These parameters are used to map the FP32 weight values to the INT8 range. This is a one-time conversion performed before deployment.

### 2. Activations (Feature Maps)

Activations are the outputs of intermediate layers, which serve as the inputs to subsequent layers. Quantizing them is **critical for achieving performance gains**.

* **Why**: To perform computations (like convolutions and matrix multiplications) using fast integer-only hardware units, both the weights and the inputs (activations) must be integers.
* **How**:
    * **Static Quantization**: This is the most common and highest-performing method. It requires a small "calibration dataset" to be passed through the FP32 model to observe the typical range of activations for each layer. Based on this, fixed `scale` and `zero-point` values are calculated for each activation tensor. During inference, these fixed parameters are used to convert activations to INT8.
    * **Dynamic Quantization**: In this approach, weights are quantized offline, but activations are quantized "on-the-fly" during each inference pass. It's simpler as it doesn't require a calibration set but offers less performance gain due to the runtime overhead of calculating the range for each activation. It's often used for models like LSTMs and Transformers.

### 3. Biases

Biases are a special case and are **typically not quantized to INT8**.

* **Why**: The result of an INT8 matrix multiplication (`INT8_activation * INT8_weight`) is accumulated into a higher-precision integer (usually INT32). To maintain precision, the bias is added at this INT32 level. Since biases are very few in number compared to weights, keeping them at a higher precision (INT32 or FP32) has a negligible impact on model size but is crucial for preventing significant accuracy degradation.

## How Inputs and Outputs Are Handled

* **Model Input**: The model interface typically still accepts standard FP32 data (e.g., a normalized image tensor). The inference engine uses a pre-calculated `scale` and `zero-point` to **quantize this input to INT8** just before it enters the first layer of the network.

* **Model Output**: The final output of the network (which is in INT8 or INT32) is almost always **de-quantized back to FP32**. This provides a user-friendly, interpretable result (like class probabilities) to the downstream application.

## Simulated vs. True Quantization

This is a crucial distinction between the analysis phase and the deployment phase.

| Aspect                | Simulated Quantization (Fake Quant)                                        | True Quantization (for Deployment)                                                              |
| :-------------------- | :------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------- |
| **Primary Goal** | **To assess accuracy loss** and find the best quantization strategy.       | **To achieve maximum inference speed** and minimum model size.                                  |
| **How it Works** | Weights are quantized and then immediately **de-quantized** back to FP32. All calculations remain in the FP32 domain. | The model is fully converted. Weights are stored as INT8, and computations are performed using **integer arithmetic units**. |
| **Performance** | **Slower** than the original FP32 model due to extra simulation overhead.    | **Significantly faster** than the FP32 model.                                                   |
| **Use Case** | Model analysis, research, and quantization-aware training (QAT).           | Final deployment on edge devices, servers, and cloud environments.                              |

In summary, a real-world quantized inference pipeline looks like this:

**FP32 Input → `Quantize` → [INT8 Compute → INT8 Compute → ...] → `De-quantize` → FP32 Output**