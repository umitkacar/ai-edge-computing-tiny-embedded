# AI Edge Computing & TinyML - Comprehensive Guide

> **Latest Update**: January 2025 - State-of-the-Art Algorithms & Trends for Edge AI and Embedded Systems

## 📊 Table of Contents
- [🔥 SOTA Models & Algorithms (2024-2025)](#-sota-models--algorithms-2024-2025)
- [Object Detection & Vision](#object-detection--vision)
- [Inference Frameworks & Runtimes](#inference-frameworks--runtimes)
- [Model Compression & Optimization](#model-compression--optimization)
- [Hardware Acceleration](#hardware-acceleration)
- [Research Papers & Resources](#research-papers--resources)

---

## 🔥 SOTA Models & Algorithms (2024-2025)

### 🎯 Object Detection Models

#### **YOLOv11 (YOLO11)** - November 2024
State-of-the-art real-time object detection with transformer-based improvements.

**Key Features:**
- Transformer-based backbone with C3k2 blocks
- Partial Self-Attention (PSA) mechanism
- NMS-free training with dual label assignment
- 25-40% lower latency vs YOLOv10
- 10-15% improvement in mAP
- 60+ FPS processing capability

**Resources:**
- [Ultralytics YOLO11 Docs](https://docs.ultralytics.com/models/)
- [YOLO Evolution Paper](https://arxiv.org/html/2510.09653v2)

#### **YOLOv10** - May 2024
Eliminates NMS for end-to-end real-time detection.

**Performance:**
- YOLOv10s: 1.8x faster than RT-DETR-R18
- YOLOv10b: 46% less latency, 25% fewer parameters than YOLOv9-C
- mAP: 38.5 - 54.4

**Resources:**
- [YOLOv10 Paper](https://arxiv.org/pdf/2405.14458)
- [Ultralytics YOLOv10](https://docs.ultralytics.com/models/yolov10/)

#### **RT-DETR & RT-DETRv2** - 2024
First practical real-time detection transformer.

**Performance:**
- 53.1% AP at 108 FPS (NVIDIA T4)
- RT-DETRv2: >55% AP without speed loss
- mAP: 46.5 - 54.8

**Resources:**
- [RT-DETR vs YOLO11 Comparison](https://docs.ultralytics.com/compare/rtdetr-vs-yolo11/)

### 📱 Efficient Vision Models for Edge

#### **MobileNetV4** - ECCV 2024
Universal efficient architecture for mobile ecosystem.

**Innovations:**
- Universal Inverted Bottleneck (UIB) block
- Mobile MQA attention (39% speedup)
- Optimized NAS recipe
- 87% ImageNet accuracy @ 3.8ms (Pixel 8 EdgeTPU)

**Resources:**
- [MobileNetV4 Paper (Springer)](https://link.springer.com/chapter/10.1007/978-3-031-73661-2_5)
- [Google Research](https://syncedreview.com/2024/04/18/87-imagenet-accuracy-3-8ms-latency-googles-mobilenetv4-redefines-on-device-mobile-vision/)

#### **EfficientViT** - 2024
Lightweight multi-scale attention for high-resolution tasks.

**Features:**
- Memory-efficient Vision Transformer
- Cascaded group attention
- Suitable for dense prediction tasks

### 🤖 Small Language Models (SLMs) for Edge

#### **Microsoft Phi-3** - 2024
High-capability small language model family.

**Variants:**
- Phi-3-mini: 3.8B parameters
- Context: Up to 128K tokens
- Optimized for GPU, CPU, mobile deployment

**Resources:**
- [Phi-3 Overview](https://datasciencedojo.com/blog/small-language-models-phi-3/)

#### **TinyLlama** - 2024
Compact yet powerful language model.

**Specs:**
- 1.1B parameters
- Ideal for mobile/edge devices
- Strong performance for size

#### **Google Gemini Nano** - 2024
On-device AI for smartphones.

**Variants:**
- 1.8B / 3.25B parameters
- Edge-optimized for phones/IoT
- Context-aware reasoning, translation, summarization

#### **Meta Llama 3.2** - 2024
Edge AI and vision capabilities.

**Features:**
- Optimized for edge deployment
- Vision-language capabilities
- Mobile-friendly variants

**Resources:**
- [Llama 3.2 Announcement](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)

#### **MobileVLM** - 2024
Efficient vision-language model.

**Specs:**
- mobileLLaMA: 2.7B parameters
- Trained from scratch on open datasets
- Optimized for mobile devices

### ⚡ State Space Models - Efficient Transformers

#### **Mamba** - 2024
Linear-time sequence modeling with selective state spaces.

**Performance:**
- 5x higher throughput than Transformers
- Linear scaling in sequence length
- Mamba-3B outperforms Transformers of same size
- Matches Transformers 2x its size

**Resources:**
- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Mamba GitHub](https://github.com/state-spaces/mamba)
- [Mamba Survey](https://arxiv.org/html/2408.01129v1)

#### **eMamba** - 2024
Edge-optimized Mamba acceleration framework.

**Features:**
- End-to-end hardware acceleration
- Designed for edge platforms
- Leverages linear time complexity

**Resources:**
- [eMamba Paper](https://arxiv.org/html/2508.10370)

---

## 🚀 Inference Frameworks & Runtimes

### **TensorRT-LLM** - NVIDIA 2024
High-performance LLM inference on NVIDIA GPUs.

**Features:**
- State-of-the-art optimizations for LLMs
- Python and C++ API
- Significant speedup (70% faster than llama.cpp on RTX 4090)
- Maintains quality across all precisions

**Resources:**
- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [Deployment Guide](https://towardsdatascience.com/deploying-llms-into-production-using-tensorrt-llm-ed36e620dac4/)

### **vLLM** - UC Berkeley 2024
High-throughput LLM serving with PagedAttention.

**Features:**
- PagedAttention memory management
- Optimized key-value cache handling
- Supports AMD GPU, Google TPU, AWS Inferentia
- Built on PyTorch

**Resources:**
- [vLLM vs TensorRT-LLM](https://northflank.com/blog/vllm-vs-tensorrt-llm-and-how-to-run-them)

### **ExecuTorch** - Meta 2024
Efficient LLM execution on edge devices.

**Features:**
- Lightweight runtime for edge hardware
- Static memory planning
- Supports CPU, GPU, AI accelerators
- TorchAO quantization integration

**Resources:**
- [PyTorch Conference 2024](https://www.infoq.com/news/2024/09/pytorch-conference-2024/)

### **llama.cpp** - 2024
CPU-optimized LLM inference.

**Features:**
- Lower memory usage
- No GPU required
- Fast generation quality
- Cross-platform support

**Comparison:**
- [vLLM vs Ollama vs llama.cpp vs TGI vs TensorRT-LLM](https://itecsonline.com/post/vllm-vs-ollama-vs-llama.cpp-vs-tgi-vs-tensort)

---

## 🔧 Model Compression & Optimization

### 📉 Advanced Quantization Techniques

#### **AWQ (Activation-aware Weight Quantization)** - MIT 2024
**🏆 MLSys 2024 Best Paper Award**

**Innovation:**
- Protects salient weights based on activations
- Not all weights are equally important
- Skips critical weights during quantization

**Resources:**
- [AWQ GitHub](https://github.com/mit-han-lab/llm-awq)
- [MIT HAN Lab](https://hanlab.mit.edu/)

#### **GPTQ** - 2024
Early successful 4-bit quantization for large models.

**Features:**
- Row-wise weight matrix quantization
- Hessian matrix optimization
- GPU-focused inference
- Quantized 175B models (BLOOM, OPT-175B)

#### **QLoRA** - 2024
Efficient fine-tuning with quantization.

**Features:**
- 4-bit NormalFloat (NF4) data type
- Double quantization
- Low-Rank Adaptation matrices
- Fine-tune 65B models on single GPU

#### **Unsloth Dynamic 4-bit** - December 2024
Latest quantization innovation.

**Features:**
- Built on BitsandBytes
- Dynamic parameter quantization
- Determines quantization per-parameter

**Resources:**
- [Quantization Comparison Guide](https://generativeai.pub/practical-guide-of-llm-quantization-gptq-awq-bitsandbytes-and-unsloth-bdeaa2c0bbf6)
- [GPTQ vs GGUF vs AWQ](https://newsletter.maartengrootendorst.com/p/which-quantization-method-is-right)

### 🔬 Neural Architecture Search (NAS)

**Overview:**
NAS automates neural network architecture design, reducing manual intervention.

**Key Approaches:**

#### **Once-for-All (OFA)** - 2024
Train once, deploy everywhere.

**Features:**
- Weight-sharing supernetwork
- Represents any architecture in search space
- Significantly reduces computational costs
- Applied to ImageNet with ProxylessNAS and MobileNetV3

**Resources:**
- [NAS Overview](https://www.automl.org/nas-overview/)
- [MIT HAN Lab NAS](https://hanlab.mit.edu/techniques/nas)

### 🎓 Knowledge Distillation & Pruning

#### **TinyBERT** - 2024-2025
Two-stage distillation approach.

**Performance:**
- 96.8% of BERT-base performance
- 7.5x smaller (4 self-attention layers)
- Lowest energy variability (SD = 0.1032 kWh)
- Task-agnostic + task-specific distillation

#### **DistilBERT** - 2024
Single-phase task-agnostic distillation.

**Performance:**
- 97% of BERT performance
- 40% smaller
- 60% faster
- General-purpose applications

**Recent Research (2025):**
- 32% energy reduction with pruning on BERT
- Iterative methods combining distillation + adaptive pruning

**Resources:**
- [Nature Scientific Reports 2025](https://www.nature.com/articles/s41598-025-07821-w)
- [DistilBERT Medium](https://medium.com/huggingface/distilbert-8cf3380435b5)

---

## 🎯 TinyML & MCU-specific Advances

### **MCUNet Series** - MIT HAN Lab

**MCUNetV1:**
- Neural architecture for microcontrollers
- Co-design of model and inference engine

**MCUNetV2:**
- Record 71.8% ImageNet accuracy on MCU
- >90% accuracy on visual wake words (32kB SRAM)
- Enables object detection on tiny devices

**MCUNetV3:**
- Latest iteration with enhanced efficiency

**TinyTL:**
- Tiny transfer learning for MCUs
- On-device learning capabilities

**PockEngine:**
- Inference engine optimization

**Resources:**
- [MCUNet Official](https://mcunet.mit.edu/)
- [MCUNet GitHub](https://github.com/mit-han-lab/mcunet)
- [TinyML Projects](https://hanlab.mit.edu/projects/tinyml)

### **TinyDL (Tiny Deep Learning)** - 2024
Evolution from TinyML to deep learning on edge.

**Focus:**
- Deploying deep learning on ultra-constrained hardware
- Power consumption in mW range
- Sensor data analytics on-device

**Resources:**
- [TinyDL Survey](https://arxiv.org/html/2506.18927v1)

---

## 🔩 Hardware Acceleration & Platforms

### **NVIDIA Jetson Orin Nano Super** - Late 2024
Powerful edge AI development kit.

**Specs:**
- 67 INT8 TOPS compute
- 1.7x higher generative AI inference vs previous Orin Nano
- $249 price point

### **Edge TPU & Neural Accelerators**
- Google Pixel EdgeTPU
- Apple Neural Engine
- Specialized AI accelerators

### 📱 Mobile Deployment Targets
- ARM CPUs
- Mobile DSPs
- Mobile GPUs
- NPUs (Neural Processing Units)

---

---

## 🛠️ Implementation Resources & Tools

### ONNX Runtime
Cross-platform inference with ONNX models.

**Documentation & Tutorials:**
- [ONNX Runtime C++ Inference](https://leimao.github.io/blog/ONNX-Runtime-CPP-Inference/)
- [PyTorch to ONNX Tutorial](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- [ONNX Registry Tutorial](https://pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html)
- [On-Device Training](https://onnxruntime.ai/docs/api/python/on_device_training/training_artifacts.html)

**Compatibility:**
- [ONNX Runtime Compatibility](https://onnxruntime.ai/docs/reference/compatibility.html) (ONNX, OPSET, TensorRT, CUDA, CUDNN)
- [ONNX Versioning](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)
- [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

**Example Implementations:**
- [C++ ResNet Console App](https://github.com/cassiebreviu/cpp-onnxruntime-resnet-console-app)
- [ONNX Runtime C++ Example](https://github.com/k2-gc/onnxruntime-cpp-example)
- [ONNX Runtime Android](https://github.com/Rohithkvsp/OnnxRuntimeAndorid)
- [ByteTrack ONNX Inference](https://github.com/ifzhang/ByteTrack/blob/main/deploy/ONNXRuntime/onnx_inference.py)

**Model Repositories:**
- [HuggingFace ONNX Models](https://huggingface.co/models?sort=trending&search=onnx)
- [txtai ONNX Pipeline](https://neuml.github.io/txtai/pipeline/train/hfonnx/)
- [Ultralytics Export](https://docs.ultralytics.com/modes/export/#arguments)

### ONNX Runtime Quantization
- [Quantization Tools](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/quantization)
- [Float16 Optimization](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)
- [Quantization Examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization)

### YOLO Implementations

**YOLO-NAS with ONNX:**
- [YOLO-NAS ONNXRuntime](https://github.com/jason-li-831202/YOLO-NAS-onnxruntime)

**YOLO + TensorRT (Detection, Pose, Segmentation):**
- [YOLOv8-TensorRT-CPP](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP)
- [TensorRT C++ API](https://github.com/cyrusbehr/tensorrt-cpp-api)
- [YOLOv8-TensorRT (Python + C++)](https://github.com/triple-Mu/YOLOv8-TensorRT)
- [YOLO Pose C++](https://github.com/mattiasbax/yolo-pose_cpp)
- [TensorRT Samples](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec)
- [YOLOv8 TensorRT Tutorial](https://www.youtube.com/watch?v=Z0n5aLmcRHQ)

**YOLO + ONNXRuntime (All Tasks):**
- [YOLOv8-ONNX-CPP (Python + C++)](https://github.com/FourierMourier/yolov8-onnx-cpp/tree/main)
- [YOLOv8 Pose Implementation](https://github.com/mallumoSK/yolov8/blob/master/yolo/YoloPose.cpp)
- [YOLOv8 TensorRT Pose](https://github.com/triple-Mu/YOLOv8-TensorRT/blob/main/csrc/pose/normal/main.cpp)
- [YOLO-ONNXRuntime-CPP](https://github.com/Amyheart/yolo-onnxruntime-cpp)
- [YOLOv8-OpenCV-ONNXRuntime-CPP](https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp)
- [Ultralytics YOLOv8 C++ Examples](https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-ONNXRuntime-CPP)
- [YOLOv6-OpenCV-ONNXRuntime](https://github.com/hpc203/yolov6-opencv-onnxruntime/tree/main)
- [YOLOv5 Pose OpenCV](https://github.com/hpc203/yolov5_pose_opencv)

**Community Resources:**
- [hpc203 Repositories](https://github.com/hpc203?tab=repositories)
- [YOLO Issue Discussions](https://github.com/ultralytics/ultralytics/issues/1852)
- [YOLOv5 Fixed Bugs](https://github.com/ultralytics/yolov5/issues/916)
- [Chinese Tutorial](https://zhuanlan.zhihu.com/p/466677699)
- [ONNX Runtime Install Guide](https://velog.io/@dnchoi/ONNX-runtime-install)

### TensorRT
NVIDIA's high-performance deep learning inference optimizer.

**Resources:**
- [TensorRT Execution Provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements)
- [TensorRT Engine Cache](https://gitee.com/arnoldfychen/onnxruntime/blob/master/docs/execution_providers/TensorRT-ExecutionProvider.md#specify-tensorrt-engine-cache-path)

---

## 🌐 Edge Deployment Frameworks

### **FastDeploy** - PaddlePaddle
Easy-to-use deployment toolbox for AI models.

**Resources:**
- [FastDeploy GitHub](https://github.com/PaddlePaddle/FastDeploy)
- [Prebuilt Libraries](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/build_and_install/download_prebuilt_libraries.md)

### **DeepSparse & SparseML** - Neural Magic
CPU-optimized inference with sparsity.

**Features:**
- CPU inference acceleration
- Sparsity-aware optimization
- YOLOv5 benchmarks on CPUs

**Resources:**
- [YOLOv5 CPU Benchmark](https://neuralmagic.com/blog/benchmark-yolov5-on-cpus-with-deepsparse/)
- [SparseML GitHub](https://github.com/neuralmagic/sparseml/tree/main)
- [DeepSparse GitHub](https://github.com/neuralmagic/deepsparse)

### **NCNN** - Tencent
High-performance neural network inference framework for mobile.

**Resources:**
- [NCNN GitHub](https://github.com/Tencent/ncnn)
- [NCNN C++ Usage](https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/use-ncnn-with-alexnet.md)
- [YoloMobile](https://github.com/wkt/YoloMobile)
- [Awesome NCNN Collection](https://github.com/umitkacar/awesome-ncnn-collection)
- [Model Converter](https://convertmodel.com/)

### **MACE** - Xiaomi
Mobile AI Compute Engine.

**Resources:**
- [MACE GitHub](https://github.com/xiaomi/mace)

### **CoreML** - Apple
Machine learning framework for iOS/macOS.

**Model Collections:**
- [Semantic Segmentation CoreML](https://github.com/tucan9389/SemanticSegmentation-CoreML)
- [CoreML Models Collection](https://github.com/john-rocky/CoreML-Models#u2net)
- [Awesome CoreML Models](https://github.com/likedan/Awesome-CoreML-Models)
- [Awesome CoreML Models 2](https://github.com/SwiftBrain/awesome-CoreML-models)
- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)

**Tools & Documentation:**
- [PyTorch to CoreML Conversion](https://coremltools.readme.io/docs/pytorch-conversion)
- [CoreML Helpers](https://github.com/hollance/CoreMLHelpers)
- [Apple ML API](https://developer.apple.com/machine-learning/api/)
- [CoreML Performance Tool](https://github.com/vladimir-chernykh/coreml-performance)

**Stable Diffusion on CoreML:**
- [Apple ML-4M](https://github.com/apple/ml-4m/)
- [Apple ML Stable Diffusion](https://github.com/apple/ml-stable-diffusion)
- [Stable Diffusion 2 Base](https://huggingface.co/stabilityai/stable-diffusion-2-base)
- [Stability AI SD](https://github.com/Stability-AI/stablediffusion)
- [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- [RunwayML Stable Diffusion](https://github.com/runwayml/stable-diffusion)
- [Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

---

## ⚙️ Compilers & Low-Level Frameworks

### **TVM** - Apache
End-to-end deep learning compiler stack.

**Resources:**
- [TVM GitHub](https://github.com/apache/tvm)

### **LLVM**
Compiler infrastructure project.

**Resources:**
- [LLVM Project](https://github.com/llvm/llvm-project)

### **XNNPack** - Google
High-efficiency floating-point neural network inference operators.

**Resources:**
- [XNNPack GitHub](https://github.com/google/XNNPACK)

### **ARM-NN**
Inference engine for ARM platforms.

**Resources:**
- [ARM-NN GitHub](https://github.com/ARM-software/armnn)
- [ARM-NN Tutorial](https://www.youtube.com/watch?v=QuNOaFLobSg)

### **CMSIS-NN**
Efficient neural network kernels for ARM Cortex-M.

**Resources:**
- [CMSIS-NN GitHub](https://github.com/ARM-software/CMSIS_5)

### **Samsung ONE**
On-device Neural Engine compiler.

**Resources:**
- [ONE GitHub](https://github.com/Samsung/ONE)

---

## 💼 Industry & Commercial Solutions

### **Deeplite**
AI-Driven Optimizer for Deep Neural Networks.

**Focus:**
- ✅ Faster inference
- ✅ Smaller model size
- ✅ Energy-efficient deployment
- ✅ Cloud to edge optimization
- ✅ Maintain accuracy

**Resources:**
- [Deeplite Website](https://www.deeplite.ai/)

---

## 🔧 Utility Frameworks & Tools

### **OpenCV**
Computer vision library with C++ support.

**Resources:**
- [OpenCV C++ Playlist](https://www.youtube.com/playlist?list=PLUTbi0GOQwghR9db9p6yHqwvzc989q_mu)
- [Build OpenCV C++](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)

### **VQRF** - Video Compression
Vector Quantized Radiance Fields.

**Resources:**
- [VQRF GitHub](https://github.com/AlgoHunt/VQRF)

---

## 🖼️ Additional Model Architectures

### **PP-PicoDet**
Lightweight real-time object detector for mobile.

**Resources:**
- [PP-PicoDet Paper](https://arxiv.org/pdf/2111.00902.pdf)

### **EtinyNet**
Extremely tiny network for TinyML.

**Resources:**
- [EtinyNet GitHub](https://github.com/aztc/EtinyNet)

![TinyML Architecture](./tinyML.png)

---

## 🧠 Computing Architectures & APIs

**Supported Platforms:**
- **ARM** - Mobile & embedded processors
- **RISC-V** - Open-source instruction set
- **CUDA** - NVIDIA GPU computing
- **Metal** - Apple GPU framework
- **OpenCL** - Cross-platform parallel programming
- **Vulkan** - Cross-platform graphics & compute API

---

## 📚 Research Papers & Academic Resources

### Foundational Surveys (2024-2025)

**Edge Computing & Deep Learning:**
- [Deep Learning With Edge Computing: A Review](https://www.cs.ucr.edu/~jiasi/pub/deep_edge_review.pdf)
- [Convergence of Edge Computing and Deep Learning: Comprehensive Survey](https://arxiv.org/pdf/1907.08349.pdf)
- [Machine Learning at the Network Edge: A Survey](https://arxiv.org/pdf/1908.00080.pdf)
- [Edge Deep Learning in Computer Vision & Medical Diagnostics](https://link.springer.com/article/10.1007/s10462-024-11033-5)

**TinyML Specific:**
- [From Tiny Machine Learning to Tiny Deep Learning: A Survey (2024)](https://arxiv.org/html/2506.18927v1)
- [EtinyNet: Extremely Tiny Network for TinyML](https://ojs.aaai.org/index.php/AAAI/article/download/20387/version/18684/20146)
- [Ultra-low Power TinyML System for Real-time Visual Processing at Edge](https://arxiv.org/pdf/2207.04663.pdf)

**State Space Models & Efficient Architectures:**
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Mamba-360: Survey of State Space Models](https://arxiv.org/html/2404.16112v1)
- [eMamba: Efficient Acceleration Framework for Edge Computing](https://arxiv.org/html/2508.10370)

**Vision Models:**
- [MobileNetV4: Universal Models for the Mobile Ecosystem (ECCV 2024)](https://link.springer.com/chapter/10.1007/978-3-031-73661-2_5)
- [Vision Transformer Models for Mobile/Edge Devices: A Survey](https://link.springer.com/article/10.1007/s00530-024-01312-0)
- [YOLO Evolution: YOLOv5, YOLOv8, YOLO11, YOLO26](https://arxiv.org/html/2510.09653v2)
- [YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/pdf/2405.14458)

**Model Compression & Optimization:**
- [Comparative Analysis of Model Compression for Carbon Efficient AI (2025)](https://www.nature.com/articles/s41598-025-07821-w)
- [Systematic Review on Neural Architecture Search (2024)](https://link.springer.com/article/10.1007/s10462-024-11058-w)
- [Advances in Neural Architecture Search](https://academic.oup.com/nsr/article/11/8/nwae282/7740455)

**Collections:**
- [Awesome Embedded and Mobile Deep Learning](https://github.com/csarron/awesome-emdl/blob/master/README.md)

---

## 🎓 Contributing & Community

This repository serves as a comprehensive resource for AI edge computing and TinyML practitioners. Contributions, updates, and corrections are welcome!

**Last Updated:** January 2025
**Maintainer:** [Your GitHub Profile]

---

**Keywords:** TinyML, Edge AI, Embedded ML, Model Compression, Quantization, Neural Architecture Search, YOLO, MobileNet, Transformer, State Space Models, ONNX Runtime, TensorRT, Inference Optimization
