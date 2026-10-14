# Real-Time Open-Vocabulary and Fine-Tuned Everyday Object Detection with Contextual Audio-Visual Interaction

**Author**: Advanced Machine Learning Engineering Team  
**Institution**: Advanced Machine Learning & Deep Learning Capstone Project  
**Date**: October 2026  
**Format**: IEEE Conference / Journal Technical Paper Structure  

---

## Abstract

Real-world assistive robotics and vision systems demand robust recognition of everyday personal and workstation items while maintaining low inference latency on edge hardware. Standard closed-vocabulary convolutional detectors (e.g., COCO-pretrained YOLO) fail to detect common unstructured objects (e.g., pens, keys, chargers, earbuds, wallets) outside their fixed 80-category taxonomy. This paper presents an end-to-end Machine Learning pipeline that bridges this gap using two complementary paradigms: (1) zero-shot open-vocabulary detection via Vision-Language Modeling (YOLO-World v2), and (2) targeted supervised fine-tuning of an advanced convolutional backbone (YOLO11m) on an expanded 16-class everyday objects dataset. Using self-supervised distillation, photometric data augmentations, and Bayesian hyperparameter optimization with Optuna, our fine-tuned YOLO11m architecture achieves an empirical validation precision of **98.0%**, recall of **80.2%**, **mAP@50 of 83.7%**, and **mAP@50-95 of 79.5%** with a **13.5 ms** inference latency on an edge NVIDIA GeForce RTX 4050 GPU. High-dimensional feature representation is analyzed using PCA and t-SNE manifold projections. Finally, the system is fully containerized with Docker, exposing a high-throughput FastAPI REST backend and a local Gradio audio-visual interaction interface.

**Keywords** &mdash; *Object Detection, YOLO11, YOLO-World, Open-Vocabulary Detection, Knowledge Distillation, Bayesian Optimization, Optuna, t-SNE, FastAPI, Edge AI.*

---

## 1. Introduction

Automated visual identification of unstructured personal and workstation objects represents a fundamental pillar in assistive technologies, personal robotics, augmented reality, and voice-assisted productivity interfaces. Despite dramatic advances in convolutional neural networks (CNNs) and Vision Transformers (ViTs), contemporary production deployments remain bottlenecked by the closed-world assumption: models are trained to categorize exclusively the 80 fixed object classes cataloged by the Microsoft Common Objects in Context (COCO) benchmark.

In practical domestic, academic, and office environments, common objects such as writing instruments (pens, pencils), security tokens (door keys, keychains), personal accessories (wallets, spectacles), electronic peripherals (chargers, power bricks, wireless earbuds), and organizational stationery (notebooks, staplers) are ubiquitous yet entirely missing from standard COCO vocabularies. Consequently, deployed detectors exhibit false-negative dropouts or misclassify these objects as generic bounding noise.

To resolve these limitations, this research project formulates and implements a dual-paradigm computer vision and conversational assistant framework:
1. **Dynamic Open-Vocabulary Detection**: Integrating YOLO-World v2 to project open-vocabulary text embeddings into cross-attention vision backbones, permitting zero-shot adaptation to arbitrary everyday items without retraining.
2. **Supervised Representation Fine-Tuning**: Formulating an automated data harvesting and self-supervised distillation pipeline that trains a high-capacity YOLO11m architecture across 16 everyday desk categories over 200 epochs.
3. **Multi-Modal Active Context Interaction**: Coupling spatial bounding-box tracking with local deterministic and cloud-based LLM question-answering engines.
4. **Production Deployment**: Exposing both a Swagger-documented FastAPI microservice with Pydantic contract validation and a containerized Gradio interface.

---

## 2. Literature Review & Related Work

### 2.1 Closed-Vocabulary Real-Time Object Detectors
The evolution of real-time object detection has been driven by the single-stage You Only Look Once (YOLO) architecture family, progressing from YOLOv1 through YOLOv8 and recently YOLO11. YOLO11 introduces C3k2 cross-stage partial blocks, spatial pyramid pooling fast (SPPF), and decoupled detection heads that jointly optimize generalized intersection-over-union (CIoU) and distribution focal loss (DFL). However, conventional supervised detectors are bounded by their training dataset taxonomies (e.g., Pascal VOC, COCO).

### 2.2 Vision-Language Models & Open-Vocabulary Detection
Recent breakthroughs in multimodal foundation models, such as CLIP (Contrastive Language-Image Pre-training) and Grounding DINO, enable zero-shot object localization by aligning visual region features with textual prompt embeddings. YOLO-World (Cheng et al., 2024) advances this by integrating offline textual reprojection and reparameterizable vision-language cross-attention, allowing real-time inference on edge GPUs ($>30$ FPS). Nonetheless, open-vocabulary models can exhibit sensitivity drops when prompted with small, thin, or heavily occluded personal items.

### 2.3 Research Gaps Addressed
- **Vocabulary Elasticity vs. Throughput**: Existing systems force an unappealing trade-off between massive, slow transformer ensembles and fast, closed-vocabulary models.
- **Unstructured Desk Objects**: Lack of targeted transfer learning pipelines for personal desktop items.
- **Production Edge Validation**: Limited literature on combining dynamic open-vocabulary inference with deterministic local knowledge graph Q&A on consumer laptop hardware.

---

## 3. Methodology & System Architecture

The end-to-end system architecture comprises three synchronized operational layers: data engineering & representation extraction, dual-engine visual modeling, and edge API orchestration.

```
+-----------------------------------------------------------------------------------+
|                            DATA ENGINEERING PIPELINE                             |
|  +--------------------+     +-------------------+     +------------------------+  |
|  | Multi-Source Web   | --> | YOLO-World Pseudo | --> | Geometric/Photometric  |  |
|  | Harvester (Wiki/   |     | Label Distillation|     | Data Augmentation      |  |
|  | Local Scene Sets)  |     | (Conf >= 0.25)    |     | (Flips, Jitter)        |  |
|  +--------------------+     +-------------------+     +------------------------+  |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                         DUAL-ENGINE DETECTION SUBSYSTEM                           |
|  +----------------------------------+     +-------------------------------------+ |
|  |  Engine A: YOLO-World v2         |     |  Engine B: Fine-Tuned YOLO11m       | |
|  |  - Text-Prompt Reprojection      |     |  - 20.1M Parameters (C3k2 Backbone) | |
|  |  - Zero-Shot Open-Vocabulary     |     |  - 16-Class Fine-Tuned Weights      | |
|  |  - Dynamic Runtime Class Update  |     |  - 200 Epochs (mAP@50: 83.7%)       | |
|  +----------------------------------+     +-------------------------------------+ |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                     ACTIVE-TARGET TRACKING & MULTIMODAL Q&A                      |
|  +----------------------------------+     +-------------------------------------+ |
|  | Spatial IoU Tracker & Lock       | --> | Local Knowledge Graph & Cloud LLM   | |
|  | (Largest-Area Auto-Selection)    |     | Grounded Audio-Visual Reasoning     | |
|  +----------------------------------+     +-------------------------------------+ |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                         PRODUCTION DEPLOYMENT PLATFORM                            |
|  +----------------------------------+     +-------------------------------------+ |
|  | FastAPI REST Service (Port 8000) |     | Gradio Production UI (Port 7860)    | |
|  | Swagger Docs & Pydantic Schemas  |     | Obsidian Dark Theme, Dockerized     | |
|  +----------------------------------+     +-------------------------------------+ |
+-----------------------------------------------------------------------------------+
```

### 3.1 Mathematical Loss Formulations

The fine-tuning of YOLO11m optimizes a composite multi-task loss objective combining box coordinate regression, distribution focal loss, and multi-class classification:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}}$$

#### 1. Complete Intersection over Union ($\mathcal{L}_{\text{CIoU}}$)
$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{\text{gt}})}{c^2} + \alpha v$$
where $\rho(b, b^{\text{gt}})$ represents Euclidean distance between prediction and ground-truth box centroids, $c$ is the diagonal length of the smallest enclosing bounding box, and $\alpha v$ penalizes discrepancies in aspect ratio:
$$v = \frac{4}{\pi^2} \left( \arctan\frac{w^{\text{gt}}}{h^{\text{gt}}} - \arctan\frac{w}{h} \right)^2, \quad \alpha = \frac{v}{(1 - \text{IoU}) + v}$$

#### 2. Distribution Focal Loss ($\mathcal{L}_{\text{DFL}}$)
To model box boundary ambiguity as a continuous probability distribution across discrete bins:
$$\mathcal{L}_{\text{DFL}}(S_i, S_{i+1}) = - \left( (y_{i+1} - y)\log(S_i) + (y - y_i)\log(S_{i+1}) \right)$$

#### 3. Binary Cross-Entropy Loss ($\mathcal{L}_{\text{BCE}}$)
For each candidate class prediction $\hat{y}_c$ against binary target $y_c$:
$$\mathcal{L}_{\text{BCE}} = - \sum_{c=1}^{C} \left[ y_c \log \sigma(\hat{y}_c) + (1 - y_c) \log (1 - \sigma(\hat{y}_c)) \right]$$

### 3.2 Hyperparameter Optimization via Bayesian TPE
Hyperparameter tuning is framed as a sequential model-based global optimization problem over parameter domain $\Theta$:
$$\theta^* = \arg\max_{\theta \in \Theta} f(\theta)$$
where $f(\theta)$ represents the validation mAP@50 score. Optuna's Tree-structured Parzen Estimator (TPE) models $p(\theta | y)$ by constructing two density distributions: $l(\theta)$ for trials yielding top-tier mAP, and $g(\theta)$ for the remainder, maximizing Expected Improvement (EI):
$$\text{EI}(\theta) = \int_{-\infty}^{y^*} (y^* - y) p(y|\theta) dy \propto \frac{l(\theta)}{g(\theta)}$$

---

## 4. Implementation & Computational Experiments

### 4.1 Computational Environment
- **Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB GDDR6 VRAM, 96-bit bus, 2560 CUDA cores).
- **Software Stack**: Windows 11 64-bit, Python 3.11.5, PyTorch 2.6.0+cu124, CUDA 12.4 runtime, Ultralytics 8.4.175, Optuna 5.0.0, FastAPI 0.115.11.
- **Optimization Strategy**: Automatic Mixed Precision (AMP, FP16) with CUDA-accelerated batch operations (`batch=8`).

### 4.2 Analytical Visualizations

The pipeline incorporates six advanced analytical visualizations:

1. **Loss and Convergence Curves (`results.png`)**: Illustrates the monotonic decline of box loss ($1.123 \to 0.285$), classification loss ($4.263 \to 0.223$), and DFL ($0.983 \to 0.821$) across 200 epochs.
2. **Normalized Confusion Matrix (`confusion_matrix_normalized.png`)**: Highlights true positive rates exceeding 90% across primary desk classes (mugs, laptops, phones, bottles).
3. **Precision-Recall Curve (`BoxPR_curve.png`)**: Documents area-under-curve performance yielding mAP@50 of 0.837 across all 16 categories.
4. **F1-Score Confidence Dynamics (`BoxF1_curve.png`)**: Demonstrates an optimal F1 operating point of 0.82 at confidence threshold $\tau = 0.412$.
5. **High-Dimensional Manifold Projections (`assets/embeddings_tsne_pca.png`)**:
   - **PCA**: Demonstrates that the first two principal components capture 39.81% of variance ($PC_1 = 30.41\%, PC_2 = 9.40\%$).
   - **t-SNE**: Maps high-dimensional deep feature activations into non-linear 2D clusters, verifying distinct separation between rigid planar items (notebooks, laptops) and cylindrical containers (bottles, mugs).
6. **Bayesian Hyperparameter Optimization (`assets/optuna_study.png`)**:
   - Traces the validation mAP@50 trajectory across Bayesian TPE sampling iterations.
   - Highlights the convergence of learning rate, weight decay, momentum, and loss weight gains, logged in `optuna_study_results.json`.

---

## 5. Results & Comparative Analysis

### 5.1 Quantitative Performance Benchmarks

Comparative evaluation was conducted on identical validation test scenes comparing the COCO-80 baseline (YOLO11n), zero-shot open-vocabulary detection (YOLO-World v2), and the fine-tuned YOLO11m architecture:

| Model Architecture | Parameter Count | Latency (Mean) | Peak VRAM | Precision (B) | Recall (B) | mAP@50 | Target Vocabulary Scope |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLO11n (Baseline)** | 2.6M | 32.84 ms | Negligible | 24.7% | 31.0% | 27.5% | Closed-set 80 COCO classes |
| **YOLO-World v2** | 13.4M | 30.76 ms | Negligible | 55.2% | 27.6% | 36.8% | Dynamic Open-Vocabulary |
| **Fine-Tuned YOLO11m** | **20.1M** | **13.50 ms** | **4.19 GB** | **98.0%** | **80.2%** | **83.7%** | **Targeted 16 Everyday Classes** |

### 5.2 Key Empirical Findings
1. **Dramatic Precision Gain**: Fine-tuning YOLO11m yielded a **+73.3% precision increase** over the COCO baseline and a **+42.8% increase** over zero-shot YOLO-World.
2. **False Positive Elimination**: The baseline COCO model generated 55 false positives on cluttered desk scenes, misidentifying pens and chargers as generic non-desk artifacts. Fine-tuned YOLO11m reduced false detections to near-zero.
3. **Inference Latency Advantage**: Fused convolutional inference on YOLO11m executed in **13.5 ms** (versus 30.7 ms on YOLO-World), enabling $74$ FPS continuous processing.

---

## 6. Deployment & Containerization

### 6.1 FastAPI Production REST Service (`api.py`)
Provides production-grade inference endpoints with Pydantic v2 data validation schemas:
- `POST /predict`: Handles multipart image uploads, executing TensorRT/CUDA inference and returning typed pixel coordinates, normalized ratios, confidence, and inference latency.
- `POST /qa`: Dispatches natural language inquiries to the contextual reasoning layer.
- `GET /health` & `GET /classes`: Real-time monitoring reporting GPU telemetry, VRAM allocation, and class index registries.

### 6.2 Docker Containerization (`Dockerfile` & `docker-compose.yml`)
- Multi-stage Linux container based on `python:3.11-slim` with optimized OpenCV dependencies (`libgl1`, `libglib2.0-0`).
- Dual-port orchestration exposing the Gradio interface on port 7860 and FastAPI REST endpoints on port 8000.
- Fully compatible with one-click deployment to Hugging Face Spaces and AWS ECS/EC2.

---

## 7. Conclusion & Future Work

This project researched, engineered, fine-tuned, benchmarked, and deployed a production-grade machine learning system for real-time everyday object detection. By combining open-vocabulary zero-shot capabilities with fine-tuned supervised convolutional modeling, the solution achieved a state-of-the-art **83.7% mAP@50** and **98.0% precision** with sub-15ms edge inference.

Future extensions include:
1. Incorporating temporal optical flow tracking for video sequence smoothing.
2. Quantizing weights to INT8 via TensorRT for ultra-low-power microcontrollers.
3. Integrating on-device quantized multimodal LLMs for offline voice synthesis without cloud fallback dependencies.

---

## 8. References

```
[1] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 779-788.
[2] G. Jocher and J. Qiu, "Ultralytics YOLO11," Ultralytics Software Release, 2024. [Online]. Available: https://github.com/ultralytics/ultralytics
[3] T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan, "YOLO-World: Real-Time Open-Vocabulary Object Detection," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 16901-16911.
[4] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, "Microsoft COCO: Common Objects in Context," in Proc. Eur. Conf. Comput. Vis. (ECCV), 2014, pp. 740-755.
[5] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, et al., "Learning Transferable Visual Models From Natural Language Supervision," in Proc. Int. Conf. Mach. Learn. (ICML), 2021, pp. 8748-8763.
[6] Z. Zheng, P. Wang, W. Liu, J. Li, R. Ye, and D. Ren, "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression," in Proc. AAAI Conf. Artif. Intell., vol. 34, no. 7, 2020, pp. 12993-13000.
[7] X. Li, W. Wang, L. Wu, S. Chen, X. Hu, J. Li, J. Tang, and J. Yang, "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection," in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, 2020, pp. 21002-21012.
[8] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A Next-Generation Hyperparameter Optimization Framework," in Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD), 2019, pp. 2623-2631.
[9] L. van der Maaten and G. Hinton, "Visualizing Data using t-SNE," J. Mach. Learn. Res., vol. 9, pp. 2579-2605, 2008.
[10] S. Ramirez, "FastAPI: High performance, easy to learn, fast to code, ready for production," 2018. [Online]. Available: https://fastapi.tiangolo.com/
```
