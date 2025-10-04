# Q1 — Vision Transformer Fine-tuning on CIFAR-10

## Overview
This project implements fine-tuning of a pretrained Vision Transformer (ViT-B/16) model on the CIFAR-10 dataset using PyTorch. The implementation achieves **97.28% test accuracy** through strategic layer unfreezing, advanced data augmentation, and modern training techniques including mixed precision training, cosine annealing with warmup, and label smoothing.

## Features
* **Transfer Learning**: Leverages pretrained ViT-B/16 weights from ImageNet-1K with selective layer unfreezing (last 3 transformer blocks)
* **Advanced Augmentation**: Implements comprehensive data augmentation pipeline including RandomErasing, ColorJitter, and rotation
* **Mixed Precision Training**: Utilizes automatic mixed precision (AMP) for faster training and reduced memory usage
* **Learning Rate Scheduling**: Cosine annealing with 3-epoch warmup for optimal convergence
* **Comprehensive Visualization**: Generates confusion matrix, training curves, and per-class accuracy reports

## Pipeline Architecture

<img width="1759" height="1137" alt="q1_pipeline" src="https://github.com/user-attachments/assets/9aa6c730-e656-42ea-8f94-f623e3b736b8" />

## Technical Components

### Models Used
* **ViT-B/16**: Vision Transformer Base model with 16×16 patch size (85.8M total parameters, 21.3M trainable)
* **Pretrained Weights**: IMAGENET1K_V1 from torchvision
* **Custom Head**: Dropout(0.1) + Linear layer for 10-class classification

### Key Technologies
* **PyTorch**: Deep learning framework with CUDA acceleration
* **Mixed Precision Training**: `torch.cuda.amp` for GPU memory efficiency
* **Data Augmentation**: torchvision transforms with RandomErasing
* **Optimization**: AdamW optimizer with weight decay and gradient clipping
* **Visualization**: Matplotlib and Seaborn for training analysis

## Installation

### Prerequisites
* Python 3.8+
* CUDA-capable GPU (tested on Tesla T4 with 15GB VRAM)
* PyTorch 2.0+
* 16GB+ system RAM recommended

### Setup Commands

```bash
# Check GPU availability
!nvidia-smi

# Install required packages
!pip install timm einops torchsummary

# Import and verify installations
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Usage

### Basic Configuration

```python
# Hyperparameters (modify as needed)
CFG = {
    "epochs": 7,
    "batch_size": 128,         # Reduce to 64 if OOM
    "lr": 1e-4,                # Learning rate
    "weight_decay": 0.01,      # AdamW weight decay
    "num_workers": 4,          # DataLoader workers
    "use_amp": True,           # Mixed precision
    "patience": 7,             # Early stopping patience
    "label_smoothing": 0.1,    # Label smoothing factor
}
```

### Running the Pipeline
1. **Prepare Environment**: Upload the code to Google Colab or Jupyter notebook with GPU runtime
2. **Install Dependencies**: Run the pip install commands to get required packages
3. **Execute Main Script**: Run the complete training script (automatically downloads CIFAR-10)
4. **Monitor Training**: Progress bars show real-time loss, accuracy, and learning rate
5. **Collect Outputs**: Find saved model and visualizations in working directory

## Results

### Performance Metrics
* **Best Test Accuracy**: 97.28%
* **Final Train Accuracy**: 97.39%
* **Training Time**: ~45 minutes on Tesla T4 (7 epochs)
* **Trainable Parameters**: 21.3M (24.79% of total model)
* **Peak GPU Memory**: ~12GB

### Output Files
* **best_vit_b16_cifar10.pth**: Best model checkpoint with optimizer state
* **confusion_matrix.png**: 10×10 confusion matrix heatmap (300 DPI)
* **training_curves.png**: Loss, accuracy, and learning rate plots (300 DPI)

### Per-Class Accuracy
| Class      | Accuracy | Correct/Total |
|------------|----------|---------------|
| airplane   | 98.20%   | 982/1000      |
| automobile | 99.40%   | 994/1000      |
| bird       | 96.00%   | 960/1000      |
| cat        | 93.60%   | 936/1000      |
| deer       | 97.30%   | 973/1000      |
| dog        | 96.70%   | 967/1000      |
| frog       | 99.00%   | 990/1000      |
| horse      | 97.20%   | 972/1000      |
| ship       | 98.10%   | 981/1000      |
| truck      | 97.30%   | 973/1000      |

## Limitations
* **Image Resolution**: CIFAR-10's native 32×32 resolution requires upscaling to 224×224 for ViT
* **Training Time**: ViT requires more epochs than CNNs due to transformer architecture
* **Memory Constraints**: Batch size limited by GPU memory (128 works on T4, may need reduction)
* **Class Imbalance**: Cat class shows slightly lower accuracy (93.60%) compared to others

## Known Issues
* **FutureWarning**: `torch.cuda.amp.GradScaler` deprecation warnings (functionality works correctly)
* **num_workers > 0**: May cause issues on Windows; set to 0 if encountering DataLoader errors
* **Persistent Workers**: Requires stable Jupyter kernel; disable if notebook crashes

## Future Improvements
* **Data Efficiency**: Implement test-time augmentation (TTA) for improved accuracy
* **Model Variants**: Compare ViT-B/16 with ViT-L/16 or Swin Transformer architectures
* **Hyperparameter Tuning**: Grid search over learning rate, batch size, and unfrozen layers
* **Ensemble Methods**: Combine multiple checkpoints for prediction averaging
* **Export to ONNX**: Enable deployment on edge devices with ONNX runtime

## Dependencies
* torch >= 2.0.0
* torchvision >= 0.15.0
* numpy >= 1.20.0
* scikit-learn >= 1.0.0
* matplotlib >= 3.5.0
* seaborn >= 0.12.0
* tqdm >= 4.65.0
* timm >= 0.9.0 (optional, for additional models)

## References
* [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) - Dosovitskiy et al., ICLR 2021
* [PyTorch Vision Transformer Documentation](https://pytorch.org/vision/main/models/vision_transformer.html)
* [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) - Krizhevsky & Hinton
* [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html) - NVIDIA & PyTorch

---



# Q2 — Text-Driven Image Segmentation with SAM 2

## Overview
This project demonstrates text-prompted object segmentation and tracking using **SAM 2 (Segment Anything Model 2)** combined with **Grounding DINO** for zero-shot object detection. The implementation supports both single-image segmentation and video object tracking.

## Features
- **Text-Driven Detection**: Use natural language prompts to detect objects
- **Precise Segmentation**: Generate high-quality masks using SAM 2
- **Video Tracking**: Propagate masks across video frames for temporal consistency
- **Multiple Prompt Types**: Support for box, point, and mask-based prompts
- **Google Colab Ready**: Fully runnable in Colab with GPU acceleration

## Pipeline Architecture
<img width="5399" height="4140" alt="q2_Pipeline" src="https://github.com/user-attachments/assets/0b7858b6-7769-45c6-bada-3b5488f8d518" />

## Technical Components

### Models Used
- **SAM 2.1 (Hiera-Large)**: State-of-the-art segmentation model
- **Grounding DINO (Tiny)**: Zero-shot object detection from text
- **Supervision**: Annotation and visualization utilities

### Key Technologies
- PyTorch for deep learning inference
- OpenCV for video/image processing
- Transformers for model loading
- CUDA acceleration for GPU performance

## Installation

### Prerequisites
- Google Colab with GPU runtime (T4 or better recommended)
- ~2GB disk space for model checkpoints
- Python 3.10+

### Setup Commands

```bash
# Install packages
!pip install -q git+https://github.com/facebookresearch/segment-anything-2.git
!pip install -q supervision transformers timm opencv-python-headless pillow

# Download models
!mkdir -p checkpoints configs/sam2.1
!wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O checkpoints/sam2.1_hiera_large.pt
!wget -q https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml -O configs/sam2.1/sam2.1_hiera_l.yaml
```

## Usage

### Basic Configuration
```python
VIDEO_PATH = "/content/Cars.mp4"      # Input video path
TEXT_PROMPT = "car."                   # Detection prompt (lowercase + dot)
PROMPT_TYPE = "box"                    # Options: "box", "point", "mask"
BOX_THRESHOLD = 0.25                   # Detection confidence threshold
TEXT_THRESHOLD = 0.3                   # Text matching threshold
```

### Running the Pipeline
1. Upload your video to `/content/<your_video_file_name.mp4>`
2. Run installation cells (1-2)
3. Execute main code cell
4. Output video will be at `/content/<your_video_file_name_tracking_output.mp4>`

## Results

### Performance Metrics
- **Frame Processing**: ~25-30 FPS extraction
- **Detection Speed**: ~0.5s per frame
- **Tracking Speed**: ~2-3 FPS for propagation
- **Total Processing**: ~2-3 minutes for 10s video (300 frames)

### Output Files
- **Extracted Frames**: `/content/car_frames/`
- **Annotated Frames**: `/content/tracking_results/`
- **Output Video**: `/content/car_tracking_output.mp4`

# Actual video

https://github.com/user-attachments/assets/a1e11b1f-fc98-4016-9e47-519adc30b20b

# Output video

https://github.com/user-attachments/assets/ce758bc9-aa33-4612-a430-28e02cc3f2bf

## Limitations

- The model requires strict text prompts (lowercase with period) and struggles with nuanced language.
- Detection accuracy depends on threshold tuning, with issues on small, occluded, or ambiguous objects.
- High GPU and memory demands make long or high-res videos computationally expensive.
- Tracking is weak under occlusion, fast motion, or re-appearance of objects.
- Objects may swap IDs or be confused when multiple similar ones appear across frames.


### Known Issues
- Grounding DINO may require different threshold values per scene
- SAM 2 masks can be oversensitive to background clutter
- Video codec compatibility may vary (mp4v used for broad support)

## Future Improvements
- Add automatic threshold tuning
- Implement object re-identification
- Support for multiple text prompts simultaneously
- Real-time processing optimization
- Interactive prompt refinement

## Dependencies
```
segment-anything-2 (from GitHub)
supervision >= 0.16.0
transformers >= 4.35.0
torch >= 2.0.0
opencv-python >= 4.8.0
Pillow >= 10.0.0
timm >= 0.9.0
tqdm >= 4.65.0
```

## References
- [SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/segment-anything-2)
- [Grounding DINO: Marrying DINO with Grounded Pre-Training](https://github.com/IDEA-Research/GroundingDINO)
- [Supervision: Reusable Computer Vision Tools](https://github.com/roboflow/supervision)

---
