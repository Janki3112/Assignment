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
# Cell 1: Install packages
!pip install -q git+https://github.com/facebookresearch/segment-anything-2.git
!pip install -q supervision transformers timm opencv-python-headless pillow

# Cell 2: Download models
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


Uploading car_tracking_output.mp4…



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

**Note**: This implementation demonstrates the bonus video extension feature for additional marks.
