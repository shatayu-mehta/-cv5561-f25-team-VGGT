# VGGT Implementation - Computer Vision Project

This repository contains our implementation and extensions of Visual Geometry Grounded Transformer (VGGT) for 3D reconstruction and scene understanding.

## Repository Structure

- `vggt/` - Main VGGT implementation (see [vggt/README.md](vggt/README.md) for details)
- `my_datasets/` - Local datasets for testing (images available via Google Drive)
- `Cabinet_new/` - Cabinet scene dataset and processing results
- `sam2/` - SAM2 segmentation model integration

## Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/shatayu-mehta/-cv5561-f25-team-VGGT.git
cd -cv5561-f25-team-VGGT
```

### 2. Install Dependencies

```bash
cd vggt
pip install -r requirements.txt
pip install -r requirements_demo.txt  # For visualization tools
```

### 3. Download Model Weights and Example Data

Large model files and example datasets are hosted on Google Drive to keep the repository lightweight.

**Google Drive Link:** [https://drive.google.com/drive/folders/17klTdjohqyWgIolCivYiltYGwM5Vg-RT?usp=sharing](https://drive.google.com/drive/folders/17klTdjohqyWgIolCivYiltYGwM5Vg-RT?usp=sharing)

Download the `vggt_models_and_example.tar.gz` file and extract it:

```bash
# Download the tar.gz file from the Google Drive link above
# Then extract it in the vggt directory:
cd vggt
tar -xzf /path/to/downloaded/vggt_models_and_example.tar.gz

# This will create:
# - Model weights (*.pt files)
# - example_data/Cabinet/images/ (example image dataset)
```

The tar file contains:
- `sam_b.pt` (358 MB) - SAM segmentation model
- `yolo11x-seg.pt` (120 MB) - YOLO11 segmentation model
- `yolov8l-world.pt` (92 MB) - YOLO v8 world model
- `yolov8n-seg.pt` (6.8 MB) - YOLO v8 nano model
- `example_data/Cabinet/images/` - Example image dataset

## Key Custom Scripts (Useful for Other Users)

The following files contain our custom implementations and are recommended for use:

### 1. **`vggt_robust.py`** - Main 3D Reconstruction Pipeline
Our robust implementation of VGGT for 3D reconstruction with enhanced error handling and processing.

```bash
python vggt/vggt_robust.py --scene_dir /path/to/your/images/
```

### 2. **`own_visualization.py`** - Custom Visualization Tools
Enhanced visualization utilities for 3D point clouds and reconstruction results.

```bash
python vggt/own_visualization.py --scene_dir /path/to/scene/
```

### 3. **`scene_priors.json`** - Scene Configuration
JSON configuration file containing scene priors, object categories, and semantic information for reconstruction.

Located at: `vggt/scene_priors.json`

### 4. **`viser_vis.py`** - Interactive 3D Viewer
Interactive 3D visualization using Viser for exploring reconstruction results.

```bash
python vggt/viser_vis.py --scene_dir /path/to/scene/
```

**Note:** Other files in the repository are experimental or work-in-progress and may not be fully functional.

## Basic Usage Example

```python
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize model
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load your images
image_names = ["path/to/imageA.png", "path/to/imageB.png"]  
images = load_and_preprocess_images(image_names).to(device)

# Run reconstruction
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)
```

## Running 3D Reconstruction

### Using Our Custom Pipeline

```bash
# Basic reconstruction
python vggt/vggt_robust.py --scene_dir /path/to/images/

# With visualization
python vggt/viser_vis.py --scene_dir /path/to/images/
```

### Using Original VGGT Tools

See detailed documentation in [vggt/README.md](vggt/README.md) for:
- Gradio web interface: `python demo_gradio.py`
- Viser 3D viewer: `python demo_viser.py --image_folder path/to/images/`
- COLMAP export: `python demo_colmap.py --scene_dir=/path/to/scene/`

## Project Features

- **3D Scene Reconstruction**: Feed-forward 3D reconstruction from multiple views
- **Camera Pose Estimation**: Automatic camera intrinsic and extrinsic estimation
- **Depth Map Generation**: Dense depth estimation for each view
- **Point Cloud Generation**: 3D point cloud reconstruction with confidence scores
- **Object Segmentation**: Integration with SAM2 for object-aware processing
- **Scene Understanding**: Semantic scene priors for improved reconstruction

## Dataset Structure

For best results, organize your data as follows:

```
your_scene/
├── images/
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── ...
└── (outputs will be generated here)
```

## GPU Requirements

- Minimum: NVIDIA GPU with 8GB VRAM (for small scenes, 1-10 images)
- Recommended: 16GB+ VRAM for larger scenes (20+ images)
- Optimal: NVIDIA H100 or A100 for large-scale reconstructions (100+ images)

## Troubleshooting

### Out of Memory Errors
- Reduce batch size in processing scripts
- Process fewer images at once
- Use lower resolution images

### Model Download Issues
- Manually download model weights from Google Drive link above
- Ensure internet connection for automatic Hugging Face downloads

## Citation

If you use this code, please cite the original VGGT paper:

```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## License

This project follows the VGGT license. See [vggt/LICENSE.txt](vggt/LICENSE.txt) for details.

## Team

CSCI 5561 Computer Vision - Fall 2025  
University of Minnesota

## Additional Resources

- **Original VGGT Repository**: [https://github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt)
- **VGGT Project Page**: [https://vgg-t.github.io/](https://vgg-t.github.io/)
- **Hugging Face Demo**: [https://huggingface.co/spaces/facebook/vggt](https://huggingface.co/spaces/facebook/vggt)
- **Paper**: [Visual Geometry Grounded Transformer (CVPR 2025)](https://arxiv.org/abs/2503.11651)

## Support

For issues or questions:
1. Check the [original VGGT documentation](vggt/README.md)
2. Review example scripts in the repository
3. Ensure all dependencies are properly installed
4. Verify model weights are downloaded correctly
