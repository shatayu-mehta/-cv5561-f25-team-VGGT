# VGGT Implementation - Computer Vision Project

This repository contains our implementation and extensions of Visual Geometry Grounded Transformer (VGGT) for 3D reconstruction and scene understanding. We implement two methods for 3D object segmentation and reconstruction:
- **Method A**: YOLO + SAM with 3D lifting
- **Method B**: Geometric propagation-based segmentation

## Repository Structure

```
├── vggt/                          # Main implementation directory
│   ├── vggt_infererence.py        # VGGT inference and point cloud processing
│   ├── Yolo_SAM.py                # Method A: YOLO+SAM segmentation with 3D lifting
│   ├── propogation.py             # Method B: Geometric propagation segmentation
│   ├── viser_visualization.py     # Interactive 3D visualization tool
│   ├── scene_priors.json          # Scene configuration and object categories
│   └── ...                        # Additional utility scripts
├── VGGT_Evaluation/               # Depth evaluation against TUM RGB-D ground truth
│   ├── evaluate_depth_multiple.py # Script for computing depth metrics
│   ├── Datasets/                  # TUM RGB-D timestamped images and ground truth depth
│   │   ├── Cabinet/               # Cabinet scene with aligned GT depth
│   │   └── Plates/                # Plates scene with aligned GT depth
│   ├── results_vggt_cabinet/      # Evaluation results and metric plots for Cabinet
│   └── results_vggt_Plates/       # Evaluation results and metric plots for Plates
└── my_datasets/                   # Input datasets (images only, organized by scene)
```

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

### 1. **`vggt_infererence.py`** - Main 3D Reconstruction Pipeline
Our robust implementation of VGGT for 3D reconstruction with enhanced error handling and processing.

```bash
python vggt/vggt_infererence.py --scene_dir /path/to/your/images/
```



## Complete Workflow Example

Here's a complete pipeline from images to visualization:

### Step 1: Prepare Your Dataset

Organize your images in the following structure:
```
my_datasets/your_scene/
└── images/
    ├── frame_001.jpg
    ├── frame_002.jpg
    └── ...
```

### Step 2: Run VGGT Inference

Generate depth maps and camera poses:
```bash
cd vggt
python vggt_infererence.py --scene_dir ../my_datasets/your_scene/
```

This creates:
- `your_scene/outputs/depths/` - Depth maps for each frame
- `your_scene/outputs/poses/` - Camera intrinsics and extrinsics

### Step 3: Run Segmentation (Choose Method A or B)

**Method A: YOLO+SAM (Recommended for distinct objects)**
```bash
python Yolo_SAM.py --scene_dir ../my_datasets/your_scene/ --target_object "cabinet"
```

**Method B: Geometric Propagation (Better for texture-less objects)**
```bash
python propogation.py --scene_dir ../my_datasets/your_scene/ --target_object "cabinet"
```

### Step 4: Visualize Results

Launch interactive 3D viewer:
```bash
python viser_visualization.py --scene_dir ../my_datasets/your_scene/
```

Open your browser to `http://localhost:8080` to explore the 3D reconstruction.

## Project Features

- **3D Scene Reconstruction**: Feed-forward 3D reconstruction from multiple views
- **Camera Pose Estimation**: Automatic camera intrinsic and extrinsic estimation
- **Depth Map Generation**: Dense depth estimation for each view
- **Point Cloud Generation**: 3D point cloud reconstruction with confidence scores
- **Object Segmentation**: Integration with SAM for precise mask generation
- **Scene Understanding**: Semantic scene priors for improved reconstruction

## Implementation Details

### Method A: YOLO+SAM Segmentation
- **Detection**: Uses YOLO-World for open-vocabulary object detection
- **Segmentation**: Applies SAM (Segment Anything Model) for precise mask generation
- **3D Lifting**: Projects 2D masks to 3D using VGGT depth estimates
- **Advantages**: Accurate for well-textured objects with clear boundaries

### Method B: Geometric Propagation
- **Anchor Selection**: Detects objects in reference frame using YOLO+SAM
- **Propagation**: Uses homography and depth information to propagate masks across frames
- **Refinement**: Applies morphological operations and confidence thresholding
- **Advantages**: More robust for texture-less objects and consistent across frames

### Code Organization

The codebase is organized into logical modules:

1. **Inference Module** (`vggt_infererence.py`): Core VGGT inference and point cloud processing
2. **Segmentation Modules**:
   - `Yolo_SAM.py`: Detection-based segmentation (Method A)
   - `propogation.py`: Propagation-based segmentation (Method B)
3. **Visualization Module** (`viser_visualization.py`): Interactive 3D rendering
4. **Configuration** (`scene_priors.json`): Scene-specific parameters and object categories
5. **Utilities** (`visual_util.py`, `yolo_world.py`): Helper functions for processing and visualization

### Documentation

All major functions include docstrings and comments explaining:
- Input/output specifications
- Algorithm steps and logic
- Parameter meanings and default values
- Usage examples

## GPU Requirements

- Minimum: NVIDIA GPU with 8GB VRAM (for small scenes, 1-10 images)
- Recommended: 16GB+ VRAM for larger scenes (20+ images)
- Optimal: NVIDIA H100 or A100 for large-scale reconstructions (100+ images)

## VGGT Depth Evaluation

The `VGGT_Evaluation/` directory contains our evaluation of VGGT depth predictions against TUM RGB-D ground truth data.

### What It Does

This evaluation measures the accuracy of VGGT's depth estimation by comparing predicted depth maps against ground truth depth from the TUM RGB-D dataset. The evaluation computes standard depth metrics including:

- **Absolute Relative Error (abs_rel)**: Mean absolute relative difference
- **Square Relative Error (sq_rel)**: Mean squared relative difference  
- **RMSE**: Root Mean Square Error
- **RMSE log**: RMSE in log space
- **MAE**: Mean Absolute Error
- **SILog**: Scale-Invariant Logarithmic Error
- **δ < 1.25, δ < 1.25², δ < 1.25³**: Threshold accuracy (percentage of pixels within threshold)

### Why Use Only These Files

The datasets in `VGGT_Evaluation/Datasets/` are specifically prepared for accurate evaluation:

1. **Timestamped Images**: RGB images from TUM RGB-D with precise timestamps (e.g., `1341841279.510715.png`)
2. **Aligned Ground Truth**: Ground truth depth maps with matching timestamps (e.g., `1341841279.510727.png`)
3. **Manual Alignment**: Some images were manually renamed to ensure proper timestamp correspondence between RGB and depth
4. **Same Format**: All data follows TUM RGB-D conventions (depth in uint16, scale factor 5000.0)

Using these pre-aligned datasets ensures:
- Correct frame-to-frame correspondence between predicted and ground truth depth
- Proper scale alignment for fair comparison
- Reproducible evaluation results

### Running the Evaluation

```bash
cd VGGT_Evaluation

# Evaluate a single scene (e.g., Cabinet)
python evaluate_depth_multiple.py \
  --gt_folder Datasets/Cabinet/GT_Depth \
  --pred_folder Datasets/Cabinet/VGGT_Predicted_Depth \
  --output_dir results_vggt_cabinet \
  --depth_scale 5000.0 \
  --align_method median
```

### Output

The evaluation generates:
- **Metric plots**: Individual curves for each metric showing per-frame performance and cumulative mean
- **Summary statistics**: Aggregated metrics across all frames
- **Visual comparisons**: Side-by-side depth map comparisons

Results are saved in `results_vggt_cabinet/` and `results_vggt_Plates/` with metric curve plots showing VGGT's depth prediction accuracy.

## Datasets

### Included Example Dataset
- **Cabinet Scene**: 24 frames from TUM RGB-D dataset
- Located in: `vggt/example_data/Cabinet/images/`

### External Datasets Used
- **TUM RGB-D Dataset**: [https://cvg.cit.tum.de/data/datasets/rgbd-dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
  - Used for evaluation and testing
  - Provides ground truth depth and camera poses

### Custom Datasets
All custom datasets are stored in `my_datasets/` with images only. Large files (depth maps, point clouds) are excluded from git and generated locally.

**Google Drive Link for Complete Data**: [https://drive.google.com/drive/folders/17klTdjohqyWgIolCivYiltYGwM5Vg-RT?usp=sharing](https://drive.google.com/drive/folders/17klTdjohqyWgIolCivYiltYGwM5Vg-RT?usp=sharing)

## Troubleshooting

### Out of Memory Errors
- Reduce batch size in processing scripts
- Process fewer images at once (e.g., 10-15 frames)
- Use lower resolution images
- Close other GPU applications

### Model Download Issues
- Manually download model weights from Google Drive link above
- Ensure internet connection for automatic Hugging Face downloads
- Check firewall settings if downloads fail

### Visualization Not Loading
- Ensure viser is installed: `pip install viser`
- Check that port 8080 is not in use
- Try a different port: add `--port 8081` argument

### Segmentation Quality Issues
- Verify scene_priors.json contains your target object
- Try both Method A and Method B - they work better for different object types
- Ensure adequate lighting and texture in input images
- Check that depth maps are being generated correctly

## Citations and References

This project builds upon several state-of-the-art research works and codebases:

### Primary Research Paper
```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

### External Codebases and Models

1. **VGGT**: Visual Geometry Grounded Transformer
   - Repository: [https://github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt)
   - Used for: Core 3D reconstruction and depth estimation

2. **YOLO-World**: Open-Vocabulary Object Detection
   - Repository: [https://github.com/AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World)
   - Used for: Object detection in Method A

3. **Segment Anything Model (SAM)**
   - Repository: [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
   - Used for: Precise segmentation mask generation

4. **Viser**: 3D Visualization Library
   - Repository: [https://github.com/nerfstudio-project/viser](https://github.com/nerfstudio-project/viser)
   - Used for: Interactive 3D point cloud visualization

### Additional References

- **Depth Anything**: Used for alternative depth estimation experiments
- **OpenCV**: Camera geometry and image processing utilities

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
