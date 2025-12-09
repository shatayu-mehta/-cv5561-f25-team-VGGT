# Model Weights and Large Files

Due to GitHub's file size limitations, the model weights and large files are hosted on Google Drive.

## Required Model Files

Download the following files and place them in the `vggt/` directory:

### Model Weights (Required)
- `sam_b.pt` (357 MB) - SAM (Segment Anything Model) weights
- `yolo11x-seg.pt` (119 MB) - YOLO11 segmentation model
- `yolov8l-world.pt` (91 MB) - YOLO v8 world model
- `yolov8n-seg.pt` - YOLO v8 nano segmentation model

### Google Drive Link
**Download all model files from:** https://drive.google.com/drive/folders/17klTdjohqyWgIolCivYiltYGwM5Vg-RT?usp=sharing

## Installation Steps

1. Clone this repository:
```bash
git clone https://github.com/shatayu-mehta/-cv5561-f25-team-VGGT.git
cd -cv5561-f25-team-VGGT/vggt
```

2. Download the model weights from the Google Drive link above

3. Place the downloaded `.pt` files in the `vggt/` directory

4. Verify the files are in place:
```bash
ls -lh *.pt
```

You should see:
- sam_b.pt
- yolo11x-seg.pt
- yolov8l-world.pt
- yolov8n-seg.pt

5. Install dependencies and run the code as usual

## Optional: Example Dataset
The `examples/` folder contains sample images. For full datasets, download from the Google Drive link above.

## Questions?
Contact the repository maintainers if you have issues downloading or using the model weights.
