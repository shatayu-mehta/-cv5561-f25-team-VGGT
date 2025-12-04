import argparse
import numpy as np
from metrics.segmentation_eval import compute_iou

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    args = parser.parse_args()

    pred = np.load(args.pred)
    gt = np.load(args.gt)

    iou = compute_iou(pred, gt)
    print("IoU:", iou)

if __name__ == "__main__":
    main()
