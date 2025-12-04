import argparse
import numpy as np
from metrics.completeness import completeness_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--th", type=float, default=0.05)
    args = parser.parse_args()

    pred = np.load(args.pred)
    gt = np.load(args.gt)

    score = completeness_score(pred, gt, threshold=args.th)
    print("Completeness Score:", score)

if __name__ == "__main__":
    main()
