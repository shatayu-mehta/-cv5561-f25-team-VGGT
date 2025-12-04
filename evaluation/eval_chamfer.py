import argparse
import numpy as np
from metrics.chamfer import chamfer_distance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    args = parser.parse_args()

    pred = np.load(args.pred)
    gt = np.load(args.gt)

    cd = chamfer_distance(pred, gt)
    print("Chamfer Distance:", cd)

if __name__ == "__main__":
    main()
