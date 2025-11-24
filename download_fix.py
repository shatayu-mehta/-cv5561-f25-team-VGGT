import argparse

from co3d.dataset.download_dataset_impl import download_dataset, build_arg_parser



if __name__ == "__main__":

    # Use the library's existing argument parser

    parser = build_arg_parser()

    args = parser.parse_args()

    

    # Run the download

    print(f"Starting download to: {args.download_folder}")

    download_dataset(args)
