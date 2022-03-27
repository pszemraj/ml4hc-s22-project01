"""
    shrink_dataframe_files.py - given a directory of dataframes, read the dataframes and store them in a new sub-directory in a more memory efficient format
"""

import argparse
from pathlib import Path

import pandas as pd
from general_utils import downcast_all, all_float_to_int

def create_sm_datafile(input_file, out_dir=None, no_headers=True, verbose=False):
    """
    create_sm_datafile - given a dataframe, create a smaller version of the dataframe that is more memory efficient

    Parameters
    ----------
    input_file : str, Path, the path to the input dataframe
    out_dir : _type_, optional, the path to the output directory, by default None
    no_headers : bool, optional, if the input files do not have headers, set this flag, by default True
    verbose : bool, optional, print verbose output, by default False

    Returns
    -------
    Path, the path to the output dataframe
    """
    input_file = Path(input_file)
    if out_dir is None:
        out_dir = input_file.parent / "data_as_feather"
        out_dir.mkdir(exist_ok=True)

    df = (
        pd.read_csv(input_file, header=None).convert_dtypes()
        if no_headers
        else pd.read_csv(input_file).convert_dtypes()
    )
    if no_headers:
        # update all column names to be feat_<column name>
        df.columns = [f"feat_{c}" for c in df.columns]
    if verbose:
        print(f"{input_file.name} has {len(df)} rows")

    sm_df = (
        df.pipe(all_float_to_int)
        .pipe(downcast_all, "float")
        .pipe(downcast_all, "integer")
        .pipe(downcast_all, target_type="unsigned", inital_type="integer")
    )
    sm_df.to_feather(out_dir / f"{input_file.stem}.ftr")

    return out_dir


def get_parser():
    """
    get_parser - a helper function for the argparse module
    """
    parser = argparse.ArgumentParser(
        description="Convert a directory of dataframes to a directory of dataframes in a more memory efficient format"
    )
    parser.add_argument(
        "-i",
        "--input-path",
        required=True,
        type=str,
        help="path to the input directory",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=False,
        default=None,
        help="path to the output directory",
    )
    parser.add_argument(
        "-n",
        "--no-headers",
        default=False,
        action="store_true",
        help="if the input files do not have headers, set this flag",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        action="store_true",
        help="print verbose output",
    )
    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()
    input_path = Path(args.input_path)
    output_path = (
        Path(args.output_path) if args.output_path else input_path / "data_as_feather"
    )
    output_path.mkdir(exist_ok=True)

    no_headers = args.no_headers
    verbose = args.verbose

    csv_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix == ".csv"]

    for csv_file in csv_files:
        if verbose:
            print(f"processing {csv_file.name}")
        _ = create_sm_datafile(csv_file, out_dir=output_path, no_headers=no_headers)

    print(f"\nWrote files to {output_path.resolve()}")
