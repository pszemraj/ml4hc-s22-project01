"""
    download_data.py - downloads the data from the URL and saves it to the specified directory
"""
import argparse
import os
import shutil
from pathlib import Path
import requests
from tqdm.auto import tqdm

from utils.general_utils import clean_file_name


def download_URL(url: str, file=None, dlpath=None, verbose=False):
    """
    download_URL - download a file from a URL and show progress bar

    Parameters
    ----------
    url : str,        URL to download
    file : str, optional, default None, name of file to save to. If None, will use the filename from the URL
    dlpath : str, optional, default None, path to save the file to. If None, will save to the current working directory
    verbose : bool, optional, default False, print progress bar

    Returns
    -------
    str - path to the downloaded file
    """

    if file is None:
        if "?dl=" in url:
            # is a dropbox link
            prefile = url.split("/")[-1]
            filename = str(prefile).split("?dl=")[0]
        else:
            filename = url.split("/")[-1]

        file = clean_file_name(filename)
    if dlpath is None:
        dlpath = Path.cwd()  # save to current working directory
    else:
        dlpath = Path(dlpath)  # make a path object

    r = requests.get(url, stream=True, allow_redirects=True)
    total_size = int(r.headers.get("content-length"))
    initial_pos = 0
    dl_loc = dlpath / file
    with open(str(dl_loc.resolve()), "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=file,
            initial=initial_pos,
            ascii=True,
        ) as pbar:
            for ch in r.iter_content(chunk_size=1024):
                if ch:
                    f.write(ch)
                    pbar.update(len(ch))

    if verbose:
        print(f"\ndownloaded {file} to {dlpath}\n")

    return str(dl_loc.resolve())


def get_parser():
    """
    get_parser - a helper function for the argparse module
    """
    parser = argparse.ArgumentParser(
        description="Convert a directory of dataframes to a directory of dataframes in a more memory efficient format"
    )
    parser.add_argument(
        "-u",
        "--url",
        required=False,
        default="https://www.dropbox.com/sh/hwv3msz2mdfxki1/AACOk6t8z6hNfuc3s7xM9K7-a?dl=1",
        type=str,
        help="URL to a zip file containing the dataframes. File should auto-download when loaded (in a browser)",
    )
    parser.add_argument(
        "-f",
        "--force",
        required=False,
        default=False,
        action="store_true",
        help="force download and overwrite of existing files",
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
    url = args.url
    force = args.force
    verbose = args.verbose
    # using pathlib, check if the 'data' folder exists with CSV files in the local dir. If not, download the data
    _root = Path(__file__).parent
    data_dir = _root / "data"
    data_dir.mkdir(exist_ok=True)
    csv_files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix == ".csv"]
    if force or len(csv_files) == 0:
        data_archive_path = download_URL(url, file="ml4hc_p1_data.zip", verbose=True)
        if verbose:
            print(f"extracting {data_archive_path}")
        shutil.unpack_archive(data_archive_path, extract_dir=str(data_dir.resolve()))
        if verbose:
            print(f"removing {data_archive_path}")
        os.remove(data_archive_path)
    else:
        print(f"data already downloaded to {data_dir}, found {len(csv_files)} files")
        print("use -f to force download")

    print(f"\n\tfinished downloading data to {data_dir}")
