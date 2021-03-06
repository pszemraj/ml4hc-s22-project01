"""
data2torchformat.py - converts the provided data to a format that torch can use (adding class labels and column names)
"""
# %%

from pathlib import Path

import pandas as pd

mit_map = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}

# %%


def map_to_letters(numclass):
    """
    map_to_letters - maps the numbers to letters

    Parameters
    ----------
    numclass : int, a number ID for the class

    Returns
    -------
    str, a letter ID for the class
    """
    numclass = int(numclass)
    assert numclass in mit_map.keys(), f"{numclass} not in {mit_map.keys()}"
    return mit_map[numclass]


def mitbih_to_torchformat(data_dir, out_dir=None):
    """
    mitbih_to_torchformat - converts the mitbih dataset to a format that torch can use

    Parameters
    ----------
    data_dir : _pathlib.Path, the directory containing the mitbih dataset
    out_dir : _pathlib.Path, the directory to write the torch formatted dataset to, defaults to data_dir/torch_format

    Returns
    -------
    _pathlib.Path, the directory containing the torch formatted dataset
    """

    data_dir = Path(data_dir)
    if out_dir is None:
        out_dir = data_dir / "torch_format"
        out_dir.mkdir(exist_ok=True)

    mit_files = [
        f
        for f in data_dir.iterdir()
        if f.is_file() and "mitbih" in f.name and f.suffix == ".csv"
    ]

    for mit_file in mit_files:
        df = pd.read_csv(mit_file, header=None).convert_dtypes()
        _cols = list(df.columns)
        _cols = [f"feat_{c}" for c in _cols]
        # update the last column name to be class label
        _cols[-1] = "class_label"
        df.columns = _cols
        df["class_label"] = df["class_label"].apply(map_to_letters)
        df.to_csv(out_dir / f"torchfmt_{mit_file.name}", index=False)

    return out_dir


# %%
## define paths
_root = Path(__file__).parent

_data_dir = _root / "data"

# %%
# reformat the mitbih dataset

mit_out = mitbih_to_torchformat(_data_dir)
print(f"wrotefiles to {mit_out.resolve()}")

# %%


def ptbdb_to_torchformat(data_dir, out_dir=None, random_state=42, create_test=False):
    """
    ptbdb_to_torchformat - converts the ptbdb dataset to a format that torch can use

    Parameters
    ----------
    data_dir : _pathlib.Path, the directory containing the ptbdb dataset
    out_dir : _pathlib.Path, the directory to write the torch formatted dataset to, defaults to data_dir/torch_format
    random_state : int, optional, the random state to use for splitting the dataset, defaults to 42
    create_test : bool, optional, whether to create a test set, defaults to False

    Returns
    -------
    _pathlib.Path, the directory containing the torch formatted dataset
    """
    data_dir = Path(data_dir)
    if out_dir is None:
        out_dir = data_dir / "torch_format"
        out_dir.mkdir(exist_ok=True)

    pt_files = [
        f
        for f in data_dir.iterdir()
        if f.is_file() and "ptbdb" in f.name and f.suffix == ".csv"
    ]

    full_data = pd.DataFrame()
    for pt_file in pt_files:
        df = pd.read_csv(pt_file, header=None).convert_dtypes()
        _cols = list(df.columns)
        _cols = [f"feat_{c}" for c in _cols]
        # update the last column name to be class label
        _cols[-1] = "class_label"
        df.columns = _cols
        df["class_label"] = "abnormal" if "abnormal" in pt_file.name else "normal"
        full_data = full_data.append(df, ignore_index=True)

    # shuffle the rows in the dataframe and write to csv
    if create_test:
        # create two randomly sampled dataframes from full_data
        train_df, test_df = full_data.sample(
            frac=0.8, random_state=random_state
        ), full_data.sample(frac=0.2, random_state=random_state)
        # write the train and test dataframes to csv
        train_df.to_csv(out_dir / "torchfmt_ptbdb_train.csv", index=False)
        test_df.to_csv(out_dir / "torchfmt_ptbdb_test.csv", index=False)
        print(f"wrote train and test files with random_state={random_state}")
    else:
        full_data = full_data.sample(frac=1, random_state=random_state).reset_index(
            drop=True
        )  # because merging two CSVs of one class each
        full_data.to_csv(out_dir / "torchfmt_ptbdb_full.csv", index=False)
        print(f"wrote ONE full file with random_state={random_state}")

    return out_dir


# %%
# reformat the ptbdb dataset

pt_out = ptbdb_to_torchformat(_data_dir, create_test=True)
print(f"wrotefiles to {pt_out.resolve()}")

# %%
