"""
data2torchformat.py - converts the dataframes to something that torch can use
"""
# %%

from pathlib import Path

import pandas as pd


mit_map = {
    0:"N",
    1:"S",
    2:"V",
    3:"F",
    4:"Q",
}

# %%

def map_to_letters(numclass):
    """
    maps the class labels to letters
    """
    numclass = int(numclass)
    assert numclass in mit_map.keys(), f"{numclass} not in {mit_map.keys()}"
    return mit_map[numclass]

def mitbih_to_torchformat(data_dir, out_dir=None):
    """
    Converts the mitbih dataset to a format that torch can use
    """
    data_dir = Path(data_dir)
    if out_dir is None:
        out_dir = data_dir / 'torch_format'
        out_dir.mkdir(exist_ok=True)

    mit_files = [f for f in data_dir.iterdir() if f.is_file() and "mitbih" in f.name and f.suffix == ".csv"]

    for mit_file in mit_files:
        df = pd.read_csv(mit_file, header=None).convert_dtypes()
        _cols = list(df.columns)
        _cols =[f"feat_{c}" for c in _cols]
        # update the last column name to be class label
        _cols[-1] = "class_label"
        df.columns = _cols
        df["class_label"] = df["class_label"].apply(map_to_letters)
        df.to_csv(out_dir / f"torchfmt_{mit_file.name}", index=False)

    return out_dir

# %%
# reformat the mitbih dataset
_root = Path(__file__).parent

_data_dir = _root / 'data'

mit_out = mitbih_to_torchformat(_data_dir)
print(f"wrotefiles to {mit_out.resolve()}")

# %%


def ptbdb_to_torchformat(data_dir, out_dir=None):
    """
    Converts the ptbdb dataset to a format that torch can use
    """
    data_dir = Path(data_dir)
    if out_dir is None:
        out_dir = data_dir / 'torch_format'
        out_dir.mkdir(exist_ok=True)

    pt_files = [f for f in data_dir.iterdir() if f.is_file() and "ptbdb" in f.name and f.suffix == ".csv"]

    full_data = pd.DataFrame()
    for pt_file in pt_files:
        df = pd.read_csv(pt_file, header=None).convert_dtypes()
        _cols = list(df.columns)
        _cols =[f"feat_{c}" for c in _cols]
        # update the last column name to be class label
        _cols[-1] = "class_label"
        df.columns = _cols
        df["class_label"] = "abnormal" if "abnormal" in pt_file.name else "normal"
        full_data = full_data.append(df, ignore_index=True)

    full_data.to_csv(out_dir / "torchfmt_ptbdb_full.csv", index=False)

    return out_dir

# %%

pt_out = ptbdb_to_torchformat(_data_dir)
print(f"wrotefiles to {pt_out.resolve()}")

# %%
