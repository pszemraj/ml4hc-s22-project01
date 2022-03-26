# %% [markdown]
# # ml4hc - project 1 eda
#
# > purpose: explore the data and identify the features that are most relevant to the problem as well as the features that are least relevant to the problem, and what the columns mean etc

# %%
import pandas as pd
from pathlib import Path

_root = Path(__file__).parent.parent

_data_dir = _root / "data"

# list of all files in the data directory
_files = [f for f in _data_dir.iterdir() if f.is_file()]
print(f"Found {len(_files)} files in {_data_dir}")


# %%

# load report files for ptbdb
import pprint as pp

ptbdb_ab = pd.read_csv(_data_dir / "ptbdb_abnormal.csv", header=None).convert_dtypes()
ptbdb_norm = pd.read_csv(_data_dir / "ptbdb_normal.csv", header=None).convert_dtypes()

print(f"info for ptbdb_abnormal.csv: ")
pp.pprint(ptbdb_ab.info())
print(f"\n\ninfo for ptbdb_normal.csv: {ptbdb_norm.info()}")
pp.pprint(ptbdb_norm.info())
# %%
# %% [markdown]
# ## ml4hc - project 1 eda - dataset2
#

# %%
# load the mitbih dataset
mitbih_train = pd.read_csv(_data_dir / "mitbih_train.csv").convert_dtypes()
mitbih_test = pd.read_csv(_data_dir / "mitbih_test.csv").convert_dtypes()

print(f"info for mitbih_train.csv: {mitbih_train.info()}")
print(f"info for mitbih_test.csv: {mitbih_test.info()}")
# %%
