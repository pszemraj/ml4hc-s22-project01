"""
    notebooks\analysis\merge_compare_results.py - compares the results of different methods by loading the results from different methods and merging them (source folder: results)

    NOTE: this .py script is meant to be run as a jupytext notebook. for info on what that is, see https://jupytext.readthedocs.io/en/latest/faq.html

"""
# %%
from pathlib import Path
import pandas as pd

import pprint as pp

_root = Path(__file__).parent.parent.parent # two levels up
results_dir = _root / "results"
_results_subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
result_dirs = {d.name: d for d in _results_subdirs}
pp.pprint(list(result_dirs.keys()))


# %%

# LE GRAND MAPPING DICTIONNARY

metric_map = {
    'F1': 'f1_score',
    'Accuracy': 'accuracy',
    'MCC': 'matthews_corrcoef',
    'Balanced Accuracy': 'balanced_accuracy_score',
    'AUC': 'roc_auc_score',
    'Prec.': 'precision_score',
    'Recall': 'recall_score',
    'Kappa': 'cohen_kappa_score',

}
# %%

# handle initial-training
key_phrase = "model_metrics"
# load CSV paths in initial-training directory that contain the key phrase
_csv_paths = [f for f in result_dirs["initial-training"].iterdir() if f.is_file() and key_phrase in f.name.lower()]
print(f"{len(_csv_paths)} CSV files found in {result_dirs['initial-training']}")
# %%

# load all CSV files as dataframes and print the column names

def get_unique_cols(csv_paths:list):

    _dfs = [pd.read_csv(f) for f in csv_paths]
    _cols = [list(df.columns) for df in _dfs]
    # find the unique column names between all files and dataframes
    _cols = set.intersection(*map(set, _cols))
    return _cols

get_unique_cols(_csv_paths)

# %% [markdown]
# - the first datasets define mandatory mapping between the MITBIH classes and the letters used in the dataset

# `{'accuracy', 'balanced_accuracy_score', 'f1_score', 'matthews_corrcoef', 'model_filename'}`


# %%
# load each dataframe and pivot the dataframe to have the columns as the class labels, except for the model file name

overall_results = pd.DataFrame()
for f in _csv_paths:
    df = pd.read_csv(f)
    _cols = list(df.columns)
    _cols.remove("model_filename")
    # all column entries except for the model file name now become one column titled performance_metric using pd.melt
    df = pd.melt(df, id_vars=["model_filename"], value_vars=_cols, var_name="performance_metric", value_name="metric_value")
    df['dataset'] = "mitbih" if "mitbih" in f.name.lower() else "ptbdb"
    overall_results = pd.concat([overall_results, df], axis=0)

# %%
# do a bit more cleaning

def remove_substring(text: str, substring: str=None):
    """
    remove_substring - removes a substring from a string

    Parameters
    ----------
    text : str, the string to remove the substring from
    substring : str, the substring to remove from the string

    Returns
    -------
    str, the string with the substring removed
    """
    substring = substring if substring is not None else "_weights_preds"
    return text.replace(substring, "")

overall_results["model_filename"] = overall_results["model_filename"].apply(remove_substring)
overall_results.reset_index(drop=True, inplace=True)
overall_results['source'] = "initial-training"
overall_results.head()

# %%
# load files from ensemble-training
key_phrase = "fit_search"
# load CSV paths in initial-training directory that contain the key phrase
_ensemble_root = result_dirs["ensembling-models"]
_csv_paths = [f for f in _ensemble_root.iterdir() if f.is_file() and key_phrase in f.name.lower()]
print(f"{len(_csv_paths)} CSV files found in {_ensemble_root.name}")

# %%

col_names = get_unique_cols(_csv_paths)
print(f"{col_names} columns found in {_ensemble_root.name}")

not_in_map = [c for c in col_names if c not in metric_map.keys()]
if len(not_in_map) > 0:
    print(f"{len(not_in_map)} columns not in map: {not_in_map}")
# %%

def check_mapping(col_name: str, mapping: dict, exclude_list: list=None):
    """
    check_mapping - checks if a column name is in the mapping dictionary

    Parameters
    ----------
    col_name : str, the column name to check
    mapping : dict, the mapping dictionary

    Returns
    -------
    bool, whether the column name is in the mapping dictionary
    """
    if col_name in exclude_list:
        return col_name
    if col_name in mapping.keys() :
        return mapping[col_name]
    else:
        return False

for f in _csv_paths:
    df = pd.read_csv(f)
    _cols = list(df.columns)
    # replace the column name Model by 'model_filename'
    _cols = [c if c != "Model" else "model_filename" for c in _cols]
    df.columns = _cols
    new_cols = []
    for colname in list(df.columns):
        std_name = check_mapping(colname, metric_map, exclude_list=["model_filename"])
        if not std_name:
            del df[colname]
        else:
            new_cols.append(std_name)

    df.columns = new_cols
    _cols = list(df.columns)
    _cols.remove("model_filename")
    # all column entries except for the model file name now become one column titled performance_metric using pd.melt
    df = pd.melt(df, id_vars=["model_filename"], value_vars=_cols, var_name="performance_metric", value_name="metric_value")
    df['dataset'] = "mitbih" if "mitbih" in f.name.lower() else "ptbdb"
    overall_results = pd.concat([overall_results, df], axis=0)
