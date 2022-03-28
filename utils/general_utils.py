"""
    gen_utils.py - general utilities
"""
import os
from pathlib import Path
import re
import pandas as pd


def float_to_int(ser):
    try:
        int_ser = ser.astype(int)
        if (ser == int_ser).all():
            return int_ser
        else:
            return ser
    except ValueError:
        return ser


def multi_assign(df, transform_fn, condition):
    df_to_use = df.copy()

    return df_to_use.assign(
        **{col: transform_fn(df_to_use[col]) for col in condition(df_to_use)}
    )


def all_float_to_int(df):
    df_to_use = df.copy()
    transform_fn = float_to_int
    condition = lambda x: list(x.select_dtypes(include=["float"]).columns)

    return multi_assign(df_to_use, transform_fn, condition)


def downcast_all(df, target_type, inital_type=None):
    # Gotta specify floats, unsigned, or integer
    # If integer, gotta be 'integer', not 'int'
    # Unsigned should look for Ints
    if inital_type is None:
        inital_type = target_type

    df_to_use = df.copy()

    transform_fn = lambda x: pd.to_numeric(x, downcast=target_type)

    condition = lambda x: list(x.select_dtypes(include=[inital_type]).columns)

    return multi_assign(df_to_use, transform_fn, condition)


def clean_file_name(file_path):
    file_path = Path(file_path)
    # Remove all non-alphanumeric characters
    cln_base = re.sub(r"[^\w\s]", "", file_path.stem)
    # Replace all spaces with underscores
    cln_base = re.sub(r"\s", "_", cln_base)
    return cln_base + file_path.suffix
