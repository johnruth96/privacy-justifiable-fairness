import numpy as np
from pandas import DataFrame


def get_k(df: DataFrame, quasi_identifiers: list):
    rows = df[quasi_identifiers].values
    hashes = np.array([hash(tuple(x)) for x in rows])
    _, counts = np.unique(hashes, return_counts=True)
    return int(counts.min()), len(counts)


def get_l_distinct(df: DataFrame, sensitive, quasi_identifiers: list):
    df_grouped = df.groupby(quasi_identifiers).nunique()
    min_l = int(df_grouped[sensitive].min())
    return min_l if min_l > 0 else 1
