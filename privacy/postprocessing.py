import random

import pandas as pd

from utils import gen2set, is_gen


def resample_cartesian(df: pd.DataFrame, qi):
    """
    :param df:
    :param qi: Quasi-identifiers which were generalized
    :return:
    """
    df_new = df.applymap(lambda x: list(gen2set(x)) if is_gen(x) else x)
    for col in qi:
        df_new = df_new.explode(col)
    df_new.reset_index(inplace=True)

    return df_new


def resample_uniform(df: pd.DataFrame, qi):
    """
    :param df:
    :return:
    """
    df_new = df.applymap(lambda x: random.choice(list(gen2set(x))) if is_gen(x) else x)
    df_new.reset_index(inplace=True)

    return df_new
