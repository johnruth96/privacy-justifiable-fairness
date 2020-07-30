import pandas
import pandas as pd

from config import ADULT_DATA

ADULT_HEADER = (
    'age',  # Continuous
    'workclass',
    'fnlwgt',  # Continuous
    'education',
    'education-num',  # Continuous
    'maritalStatus',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',  # Continuous
    'capital-loss',  # Continuous
    'hours-per-week',  # Continuous
    'nativeCountry',
    'income',  # Outcome
)

ADULT_CONTINUOUS = (
    'age',  # Continuous
    'fnlwgt',  # Continuous
    'education-num',  # Continuous
    'capital-gain',  # Continuous
    'capital-loss',  # Continuous
    'hours-per-week',  # Continuous
)


def load_adult(filename=ADULT_DATA, raw=False):
    df = pandas.read_csv(filename, names=ADULT_HEADER, header=0, index_col=False, delimiter=' *, *', engine='python')
    if raw:
        return df
    else:
        df = convert_to_categorical(df, ADULT_CONTINUOUS, 10)
        return df


def convert_to_categorical(df, cont_attributes, num_quantiles):
    """
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html

    :param df:
    :param cont_attributes:
    :param num_quantiles:
    :return:
    """
    df_new = df.copy()
    for attr in cont_attributes:
        series = pd.qcut(df[attr], q=num_quantiles, duplicates="drop", precision=1)
        df_new[attr] = series.apply(str)
    return df_new
