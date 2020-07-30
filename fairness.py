import sys

import numpy as np
import pandas as pd

from utils import get_domain


def contingency_matrices_iterator(df: pd.DataFrame, attrs_x, attrs_y, attrs_z):
    """
    Create contingency matrices for (X;Y|Z)

    :param df: Dataset
    :param list attrs_x: X
    :param list attrs_y: Y
    :param list attrs_z: Z
    :return:
    """
    if isinstance(attrs_x, str):
        attrs_x = [attrs_x]
    if isinstance(attrs_y, str):
        attrs_y = [attrs_y]
    if isinstance(attrs_z, str):
        attrs_z = [attrs_z]

    # Handle Z = \empty (in case of K-fairness)
    if attrs_z:
        groups = df.groupby(attrs_z)
    else:
        groups = [(None, df)]

    size = len(groups)
    for idx, (_, df_grouped) in enumerate(groups):
        sys.stdout.write("\rContingency matrices: {:.2%} ...".format((idx + 1.0) / size))
        df_cross = pd.crosstab(
            [df_grouped[attr] for attr in attrs_x],
            [df_grouped[attr] for attr in attrs_y]
        )
        # cont_matrix = df_cross.to_numpy()
        yield df_cross
    sys.stdout.write(" done\n")


def get_ratio_of_discr(df: pd.DataFrame, admissibles, outcome, sensitive):
    """
    Compute adjusted Ratio of Observational Discrimination
    """
    dom_sensitive = sorted(get_domain(df, [sensitive]))
    dom_outcome = sorted(get_domain(df, [outcome]))

    if len(dom_sensitive) != 2 or len(dom_outcome) != 2:
        return np.array([1.0])

    s0 = dom_sensitive[0]
    s1 = dom_sensitive[1]
    o0 = dom_outcome[0]
    o1 = dom_outcome[1]

    cont_matrix_iter = contingency_matrices_iterator(df, outcome, sensitive, admissibles)

    rods = []
    for cm in cont_matrix_iter:
        if cm.shape == (2, 2):
            cb = cm[s0][o1] * cm[s1][o0]
            ad = cm[s0][o0] * cm[s1][o1]
            if ad != 0:
                rods.append(cb / ad)
        else:
            rods.append(1.0)

    rods = np.array(rods)
    return rods


def measure_fairness(df: pd.DataFrame, adm, inadm, outcome, sensitive):
    rods = get_ratio_of_discr(df, adm, outcome, sensitive)
    cm_ranks = [np.linalg.matrix_rank(cm.to_numpy()) for cm in contingency_matrices_iterator(df, outcome, inadm, adm)]
    cm_ranks = np.array(cm_ranks)
    return dict(
        n_cont=len(cm_ranks),
        rod=rods.mean(),
        rod_abs=np.abs(1 - rods.mean()),
        size=len(df),
        ratio_fair=np.count_nonzero(cm_ranks == 1.0) / len(cm_ranks),
        rank_mean=cm_ranks.mean(),
        rank_median=np.median(cm_ranks),
    )
