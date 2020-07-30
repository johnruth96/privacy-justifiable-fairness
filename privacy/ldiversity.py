import pandas as pd

from utils import GEN_DELIMITER


def div(ci):
    return ci.nunique()  # np.unique(ci).size


def concat_label(l1: any, l2: any) -> tuple:
    def get_values(g):
        if g.startswith("{"):
            return set(g[1:-1].split(GEN_DELIMITER))
        else:
            return {g}

    new = []

    if not isinstance(l1, tuple):
        l1 = (l1,)
    if not isinstance(l2, tuple):
        l2 = (l2,)

    for g1, g2 in zip(l1, l2):
        values = get_values(g1)
        values = values.union(get_values(g2))
        new.append("{{{}}}".format(GEN_DELIMITER.join(sorted(values))))
    return tuple(new)


def cost_i(c1, c2):
    """
    Information loss of unifying clusters. Implements the discernibility metric.
    :param c1:
    :param c2:
    :return:
    """
    return len(c1 + c2) ** 2 - len(c1) ** 2 - len(c2) ** 2


def cost_d(l, c1, c2):
    """
    Diversity cost
    :param c1:
    :param c2:
    :return:
    """
    return max(0, l - div(c1 + c2))


def cost(l, c1, c2, w=0.5):
    return w * cost_i(c1, c2) + (1 - w) * cost_d(l, c1, c2)


def get_min_div_group(groups):
    min_div = float("inf")
    min_label = None
    for label, group in groups.items():
        c_div = div(group)
        if c_div < min_div:
            min_div = c_div
            min_label = label
    return min_label, min_div


def post_process_k_anonymity(df, l, sensitive, quasi_identifiers):
    df = df.copy()

    if div(df[sensitive]) < l:
        raise ValueError("Maximal diversity is {}, but l = {}".format(div(df[sensitive]), l))

    quasi_identifiers = sorted(quasi_identifiers)
    groups = {k: v for k, v in df.groupby(quasi_identifiers)[sensitive]}

    # Merge groups
    min_label, min_div = get_min_div_group(groups)
    while min_div < l and len(groups) > 1:
        min_cost = float("inf")
        partner_label = None
        for label, group in groups.items():
            if label != min_label:
                cur_cost = cost(l, groups[min_label], group)
                if cur_cost < min_cost:
                    partner_label = label
                    min_cost = cur_cost

        new_label = concat_label(min_label, partner_label)
        new_group = pd.concat((groups[min_label], groups[partner_label]))
        del groups[min_label]
        del groups[partner_label]
        groups[new_label] = new_group

        min_label, min_div = get_min_div_group(groups)

    # Refactor dataframe
    for column in quasi_identifiers:
        if df[column].dtype.name == "category":
            df[column] = df[column].astype(str)
    for label, rows in groups.items():
        df.loc[rows.index, quasi_identifiers] = label

    return df
