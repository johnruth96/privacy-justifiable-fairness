from datetime import datetime

import pandas as pd
from more_itertools import flatten

from utils import format_generalization


def gen_dict(anonymization, dom_values_enum):
    def f(x):
        for idx in range(len(anonymization) - 1):
            if anonymization[idx] <= x < anonymization[idx + 1]:
                return anonymization[idx]
        return anonymization[-1]

    return dict((x, f(x)) for x in dom_values_enum)


class BaseAnonymizer:
    """
    Important note: The here-user "set" means an ordered set why it is implemented as list.

    According to the Bayardo-paper, we enumerate all values from each domain.
    We convert the original dataset into a representation with every original value is replaced by its unique number.
    """

    def __init__(self, data_frame, quasi_identifiers, use_suppression=False):
        # Parameters
        self.original_column_order = data_frame.columns
        self.dataframe = data_frame[sorted(data_frame.columns)]
        self._dataframe_qi = None
        self.quasi_identifiers = sorted(quasi_identifiers)
        self.use_suppression = use_suppression
        # Stateful variables (change each run)
        self.best_head = None
        self.best_cost = None
        self._eq_class_cache = dict()
        self._k = None
        self._df_anonymized = None
        self.duration = None
        # Generate datasets
        self.dataset_original = []
        self.dataset_enum = []
        self.attr_count = 0
        self.size = 0
        self.domains = []
        self.dom_values = []
        self.dom_values_enum = []
        self.most_general_anonymization = []
        self.sigma_all = []
        self._init_dataset()

    @property
    def anonymized_df(self):
        return self._df_anonymized

    @property
    def k_max(self):
        return self.size

    def _reset_state(self, k):
        self.best_cost = float("inf")
        self.best_head = None
        self.duration = None
        self._k = k
        self._df_anonymized = None

    def _get_domain(self, attr_idx):
        return sorted({x[attr_idx] for x in self.dataset_original})

    def _init_dataset(self):
        df_qi = self.dataframe[self.quasi_identifiers]
        self.dataset_original = [tuple(x) for x in df_qi.to_numpy()]
        self.size = len(self.dataset_original)
        self.attr_count = len(self.dataset_original[0])
        self.domains = [self._get_domain(i) for i in range(self.attr_count)]
        # Generate numerical domain values
        domain_offset = [0]
        for idx in range(1, self.attr_count):
            domain_offset.append(max(domain_offset) + len(self.domains[idx - 1]))
        self.dom_values = list(flatten(self.domains))
        self.dom_values_enum = list(range(1, len(self.dom_values) + 1))
        # Generate most general anonymization
        most_general_anonymization = set(x + 1 for x in domain_offset)
        self.most_general_anonymization = sorted(most_general_anonymization)
        # Generate sigma
        self.sigma_all = sorted(set(self.dom_values_enum).difference(set(self.most_general_anonymization)))
        # Generate enumerated dataset
        self._dataframe_qi = df_qi.replace(self.dom_values, self.dom_values_enum)
        self.dataset_enum = list(map(tuple, df_qi.replace(self.dom_values, self.dom_values_enum).to_numpy()))

    def generate_anonymized_dataset(self, anonymization):
        """
        Generate enum-anonymized dataset. Function uses enumeration of values

        :param anonymization:
        :return:
        """
        d = gen_dict(anonymization, self.dom_values_enum)
        ds_anonymized = [
            tuple(d[value] for value in t) for t in self.dataset_enum
        ]
        return ds_anonymized

    def expand_head_set(self, head_set):
        return sorted(head_set + self.most_general_anonymization)

    def generate_eq_classes(self, head_set):
        """
        Calculates equivalence classes and their members. Returns list of pairs, pair is equivalence class and members

        :param head_set: head set
        :return:
        """
        # TODO: Optimize, maybe with the Union-Find-Algorithm?
        anonymization = tuple(self.expand_head_set(head_set))
        if anonymization in self._eq_class_cache:
            return self._eq_class_cache[anonymization]

        # Generate anonymized dataset
        ds_anonymized = self.generate_anonymized_dataset(anonymization)

        # Create equivalence classes
        eq_classes = [
            (eqc, ds_anonymized.count(eqc)) for eqc in
            set(ds_anonymized)
        ]

        # Cache and return
        self._eq_class_cache[anonymization] = eq_classes
        return eq_classes

    def run(self, k):
        # Integrity checks
        if not 1 <= k <= self.k_max:
            raise ValueError("k must be from [1, {}]".format(self.k_max))
        self._reset_state(k)

        start = datetime.now()
        self.anonymize()
        self.duration = datetime.now() - start
        self._df_anonymized = self.generate_output()
        return self._df_anonymized

    def anonymize(self):
        raise NotImplementedError

    def generate_output(self):
        """
        Map integers to value from original domain
        :return:
        """
        anonymization = self.expand_head_set(self.best_head)
        ds_ano_enum = self.generate_anonymized_dataset(anonymization)
        df_ano = self.dataframe[self.quasi_identifiers].copy()
        # Convert categorical columns to string
        for column in df_ano.columns.values:
            if df_ano[column].dtype.name == "category":
                df_ano[column] = df_ano[column].astype(str)
        # Replace enumerated values
        for t_idx, t in enumerate(ds_ano_enum):
            for v_idx, v in enumerate(t):
                an_idx = anonymization.index(v)
                end_value = anonymization[an_idx + 1] if an_idx < len(anonymization) - 1 else self.dom_values_enum[
                                                                                                  -1] + 1
                values = self.dom_values[v - 1:end_value - 1]
                df_ano.iloc[t_idx, v_idx] = format_generalization(values)

        missing_columns = set(self.original_column_order).difference(set(df_ano.columns))
        df = pd.concat([df_ano, self.dataframe[missing_columns].copy()], axis=1)
        df = df[self.original_column_order].sort_values(list(self.original_column_order), axis=0)
        # Remove tuples
        if self.use_suppression:
            df = suppress_only(df, self._k, self.quasi_identifiers)

        return df

    def compute_cost(self, head_set):
        """
        Implements the discernibility metric with minor changes:
        - Tuple suppression costs infinity if disallowed
        - Each merge costs 1

        :param head_set:
        :return:
        """
        eq_classes = self.generate_eq_classes(head_set)
        cost = len(self.sigma_all) - len(head_set)
        for eqc, eqc_size in eq_classes:
            if eqc_size >= self._k:
                cost += eqc_size * eqc_size
            else:
                if self.use_suppression:
                    cost += eqc_size * self.size
                else:
                    cost = float("inf")
                    break
        return cost

    def compute_lower_bound(self, head_set, all_set) -> float:
        eqc_head = self.generate_eq_classes(head_set)
        if next((True for _, size in eqc_head if size < self._k), False):
            return float("inf")
        eqc_all = self.generate_eq_classes(all_set)
        generalization_cost = len(self.sigma_all) - len(all_set)  # New, compared to Bayardo et al.
        min_cost = sum(size * max(size, self._k) for _, size in eqc_all) + generalization_cost
        return min_cost


def suppress_only(df, k, QI):
    s = list(set(df.columns.values).difference(set(QI)))[0]
    group_obj = df[QI + [s]].groupby(QI)
    group_counts = group_obj.count()
    groups_to_delete = group_counts[group_counts[s] < k].index.values

    count = 0
    df_new = df.copy()
    for group in groups_to_delete:
        df_new.drop(group_obj.groups[group], inplace=True)
        count += len(group_obj.groups[group])

    return df_new
