from datetime import datetime

import pandas as pd

from privacy.base import suppress_only
from privacy.bayardo import BayardoAnonymizer


class BayardoExtendedAnonymizer:
    """
    Optimal k-Anonymity [Bayardo et al.] to generate fairness
    """

    def __init__(self, df, quasi_identifier, grouping_keys, use_suppression, use_generalization):
        """

        :param df:
        :param quasi_identifier: QI for the anonymization (Bayardo)
        :param grouping_keys: Grouping for each anonymizer (can be empty)
        :param use_suppression:
        :param use_generalization:
        """
        # Small integrity checks
        if set(quasi_identifier).intersection(grouping_keys):
            raise ValueError("QI and grouping key must be disjoint.")
        # Save parameters
        self.df = df
        self.quasi_identifier = sorted(quasi_identifier)
        self.grouping_keys = sorted(grouping_keys)
        self.use_suppression = use_suppression
        self.use_generalization = use_generalization
        self.suppression_qi = self.quasi_identifier + self.grouping_keys
        # Prepare
        self.size = len(df.index)
        self.duration = None
        if self.grouping_keys:
            groups = self.df.groupby(self.grouping_keys).groups
            self._df_groups = [self.df.iloc[g_idx] for _, g_idx in groups.items()]
        else:
            self._df_groups = [self.df]
        # State
        self._anonymizers = [
            BayardoAnonymizer(df_slice, self.quasi_identifier, use_suppression=use_suppression) for df_slice in
            self._df_groups]
        # Print INFO
        print("INFO: Initialized {} groups".format(len(self._df_groups)))
        print("INFO: k_max = {}".format(self.k_max))
        if use_generalization:
            print("INFO: Using generalization")
        if use_suppression:
            print("INFO: Using suppression")
        for idx, a in enumerate(self._anonymizers):
            print("DEBUG: Anonymizer {}: {} tuples, {} domain values".format(idx + 1, a.size, len(a.dom_values)))

    @property
    def _suppression_only(self):
        return self.use_suppression and not self.use_generalization

    @property
    def k_max(self):
        if self.grouping_keys and self.use_generalization:
            return int(self.df.groupby(self.grouping_keys).count().min().min())
        elif self._suppression_only:
            return int(self.df.groupby(self.suppression_qi).count().max().min())
        else:
            return int(self.size)

    @property
    def best_cost(self):
        if self._suppression_only:
            return -1
        else:
            return sum(a.best_cost for a in self._anonymizers)

    def generate_output(self):
        frames = [a.anonymized_df for a in self._anonymizers]
        return pd.concat(frames)

    def run(self, k):
        # Integrity checks
        if not 1 <= k <= self.k_max:
            raise ValueError("k must be from [1, {}]".format(self.k_max))

        start = datetime.now()
        if self._suppression_only:
            print("INFO: Anonymizing ... ", end="", flush=True)
            df = suppress_only(self.df, k, self.suppression_qi)
            print("done", flush=True)
            self.duration = datetime.now() - start
        else:
            print("INFO: Anonymizing ... {:.2%}".format(0), end="", flush=True)
            for idx, anonymizer in enumerate(self._anonymizers):
                anonymizer.run(k)
                print("\rINFO: Anonymizing ... {:.2%}".format((idx + 1.0) / len(self._anonymizers)), end="", flush=True)
            print("")
            self.duration = datetime.now() - start
            df = self.generate_output()

        print("INFO: Finished in {}".format(self.duration))

        return df
