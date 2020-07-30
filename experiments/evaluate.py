import os

import pandas as pd

from experiments.conf import Config
from fairness import measure_fairness
from privacy.models import get_l_distinct, get_k


def evaluate_experiment(conf: Config):
    # Load setup
    setup = conf.get_setup()
    A = setup["A"]
    I = setup["I"]
    O = setup["O"]
    S = setup["S"]
    QI = setup["QI"]

    # Evaluation
    for table_dir, result_file in zip(conf.table_dirs_resampling, conf.result_files_resampling):
        if not os.path.exists(table_dir):
            continue

        print(f"INFO: Evaluating {table_dir}")

        df_exp = pd.read_csv(conf.exp_file, header=0, index_col=[0, 1])
        indexes = []
        col_names = []
        rows = []

        # Read tables
        for k, l in df_exp.index:
            print("Evaluating ({}, {}) ...".format(k, l))
            table_file = os.path.join(table_dir, "K{}L{}.csv".format(k, l))
            df = pd.read_csv(table_file, header=0, index_col=0)

            k_df, n_df = get_k(df, QI)
            l_df = get_l_distinct(df, S, QI)
            idx = (k_df, l_df)

            if idx in indexes:
                print(f"WARNING: index ({k_df}, {l_df}) already in {table_file}")

            measurements = measure_fairness(df, A, I, O, S)
            measurements.update(
                n_groups=n_df,
                idx_original=(k, l),
            )

            if not col_names:
                col_names = sorted(measurements.keys())

            indexes.append(idx)
            rows.append([measurements[measure] for measure in col_names])

        results = pd.DataFrame(rows, columns=col_names,
                               index=pd.MultiIndex.from_tuples(indexes, names=["k", "l"]))
        print(f"Writing results to {result_file} ...", flush=True, end="")
        results.to_csv(result_file, index_label=["k", "l"], index=True)
        print(" done")
