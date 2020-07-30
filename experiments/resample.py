import os

import pandas as pd

from experiments.conf import Config, RESAMPLING_STRATEGIES


def resample_tables(conf: Config):
    # Load setup
    setup = conf.get_setup()
    QI = setup["QI"]

    # Resample tables
    results = pd.read_csv(conf.exp_file, header=0, index_col=[0, 1])

    for name, resample_func in RESAMPLING_STRATEGIES.items():
        print(f"INFO: Resampling {conf}-{name} ...")
        if not os.path.exists(conf.table_dir(name)):
            os.mkdir(conf.table_dir(name))

        for idx, (k, l) in enumerate(results.index):
            print("Resampling k={}, l={} ({}/{}) ... ".format(k, l, idx + 1, len(results.index.values)), end="")
            table_file = os.path.join(conf.base_table_dir, "K{}L{}.csv".format(k, l))
            df = pd.read_csv(table_file, header=0, index_col=0)
            df_resampled = resample_func(df, QI)
            table_res_out = os.path.join(conf.table_dir(name), "K{}L{}.csv".format(k, l))
            df_resampled.to_csv(table_res_out, index_label="k", index=True)
            print(f"{len(df_resampled)} lines written")
