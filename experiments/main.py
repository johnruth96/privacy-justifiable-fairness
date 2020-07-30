import argparse
import json
import os
from datetime import datetime

import pandas as pd
from pandas import DataFrame

import dataset
from config import RESULT_DIR
from experiments.conf import Config, CONF_ANON_MODE, CONF_VARS, CONF_QI_MAP
from experiments.evaluate import evaluate_experiment
from experiments.resample import resample_tables
from privacy.bayardoext import BayardoExtendedAnonymizer
from privacy.ldiversity import post_process_k_anonymity
from privacy.models import get_k, get_l_distinct


def run_privacy(df, conf: Config):
    print(f"---- {conf.mode} - {conf.qi_map} - {conf.var_conf} ----")

    table_file = os.path.join(conf.base_table_dir, "K{}L{}.csv")
    if not os.path.exists(conf.dir):
        os.mkdir(conf.dir)
    if not os.path.exists(conf.base_table_dir):
        os.mkdir(conf.base_table_dir)

    # Init
    A, I, O = CONF_VARS[conf.var_conf]["A"], CONF_VARS[conf.var_conf]["I"], CONF_VARS[conf.var_conf]["O"]
    S = CONF_VARS[conf.var_conf]["S"]
    if conf.qi_map == "AI":
        QI = A + I
    elif conf.qi_map == "A":
        QI = A
    elif conf.qi_map == "I":
        QI = I
    else:
        raise NotImplementedError(f"attr_map {conf.qi_map} is not supported")
    df = df[A + I + [S, O]].copy()

    print("DEBUG: Evaluating initial K-Anonymity ...")
    k_current, n_groups = get_k(df, QI)
    print("DEBUG: Evaluating initial L-Diversity...")
    l_initial = get_l_distinct(df, S, QI)
    k_lst = [k_current]
    l_lst = [l_initial]
    n_lst = [n_groups]
    k_call = [0]
    cost_lst = [0]
    dur_lst = [0]
    print("DEBUG: Initializing anonymizer ...")
    if conf.qi_map == "AI":
        a = BayardoExtendedAnonymizer(
            df, A, I, use_suppression=conf.use_suppression, use_generalization=conf.use_generalization)
    elif conf.qi_map == "A":
        a = BayardoExtendedAnonymizer(
            df, A, [], use_suppression=conf.use_suppression, use_generalization=conf.use_generalization)
    elif conf.qi_map == "I":
        a = BayardoExtendedAnonymizer(
            df, I, [], use_suppression=conf.use_suppression, use_generalization=conf.use_generalization)
    else:
        raise NotImplementedError(f"attr_map {conf.qi_map} is not supported")

    # Write setup data
    print("DEBUG: Writing setup to {}".format(conf.setup))
    setup = dict(
        A=A,
        O=O,
        I=I,
        QI=QI,
        S=S,
        k_initial=k_current,
        l_initial=l_initial,
        n_groups=n_groups,
        k_max=a.k_max,
        n=len(df),
    )
    with open(conf.setup, "w") as f:
        f.write(json.dumps(setup))

    # Save initial dataset
    print("DEBUG: Saving initial dataset ...", flush=True)
    df.to_csv(table_file.format(k_current, l_initial))

    # Anonymize dataset
    while 0 < k_current < a.k_max:
        k = k_current + 1
        print(f"---- k = {k} ----")
        df_kano = a.run(k)
        if df_kano.empty:
            print("INFO: Stopping. DataFrame is empty")
            break
        else:
            l_df_kano = get_l_distinct(df_kano, S, QI)
            k_current, n_groups = get_k(df_kano, QI)

        k_call.append(k)
        n_lst.append(n_groups)
        k_lst.append(k_current)
        l_lst.append(l_df_kano)
        cost_lst.append(a.best_cost)
        dur_lst.append(a.duration)

        print(f"INFO: Saving {k_current}-anonymized table ...", flush=True, end="")
        df_kano.to_csv(table_file.format(k_current, l_df_kano))
        print(" done")
        if l_df_kano < 2 and df_kano[S].nunique() == 2:
            print(f"---- k = {k}, l = 2 ----")
            start = datetime.now()
            df_ldiv = post_process_k_anonymity(df_kano, 2, S, QI)
            k_ldiv, n_groups_ldiv = get_k(df_ldiv, QI)

            k_call.append(k)
            k_lst.append(k_ldiv)
            n_lst.append(n_groups_ldiv)
            l_lst.append(2)
            cost_lst.append(0)
            dur_lst.append(datetime.now() - start)
            print("INFO: Finished in {}".format(dur_lst[-1]))

            print(f"INFO: Saving {k_ldiv}-anonymized 2-diverse table ...", flush=True, end="")
            df_ldiv.to_csv(table_file.format(k_ldiv, 2))
            print(" done")

    # Save timing
    results = DataFrame(dict(cost=cost_lst, duration=dur_lst, k_call=k_call, n_groups=n_lst),
                        index=pd.MultiIndex.from_arrays((k_lst, l_lst), names=["k", "l"]))
    print("INFO: Saving timing ...", flush=True, end="")
    results.to_csv(conf.exp_file, index_label=["k", "l"], index=True)
    print(" done")


def main():
    parser = argparse.ArgumentParser()
    # Actions
    parser.add_argument("--create", "-c", help="Anonymize datasets", action="store_true")
    parser.add_argument("--resample", "-r", action="store_true")
    parser.add_argument("--evaluate", "-e", help="Evaluate results", action="store_true")
    # Positional
    parser.add_argument("mode", help="Mode", choices=CONF_ANON_MODE, nargs="?")
    parser.add_argument("qi", help="Attribute mapping (AI, A, I)", choices=CONF_QI_MAP, nargs="?")
    parser.add_argument("attrs", help="Var conf", choices=tuple(CONF_VARS.keys()), nargs="?")
    args = parser.parse_args()

    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    conf = Config(args.mode, attrs=args.attrs, qi_map=args.qi)

    if args.create:
        df = dataset.load_adult()
        run_privacy(df, conf)

    if args.resample:
        resample_tables(conf)

    if args.evaluate:
        evaluate_experiment(conf)


if __name__ == '__main__':
    main()
