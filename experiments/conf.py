import json
import os

from config import RESULT_DIR
from privacy.postprocessing import resample_cartesian, resample_uniform

CONF_ANON_MODE = (
    "G", "S", "GS",
)

CONF_QI_MAP = (
    "AI", "A", "I"
)

RESAMPLING_STRATEGIES = dict(
    cartesian=resample_cartesian,
    uniform=resample_uniform,
)

CONF_VARS = dict(
    AS=dict(
        A=["age"],
        I=[],
        S="sex",
        O="income",
    ),
    ARS=dict(
        A=["age"],
        I=["race"],
        S="sex",
        O="income",
    ),
    WRS=dict(
        A=["workclass"],
        I=["race"],
        S="sex",
        O="income",
    ),
    ERS=dict(
        A=["education"],
        I=["race"],
        S="sex",
        O="income",
    ),
    HRS=dict(
        A=["hours-per-week"],
        I=["race"],
        S="sex",
        O="income",
    ),
    ORS=dict(
        A=["occupation"],
        I=["race"],
        S="sex",
        O="income",
    ),
    WAS=dict(
        A=["workclass"],
        I=["age"],
        S="sex",
        O="income",
    ),
    EAS=dict(
        A=["education"],
        I=["age"],
        S="sex",
        O="income",
    ),
    HAS=dict(
        A=["hours-per-week"],
        I=["age"],
        S="sex",
        O="income",
    ),
    OAS=dict(
        A=["occupation"],
        I=["age"],
        S="sex",
        O="income",
    ),
    AWRS=dict(
        A=["age", "workclass"],
        I=["race"],
        S="sex",
        O="income",
    ),
    WHRS=dict(
        A=["workclass", "hours-per-week"],
        I=["race"],
        S="sex",
        O="income",
    ),
    AWEHOS=dict(
        A=["age", "workclass", "education", "hours-per-week", "occupation"],
        I=[],
        S="sex",
        O="income",
    ),
    AWEHORS=dict(
        A=["age", "workclass", "education", "hours-per-week", "occupation"],
        I=["relationship"],
        S="sex",
        O="income",
    ),
    WEHOSA=dict(
        A=["workclass", "education", "hours-per-week", "occupation"],
        I=["age"],
        S="sex",
        O="income",
    ),
)


class Config:
    def __init__(self, mode, attrs=None, qi_map=None):
        self.mode = mode
        self.qi_map = qi_map
        self.var_conf = attrs

    def __str__(self):
        return f"{self.mode}-{self.qi_map}-{self.var_conf}-ADULT"

    @property
    def dir(self):
        return os.path.join(RESULT_DIR, str(self))

    @property
    def setup(self):
        return os.path.join(self.dir, "setup.json")

    def get_setup(self):
        with open(self.setup, "r") as f:
            return json.loads(f.read())

    @property
    def plot_dir(self):
        return os.path.join(self.dir, "plots")

    @property
    def base_table_dir(self):
        return os.path.join(self.dir, "tables")

    def table_dir(self, resample):
        return os.path.join(self.dir, f"tables_resample_{resample}")

    @property
    def table_dirs_resampling(self):
        return [self.table_dir(name) for name in sorted(RESAMPLING_STRATEGIES.keys())]

    def result_file(self, resample):
        return os.path.join(self.dir, f"results_resample_{resample}.csv")

    @property
    def result_files_resampling(self):
        return [self.result_file(name) for name in sorted(RESAMPLING_STRATEGIES.keys())]

    def plot_file(self, resample, column):
        return os.path.join(self.plot_dir, f"{column}_{resample}.pdf")

    def plot_files_metric(self, column):
        return [self.plot_file(name, column) for name in sorted(RESAMPLING_STRATEGIES.keys())]

    @property
    def exp_file(self):
        return os.path.join(self.dir, "experiments.csv")

    @property
    def use_suppression(self):
        return "S" in self.mode

    @property
    def use_generalization(self):
        return "G" in self.mode
