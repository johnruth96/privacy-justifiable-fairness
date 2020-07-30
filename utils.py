import re
from typing import List

INTERVAL_PATTERN = re.compile(r"[\[(][\d.]+, [\d.]+[)\]]")

GEN_DELIMITER = "; "


def format_generalization(values):
    return "{{{}}}".format(GEN_DELIMITER.join(str(x) for x in values))


def is_gen(value: str) -> bool:
    return value.startswith("{") and value.endswith("}")


def gen2set(generalization: str) -> set:
    return set(generalization[1:-1].split(GEN_DELIMITER))


def clean_adult(in_file, out_file):
    counter = 0
    with open(in_file) as fin:
        with open(out_file, "w") as fout:
            for l in fin:
                if "?" not in l:
                    fout.write(l)
                else:
                    counter += 1
    print("Removed {} lines".format(counter))


def get_domain(df, attrs: List):
    if len(attrs) > 1:
        return set(map(tuple, df[attrs].to_numpy()))
    return set(df[attrs[0]].to_numpy())


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b
