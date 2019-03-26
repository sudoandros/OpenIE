import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from udpipe_model import UDPipeModel

UDPIPE_MODEL_PATH = "data/udpipe_models/russian-syntagrus-ud-2.3-181115.udpipe"


def text_from_sts(filepath):
    res = []
    with filepath.open("r", encoding="cp1251") as f:
        for line in f:
            res.append(" ".join(line.split()[6:]))
    return "\n".join(res)
    # return res


def text_from_hdr(filepath):
    res = None
    with filepath.open("r", encoding="cp1251") as f:
        for line in f:
            if line.split("=", 1)[0] == "TEXT_THEMAN_ANNO":
                res = line.split("=", 1)[1]
    return res

def parse(dir_path, udpipe_model):
    for sts_path in tqdm(dir_path.iterdir()):
        if sts_path.suffix != ".sts":
            continue
        text = text_from_sts(sts_path)

        sentences = udpipe_model.tokenize(text)
        for s in sentences:
            udpipe_model.tag(s)
            udpipe_model.parse(s)

        conllu = udpipe_model.write(sentences, "conllu")
        towrite = dir_path / (sts_path.stem + "_udpipe.conllu")
        with towrite.open("w", encoding="utf8") as f:
            f.write(conllu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse every .sts file in specified directory using UDPipe"
        " and save results in the same directory"
    )
    parser.add_argument("directory")
    args = parser.parse_args()

    dir_path = Path(args.directory)
    udpipe_model = UDPipeModel(UDPIPE_MODEL_PATH)

    parse(dir_path, udpipe_model)