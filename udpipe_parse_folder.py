import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

from udpipe_model import UDPipeModel


def get_text(filepath, format_=None):
    if format_ == "sts":
        return get_text_from_sts(filepath)
    elif format_ == "hdr":
        return get_text_from_hdr(filepath)
    elif format_ == "htm":
        return get_text_from_htm(filepath)
    else:
        raise ValueError("This format is not supported")


def get_text_from_sts(filepath):
    res = []
    with filepath.open("r", encoding="cp1251") as f:
        for line in f:
            res.append(" ".join(line.split()[6:]))
    return "\n".join(res)


def get_text_from_hdr(filepath):
    res = None
    with filepath.open("r", encoding="cp1251") as f:
        for line in f:
            if line.split("=", 1)[0] == "TEXT_THEMAN_ANNO":
                res = line.split("=", 1)[1]
    return res


def get_text_from_htm(filepath):
    with open(filepath, mode="r", encoding="cp1251") as file:
        file_content = file.read()
        title_match = re.search(
            "<CIR_HDR><H1><CENTER>.*</CENTER></H1></CIR_HDR>",
            file_content,
            flags=re.DOTALL,
        )
        text_match = re.search("</CIR_HDR>.*</BODY>", file_content, flags=re.DOTALL)
    res = ""
    if title_match:
        res += title_match.group(0) + ". "
    res += " ".join(text_match.group(0).split())
    return res


def parse(texts_dir, conllu_dir, udpipe_model, format_="sts"):
    tag_regex = re.compile("<[^>]+>")
    for text_path in tqdm(texts_dir.iterdir()):
        if text_path.suffix == ".{}".format(format_):
            text = get_text(text_path, format_=format_)
        else:
            continue
        text = tag_regex.sub("", text)

        sentences = udpipe_model.tokenize(text)
        for s in sentences:
            udpipe_model.tag(s)
            udpipe_model.parse(s)

        conllu = udpipe_model.write(sentences, "conllu")
        conllu_path = conllu_dir / (text_path.stem + "_udpipe.conllu")
        with conllu_path.open("w", encoding="utf8") as f:
            f.write(conllu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse every text file in specified directory using UDPipe"
    )
    parser.add_argument("model_path", help="Path to the UDPipe model")
    parser.add_argument("texts_dir", help="Directory with text files")
    parser.add_argument(
        "conllu_dir", help="Directory where results of parsing should be saved to"
    )
    parser.add_argument(
        "--format",
        help="Format of the texts to be processed",
        choices=["sts", "hdr", "htm"],
    )
    args = parser.parse_args()

    texts_dir = Path(args.texts_dir)
    conllu_dir = Path(args.conllu_dir)
    udpipe_model = UDPipeModel(args.model_path)

    parse(texts_dir, conllu_dir, udpipe_model, format_=args.format)
