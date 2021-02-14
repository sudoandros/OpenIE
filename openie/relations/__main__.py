import argparse
import json
import logging
from pathlib import Path
from typing import List

import gensim.downloader
from tqdm import tqdm

from .text import TextReltuples


def build_dir_graph(
    conllu_dir: Path,
    save_dir: Path,
    stopwords: List[str],
    additional_relations: bool,
    entities_limit: int,
    w2v_model,
):
    conllu = ""
    for path in tqdm(conllu_dir.glob("*.conllu")):
        text_conllu = path.read_text(encoding="utf8")
        conllu = "{}\n{}".format(conllu, text_conllu)

    text_reltuples = TextReltuples(
        conllu, w2v_model, stopwords, additional_relations, entities_limit
    )

    json_path = save_dir / "relations_{}.json".format(conllu_dir.name)
    with json_path.open("w", encoding="utf8") as json_file:
        json.dump(text_reltuples.dictionary, json_file, ensure_ascii=False, indent=4)

    graph_path = save_dir / "graph_{}.gexf".format(conllu_dir.name)
    text_reltuples.graph.save(graph_path)
    print(text_reltuples.graph.nodes_number, text_reltuples.graph.edges_number)


if __name__ == "__main__":
    logging.basicConfig(
        handlers=[logging.FileHandler("logs/server.log", "a", "utf-8")],
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "conllu_dir",
        help="Path to the directory containing parsed text in conllu format",
    )
    parser.add_argument("save_dir", help="Path to the directory to save relations to")
    parser.add_argument(
        "--add", help="Include additional relations", action="store_true"
    )
    parser.add_argument(
        "--entities-limit",
        help="Filter extracted relations to only contain this many entities",
        type=int,
    )
    args = parser.parse_args()
    conllu_dir = Path(args.conllu_dir)
    save_dir = Path(args.save_dir)
    entities_limit = args.entities_limit or float("inf")
    with open("stopwords.txt", mode="r", encoding="utf-8") as file:
        stopwords = list(file.read().split())
    w2v_model = gensim.downloader.load("word2vec-ruscorpora-300")

    build_dir_graph(
        conllu_dir, save_dir, stopwords, args.add, entities_limit, w2v_model,
    )
