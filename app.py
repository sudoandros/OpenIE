import json
import logging
from datetime import datetime
from pathlib import Path

import typer
import chardet
import gensim.downloader

import openie.syntax
from openie.relations.text import TextReltuples

logging.basicConfig(
    handlers=[logging.FileHandler("logs/server.log", "a", "utf-8")],
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

cli = typer.Typer(add_completion=False)
with open("config.json", mode="r") as f:
    config = json.load(f)


def guess_encoding(content):
    encoding = chardet.detect(content)["encoding"]
    if encoding == "utf-8":
        return "utf-8"
    else:
        return "cp1251"


@cli.command()
def parse(filepath: Path):
    UDPIPE_MODEL = openie.syntax.UDPipeModel(config["UDPIPE_MODEL"])
    text = filepath.read_text()
    return openie.syntax.parse(text, UDPIPE_MODEL)


@cli.command()
def extract(
    filepaths: list[Path],
    is_conllu: bool = False,
    additional_relations: bool = True,
    entities_limit: int = 10000,
):
    UDPIPE_MODEL = openie.syntax.UDPipeModel(config["UDPIPE_MODEL"])
    W2V_MODEL = gensim.downloader.load("word2vec-ruscorpora-300")
    with open("stopwords.txt", mode="r", encoding="utf-8") as file:
        STOPWORDS = file.read().split()
    timestamp = datetime.now().strftime("d%Y-%m-%dt%H-%M-%S.%f")
    conllu = ""
    for path in filepaths:
        content = path.read_bytes()
        encoding = guess_encoding(content)
        text = content.decode(encoding)
        if is_conllu:
            new_conllu = text
        else:
            new_conllu = openie.syntax.parse(text, UDPIPE_MODEL, format_=path.suffix)
        conllu = f"{conllu}\n{new_conllu}"

    text_reltuples = TextReltuples(
        conllu, W2V_MODEL, STOPWORDS, additional_relations, entities_limit
    )
    graph_filename = f"{timestamp}.gexf"
    text_reltuples.graph.save(Path(config["GRAPH_DIR"], graph_filename))

    json_filename = f"{timestamp}.json"
    with Path(config["JSON_DIR"], json_filename).open(
        mode="w", encoding="utf-8"
    ) as json_file:
        json.dump(text_reltuples.dictionary, json_file, ensure_ascii=False, indent=4)

    conllu_filename = f"{timestamp}.conllu"
    with Path(config["CONLLU_DIR"], conllu_filename).open(
        mode="w", encoding="utf-8"
    ) as conllu_file:
        conllu_file.write(conllu)
    typer.echo(message=f"Relations graph: {graph_filename}")
    typer.echo(message=f"Relations JSON: {json_filename}")
    typer.echo(message=f"Syntax dependency tree: {conllu_filename}")


if __name__ == "__main__":
    cli()
