import json
from spacy.tokens import Doc
import spacy_conll
import logging
from datetime import datetime
from pathlib import Path

import chardet
import gensim.downloader
import spacy_udpipe
import typer

import openie.syntax.spacy_udpipe
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
    additional_relations: bool = True,
    entities_limit: int = 10000,
):
    nlp = spacy_udpipe.load_from_path("ru", config["UDPIPE_MODEL"])
    nlp.add_pipe("conll_formatter", last=True)
    w2v_model = gensim.downloader.load("word2vec-ruscorpora-300")
    with open("stopwords.txt", mode="r", encoding="utf-8") as file:
        STOPWORDS = file.read().split()
    timestamp = datetime.now().strftime("d%Y-%m-%dt%H-%M-%S.%f")

    docs: list[Doc] = []
    for path in filepaths:
        content = path.read_bytes()
        encoding = guess_encoding(content)
        text = content.decode(encoding)
        parsed_text = openie.syntax.spacy_udpipe.parse(text, nlp, format_=path.suffix)
        docs.append(parsed_text)

    parsed = Doc.from_docs(docs)
    text_reltuples = TextReltuples(
        parsed, w2v_model, STOPWORDS, additional_relations, entities_limit
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
        conllu_file.write("\n".join(d._.conll_str for d in docs))
    typer.echo(message=f"Relations graph: {graph_filename}")
    typer.echo(message=f"Relations JSON: {json_filename}")
    typer.echo(message=f"Syntax dependency tree: {conllu_filename}")


if __name__ == "__main__":
    cli()
