import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import chardet
import gensim.downloader
from flask import (
    Flask,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask_wtf import FlaskForm
from wtforms import BooleanField, IntegerField, MultipleFileField, SubmitField

from relations import TextReltuples
from syntax import parse_text
from udpipe_model import UDPipeModel

logging.basicConfig(
    handlers=[logging.FileHandler("logs/server.log", "a", "utf-8")],
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

app = Flask(__name__)
app.config.from_json("instance/config.json")
UDPIPE_MODEL = UDPipeModel(app.config["UDPIPE_MODEL"])
W2V_MODEL = gensim.downloader.load("word2vec-ruscorpora-300")
with open("stopwords.txt", mode="r", encoding="utf-8") as file:
    STOPWORDS = list(file.read().split())


def guess_encoding(content):
    encoding = chardet.detect(content)["encoding"]
    if encoding == "utf-8":
        return "utf-8"
    else:
        return "cp1251"


class TextForm(FlaskForm):
    text_files = MultipleFileField("Текстовые файлы для обработки")
    entities_limit = IntegerField(
        "Максимальное количество извлеченных сущностей",
        default=app.config["ENTITIES_LIMIT"],
    )
    is_conllu = BooleanField(
        "Содержимое является синтаксически разобранным текстом в формате CoNLL-U"
    )
    submit = SubmitField("Отправить")


@app.route("/", methods=["GET", "POST"])
def index(title=None):
    form = TextForm()
    if form.validate_on_submit():
        return redirect(url_for("extract"), code=307)
    return render_template("index.html", title=title, form=form)


@app.route("/parse", methods=["POST"])
def parse():
    text = request.form["text"]
    return parse_text(text, UDPIPE_MODEL)


@app.route("/extract-relations", methods=["POST"])
def extract():
    tz_moscow = timezone(timedelta(hours=3))
    timestamp = datetime.now(tz=tz_moscow).strftime("d%Y-%m-%dt%H-%M-%S.%f")
    conllu = ""
    for text_file in request.files.getlist("text_files"):
        file_content = text_file.read()
        encoding = guess_encoding(file_content)
        text = file_content.decode(encoding)
        text_format = Path(text_file.filename).suffix[1:]
        if request.form.get("is_conllu") == "y":
            conllu = text
        else:
            file_conllu = parse_text(text, UDPIPE_MODEL, format_=text_format)
            conllu = "{}\n{}".format(conllu, file_conllu)

    additional_relations = True
    entities_limit = int(request.form["entities_limit"])

    text_reltuples = TextReltuples(
        conllu, UDPIPE_MODEL, W2V_MODEL, STOPWORDS, additional_relations, entities_limit
    )
    graph_filename = "{}.gexf".format(timestamp)
    text_reltuples.graph.save(Path(app.config["GRAPH_DIR"], graph_filename))

    json_filename = "{}.json".format(timestamp)
    with Path(app.config["JSON_DIR"], json_filename).open(
        mode="w", encoding="utf-8"
    ) as json_file:
        json.dump(text_reltuples.dictionary, json_file, ensure_ascii=False, indent=4)

    conllu_filename = "{}.conllu".format(timestamp)
    with Path(app.config["CONLLU_DIR"], conllu_filename).open(
        mode="w", encoding="utf-8"
    ) as conllu_file:
        conllu_file.write(conllu)

    return render_template(
        "relations.html",
        relations_dict=text_reltuples.dictionary,
        graph_filename=graph_filename,
        json_filename=json_filename,
        conllu_filename=conllu_filename,
    )


@app.route("/download/<directory>/<filename>", methods=["GET"])
def download(directory, filename):
    return send_from_directory(directory, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
