import json
from datetime import datetime
from pathlib import Path

import chardet
from flask import Flask, abort, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileRequired
from wtforms import BooleanField, MultipleFileField, StringField, SubmitField
from wtforms.validators import DataRequired

from relations import get_text_relations
from syntax import parse_text
from udpipe_model import UDPipeModel

app = Flask(__name__)
app.config["SECRET_KEY"] = "iamtakoiclever"
UDPIPE_MODEL = UDPipeModel(r"models\russian-syntagrus-ud-2.4-190531.udpipe")
with open("stopwords.txt", mode="r", encoding="utf-8") as file:
    STOPWORDS = list(file.read().split())
NODES_LIMIT = 3000
GRAPH_DIR = Path("graphs")
JSON_DIR = Path("jsons")


class TextForm(FlaskForm):
    text_files = MultipleFileField(
        "Текстовые файлы для обработки",
        validators=[
            FileRequired(),
            FileAllowed(["txt", "conllu", "hdr", "htm"], "Только текстовые файлы"),
        ],
    )
    is_conllu = BooleanField(
        "Содержимое является синтаксически разобранным текстом в формате CoNLL-U"
    )
    submit = SubmitField("Отправить")


@app.route("/", methods=["GET"])
@app.route("/index", methods=["GET"])
def index(title=None):
    form = TextForm()
    return render_template("index.html", title=title, form=form)


@app.route("/parse", methods=["POST"])
def parse():
    text = request.form["text"]
    return parse_text(text, UDPIPE_MODEL)


@app.route("/extract-relations", methods=["POST"])
def extract():
    timestamp = datetime.now().strftime("d%Y-%m-%dt%H-%M-%S.%f")
    conllu = ""
    for text_file in request.files.getlist("text_files"):
        file_content = text_file.read()
        encoding = chardet.detect(file_content)
        text = file_content.decode(encoding["encoding"])
        text_format = Path(text_file.filename).suffix[1:]
        if request.form.get("is_conllu") == "y":
            conllu = text
        else:
            file_conllu = parse_text(text, UDPIPE_MODEL, format_=text_format)
            conllu = "{}\n{}".format(conllu, file_conllu)

    # TODO not in use for now
    if "add_rel" in request.form:
        additional_relations = request.form["add_rel"]
    else:
        additional_relations = False

    # TODO not in use for now
    if "nodes_limit" in request.form:
        nodes_limit = request.form["nodes_limit"]
    else:
        nodes_limit = NODES_LIMIT

    graph, dict_out = get_text_relations(
        conllu, UDPIPE_MODEL, STOPWORDS, additional_relations, nodes_limit
    )
    graph_filename = "{}.gexf".format(timestamp)
    graph.save(GRAPH_DIR / graph_filename)
    json_filename = "{}.json".format(timestamp)
    with (JSON_DIR / json_filename).open(mode="w", encoding="utf-8") as json_file:
        json.dump(dict_out, json_file, ensure_ascii=False, indent=4)

    return render_template(
        "relations.html",
        relations_dict=dict_out,
        graph_filename=graph_filename,
        json_filename=json_filename,
    )


@app.route("/download/<directory>/<filename>", methods=["GET"])
def download(directory, filename):
    return send_from_directory(directory, filename, as_attachment=True)
