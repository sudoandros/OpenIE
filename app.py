from datetime import datetime
from pathlib import Path

from flask import Flask, abort, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from wtforms import BooleanField, StringField, SubmitField
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


class TextForm(FlaskForm):
    text = StringField("Текст для обработки", validators=[DataRequired()])
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
    if "conllu" in request.form:
        conllu = request.form["conllu"]
    elif "text" in request.form:
        conllu = parse_text(request.form["text"], UDPIPE_MODEL)
    else:
        abort(400)

    if "add_rel" in request.form:
        additional_relations = request.form["add_rel"]
    else:
        additional_relations = False

    if "nodes_limit" in request.form:
        nodes_limit = request.form["nodes_limit"]
    else:
        nodes_limit = NODES_LIMIT

    graph, dict_out = get_text_relations(
        conllu, UDPIPE_MODEL, STOPWORDS, additional_relations, nodes_limit
    )
    timestamp = datetime.now().strftime("d%Y-%m-%dt%H-%M-%S.%f")
    graph_filename = "{}.gexf".format(timestamp)
    graph.save(GRAPH_DIR / graph_filename)
    return render_template(
        "relations.html", relations_dict=dict_out, graph_filename=graph_filename
    )


@app.route("/download/<directory>/<filename>", methods=["GET"])
def download(directory, filename):
    return send_from_directory(directory, filename)
