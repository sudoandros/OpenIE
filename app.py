from flask import Flask, request, abort, render_template

from udpipe_model import UDPipeModel
from syntax import parse_text
from relations import get_text_relations

app = Flask(__name__)
UDPIPE_MODEL = UDPipeModel(r"models\russian-syntagrus-ud-2.4-190531.udpipe")
with open("stopwords.txt", mode="r", encoding="utf-8") as file:
    STOPWORDS = list(file.read().split())
NODES_LIMIT = 3000


@app.route("/", methods=["GET"])
@app.route("/<title>", methods=["GET"])
def index(title=None):
    return render_template("index.html", title=title)


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
    return dict_out
