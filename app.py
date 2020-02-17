from flask import Flask, request

from udpipe_model import UDPipeModel
from syntax import parse_text
from relations import get_text_relations

app = Flask(__name__)
UDPIPE_MODEL = UDPipeModel(r"models\russian-syntagrus-ud-2.4-190531.udpipe")
with open("stopwords.txt", mode="r", encoding="utf-8") as file:
    STOPWORDS = list(file.read().split())
NODES_LIMIT = 3000


@app.route("/")
def root():
    return "Hello"


@app.route("/parse", methods=["POST"])
def parse():
    text = request.form["text"]
    return parse_text(text, UDPIPE_MODEL)


@app.route("/extract-relations", methods=["POST"])
def extract():
    conllu = request.form["conllu"]
    additional_relations = ("add_rel" in request.form) and request.form["add_rel"]
    nodes_limit = ("nodes_limit" in request.form) and NODES_LIMIT
    graph, dict_out = get_text_relations(
        conllu, UDPIPE_MODEL, STOPWORDS, additional_relations, nodes_limit
    )
    return dict_out
