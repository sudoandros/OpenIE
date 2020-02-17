from flask import Flask, request

from udpipe_model import UDPipeModel
from syntax import parse_text

app = Flask(__name__)
UDPIPE_MODEL = UDPipeModel(r"models\russian-syntagrus-ud-2.4-190531.udpipe")


@app.route("/")
def root():
    return "Hello"


@app.route("/parse", methods=["POST"])
def parse():
    text = request.form["text"]
    return parse_text(text, UDPIPE_MODEL)
