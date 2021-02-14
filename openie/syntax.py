import html
import re

import ufal.udpipe

# ufal.udpipe.Model etc. are SWIG-magic and cannot be detected by pylint
# pylint: disable=no-member


def parse(text, udpipe_model, format_=None):
    text = clean_text(text, format_=format_)

    sentences = udpipe_model.tokenize(text)
    for s in sentences:
        udpipe_model.tag(s)
        udpipe_model.parse(s)

    conllu = udpipe_model.write(sentences, "conllu")
    return conllu


def clean_text(text, format_=None):
    result = ""
    if format_ == "htm":
        result = clean_htm(text)
    elif format_ == "hdr":
        result = clean_hdr(text)
    elif format_ == "sts":
        result = clean_sts(text)
    else:
        result = text

    result = re.sub(r"<[^>]+>", "", result)
    result = re.sub(r"\\n+", "\n", result)
    result = html.unescape(result)

    def newline_repl(matchobj):
        return "{}. ".format(matchobj.group(1))

    result = re.sub(r"([^.!?])(\s*\n+)", newline_repl, result)

    return result


def clean_sts(text):
    result = ""
    matches = re.findall(r"(\d+\s+){6}(.+)", text)
    for m in matches:
        result = "{}\n{}".format(result, m[1])
    return result


def clean_hdr(text):
    result = ""
    matches = re.findall(r"TEXT_THEMAN_ANNO=(.+)", text)
    for m in matches:
        result = "{}\n{}".format(result, m)
    return result


def clean_htm(text):
    return re.sub(r"^\w+\s*=.*", "", text, flags=re.MULTILINE)


def read_parsed(text, in_format):
    """Load text in the given format (conllu|horizontal|vertical) and return
    list of ufal.udpipe.Sentence-s."""
    if isinstance(in_format, str):
        input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
    else:
        input_format = in_format
    if not input_format:
        raise Exception("Cannot create input format '%s'" % in_format)

    input_format.setText(text)
    error = ufal.udpipe.ProcessingError()
    sentences = []

    sentence = ufal.udpipe.Sentence()
    while input_format.nextSentence(sentence, error):
        sentences.append(sentence)
        sentence = ufal.udpipe.Sentence()
    if error.occurred():
        raise Exception(error.message)
    return sentences


def write_parsed(sentences, out_format):
    """Write given ufal.udpipe.Sentence-s in the required format
    (conllu|horizontal|vertical)."""
    output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
    output = ""
    for sentence in sentences:
        output += output_format.writeSentence(sentence)
    output += output_format.finishDocument()

    return output


class UDPipeModel:
    def __init__(self, path):
        """Load given model."""
        self.model = ufal.udpipe.Model.load(path)
        if not self.model:
            raise Exception("Cannot load UDPipe model from file '%s'" % path)

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self.read(text, tokenizer)

    @staticmethod
    def read(text, in_format):
        return read_parsed(text, in_format)

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence, self.model.DEFAULT)

    def parse(self, sentence):
        """Parse the given ufal.udpipe.Sentence (inplace)."""
        self.model.parse(sentence, self.model.DEFAULT)

    @staticmethod
    def write(sentences, out_format):
        return write_parsed(sentences, out_format)


# Can be used as
#  model = UDPipeModel('english-ud-1.2-160523.udpipe')
#  sentences = model.tokenize("Hi there. How are you?")
#  for s in sentences:
#      model.tag(s)
#      model.parse(s)
#  conllu = model.write(sentences, "conllu")
