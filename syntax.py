import html
import re


def parse_text(text, udpipe_model, format_=None):
    text = clean_text(text, format_=format_)
    conllu = get_conllu(udpipe_model, text)
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
    result = re.sub(r"([^.!?])(\s*\n+)", newline_repl, result)

    return result


def newline_repl(matchobj):
    return "{}. ".format(matchobj.group(1))


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


def get_conllu(udpipe_model, text):
    sentences = udpipe_model.tokenize(text)
    for s in sentences:
        udpipe_model.tag(s)
        udpipe_model.parse(s)

    conllu = udpipe_model.write(sentences, "conllu")
    return conllu
