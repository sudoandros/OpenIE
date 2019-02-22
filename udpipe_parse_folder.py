import json
import sys
from pathlib import Path

from tqdm import tqdm
from ufal.udpipe import Sentence

from udpipe_model import Model


def text_from_sts(filePath):
    res = []
    with filePath.open('r', encoding='cp1251') as f:
        for line in f:
            res.append(' '.join(line.split()[6:]))
    return '\n'.join(res)
    # return res

def text_from_hdr(hdrPath):
    res = None
    with hdrPath.open('r', encoding='cp1251') as f:
        for line in f:
            if line.split('=', 1)[0] == 'TEXT_THEMAN_ANNO':
                res = line.split('=', 1)[1]
    return res

if __name__ == '__main__':
    dirPath = Path(sys.argv[1])
    model = Model('udpipe_models/russian-syntagrus-ud-2.2-conll18-180430.udpipe')

    relations = []
    for stsPath in tqdm(dirPath.iterdir()):
        if stsPath.suffix != '.sts':
            continue
        text = text_from_sts(stsPath)

        sentences = model.tokenize(text)
        for s in sentences:
            model.tag(s)
            model.parse(s)

        conllu = model.write(sentences, "conllu")
        towrite = dirPath / (stsPath.stem + '_udpiped.conllu')
        with towrite.open('w', encoding='utf8') as f:
            f.write(conllu)
