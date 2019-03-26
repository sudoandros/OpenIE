import sys
import json
from tqdm import tqdm
from pathlib import Path
import argparse
from udpipe_model import UDPipeModel

UDPIPE_MODEL_PATH = "data/udpipe_models/russian-syntagrus-ud-2.3-181115.udpipe"


class SentenceReltuples:
    def __init__(self, sentence):
        self.reltuples = []
        self.sentence = sentence
        self._extract_reltuples()

    def reltuples_as_string_tuples(self):
        res = []
        for reltuple in self.reltuples:
            res.append(self.reltuple_to_string_tuple(reltuple))
        return res

    def reltuple_to_string_tuple(self, reltuple):
        left = self._entity_to_string(reltuple[0])
        center = self._relation_to_string(reltuple[1])
        right = self._entity_to_string(reltuple[2])
        return (left, center, right)

    def _extract_reltuples(self):
        self._extract_verb_reltuples()

    def _extract_verb_reltuples(self):
        verbs = [word for word in self.sentence.words if word.upostag == "VERB"]
        all_subjects = [self._get_subjects(verb) for verb in verbs]
        all_objects = [self._get_objects(verb) for verb in verbs]
        all_oblique_nominals = [self._get_oblique_nominals(verb) for verb in verbs]
        for i, verb in enumerate(verbs):
            verb_subjects = all_subjects[i]
            verb_objects = all_objects[i]
            verb_oblique_nominals = all_oblique_nominals[i]
            for subj in verb_subjects:
                for obj in verb_objects:
                    self.reltuples.append((subj, verb, obj))
            for subj in verb_subjects:
                for obl in verb_oblique_nominals:
                    self.reltuples.append((subj, verb, obl))

    def _relation_to_string(self, word):
        prefix = self._get_string_prefix(word)
        return prefix + word.form

    def _entity_to_string(self, word):
        strings = []
        if not list(word.children):
            return word.form
        for child_idx in (idx for idx in word.children if idx < word.id):
            child = self.sentence.words[child_idx]
            strings.append(self._entity_to_string(child))
        strings.append(word.form)
        for child_idx in (idx for idx in word.children if idx > word.id):
            child = self.sentence.words[child_idx]
            strings.append(self._entity_to_string(child))
        return " ".join(strings)

    def _get_string_prefix(self, word):
        res = ""
        for child_idx in word.children:
            child = self.sentence.words[child_idx]
            if child.deprel == "case":
                res += child.form + " "
            elif child.deprel == "aux":
                res += child.form + " "
            elif child.upostag == "PART":
                res += child.form + " "
        return res

    def _get_subjects(self, word):
        subj_list = []
        for child_idx in word.children:
            child = self.sentence.words[child_idx]
            if self._is_subject(child):
                subj_list.append(child)
                subj_list += self._get_conjuncts(child)
        return subj_list

    def _get_objects(self, word):
        obj_list = []
        for child_idx in word.children:
            child = self.sentence.words[child_idx]
            if self._is_object(child):
                obj_list.append(child)
                obj_list += self._get_conjuncts(child)
        return obj_list

    def _get_oblique_nominals(self, word):
        obl_list = []
        for child_idx in word.children:
            child = self.sentence.words[child_idx]
            if self._is_oblique_nominal(child):
                obl_list.append(child)
                obl_list += self._get_conjuncts(child)
        return obl_list

    def _get_conjuncts(self, word):
        conjuncts = []
        for child_idx in word.children:
            child = self.sentence.words[child_idx]
            if self._is_conjunct(child):
                conjuncts.append(child)
        return conjuncts

    def _is_subject(self, word):
        return word.deprel in ["nsubj", "nsubj:pass"]

    def _is_object(self, word):
        return word.deprel in ["obj", "iobj"]

    def _is_oblique_nominal(self, word):
        return word.deprel in ["obl", "obl:agent"]

    def _is_conjunct(self, word):
        return word.deprel == "conj"


def simple_test(model):
    sentences = model.tokenize(
        "Андрей пошел в магазин и аптеку, купил куртку и телефон, на улице становилось темно. Никита определенно не будет бегать в парке, а Андрей, Дима и Федор варили и жарили обед. Андрей, пока собирался на работу, съел завтрак."
    )
    for s in sentences:
        model.tag(s)
        model.parse(s)
        reltuples = SentenceReltuples(s)
    conllu = model.write(sentences, "conllu")
    with open("output.conllu", "w", encoding="utf8") as file:
        file.write(conllu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "conllu_dir",
        help="Path to the directory containing parsed text in conllu format",
    )
    parser.add_argument("save_dir", help="Path to the directory to save relations to")
    args = parser.parse_args()
    conllu_dir = Path(args.conllu_dir)
    save_dir = Path(args.save_dir)
    model = UDPipeModel(UDPIPE_MODEL_PATH)

    simple_test(model)

    for path in tqdm(conllu_dir.iterdir()):
        output = {}
        if not (path.suffix == ".conllu"):
            continue
        with path.open("r", encoding="utf8") as file:
            text = file.read()
        sentences = model.read(text, "conllu")
        for s in sentences:
            output[s.getText()] = SentenceReltuples(s).reltuples_as_string_tuples()

        output_path = save_dir / (path.stem + "_reltuples.json")
        with output_path.open("w", encoding="utf8") as file:
            json.dump(output, file, ensure_ascii=False, indent=4)
