import sys
import json
from tqdm import tqdm
from pathlib import Path
import argparse
from udpipe_model import UDPipeModel

UDPIPE_MODEL_PATH = "data/udpipe_models/russian-syntagrus-ud-2.3-181115.udpipe"


class SentenceRelations:
    def __init__(self, sentence):
        self.relations = []
        self.sentence = sentence
        self._extract_relations()

    def relations_as_strings_tuples(self):
        res = []
        for relation in self.relations:
            res.append(self.relation_to_strings_tuple(relation))
        return res

    def relation_to_strings_tuple(self, relation):
        left = self._word_to_string(relation[0])
        center = self._word_to_string(relation[1])
        right = self._word_to_string(relation[2])
        return (left, center, right)

    def _extract_relations(self):
        self._extract_verb_relations()

    def _extract_verb_relations(self):
        verbs = [word for word in self.sentence.words if word.upostag == "VERB"]
        all_subjects = [self._get_subjects(verb) for verb in verbs]
        all_objects = [self._get_objects(verb) for verb in verbs]
        all_oblique_nominals = [self._get_oblique_nominals(verb) for verb in verbs]
        for i, verb in enumerate(verbs):
            verb_subjects = all_subjects[i]
            verb_objects = all_objects[i]
            verb_oblique_nominals = all_oblique_nominals[i]
            if not verb_subjects:
                try:
                    verb_subjects = all_subjects[i - 1]
                    all_subjects[i] = verb_subjects
                except IndexError:
                    pass
            for subj in verb_subjects:
                for obj in verb_objects:
                    self.relations.append((subj, verb, obj))
            for subj in verb_subjects:
                for obl in verb_oblique_nominals:
                    self.relations.append((subj, verb, obl))

    def _word_to_string(self, word):
        prefix = self._get_string_prefix(word)
        return prefix + word.form

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
    relations = []
    sentences = model.tokenize(
        "Андрей пошел в магазин и аптеку, купил куртку и телефон, на улице становилось темно. Никита определенно не будет бегать в парке, а Андрей, Дима и Федор варили и жарили обед. Андрей, пока собирался на работу, съел завтрак."
    )
    for s in sentences:
        model.tag(s)
        model.parse(s)
        relations = SentenceRelations(s)
    conllu = model.write(sentences, "conllu")
    with open("output.conllu", "w", encoding="utf8") as f:
        f.write(conllu)


# TODO частицы
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("syntax_parser", choices=["syntaxnet", "udpipe"])
    parser.add_argument(
        "directory", help="Path to directory containing parsed text in conllu format"
    )
    args = parser.parse_args()
    dir_path = Path(args.directory)
    model = UDPipeModel(UDPIPE_MODEL_PATH)

    simple_test(model)

    # for path in tqdm(dir_path.iterdir()):
    #     relations = {}
    #     text = None
    #     if not (
    #         (args.syntax_parser == "syntaxnet" and "_syntaxnet.conllu" in path.name)
    #         or (args.syntax_parser == "udpipe" and "_udpiped.conllu" in path.name)
    #     ):
    #         continue

    #     with path.open("r", encoding="utf8") as f:
    #         text = f.read()

    #     sentences = model.read(text, "conllu")
    #     for s in sentences:
    #         relations[s.getText()] = get_relations(s)

    #     towrite = dir_path / (path.stem + "_relations.json")
    #     with towrite.open("w", encoding="utf8") as f:
    #         json.dump(relations, f, ensure_ascii=False, indent=4)
