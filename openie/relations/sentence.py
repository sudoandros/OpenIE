import logging
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Set, Tuple, Union

import numpy as np


@dataclass
class Reltuple:
    left_arg: str
    left_arg_lemmas: str
    left_w2v: np.ndarray
    relation: str
    relation_lemmas: str
    right_arg: str
    right_arg_lemmas: str
    right_deprel: str
    right_w2v: np.ndarray


class SentenceReltuples:
    def __init__(
        self,
        sentence,
        w2v_model,
        additional_relations: bool = False,
        stopwords: Iterable[str] = None,
    ):
        self.sentence = sentence
        self.sentence_vector = _get_phrase_vector(sentence, "all", w2v_model)
        self._stopwords: Set[str] = set() if stopwords is None else set(stopwords)
        words_ids_tuples = self._get_words_ids_tuples(
            additional_relations=additional_relations
        )
        self._reltuples = [self._to_tuple(t, w2v_model) for t in words_ids_tuples]
        self._reltuples = [
            reltuple
            for reltuple in self._reltuples
            if reltuple.left_arg != reltuple.right_arg
        ]
        logging.info(
            f"{len(self._reltuples)} relations were extracted from the sentence {self.sentence.getText()}:\n"
            + "\n".join(
                f"({reltuple.left_arg}, {reltuple.relation}, {reltuple.right_arg})"
                for reltuple in self._reltuples
            )
        )

    def __getitem__(self, index: int):
        return self._reltuples[index]

    def _to_tuple(
        self, reltuple: Tuple[List[int], Union[List[int], str], List[int]], w2v_model,
    ):
        left_arg = self._arg_to_string(reltuple[0], lemmatized=False)
        left_arg_lemmas = self._arg_to_string(reltuple[0], lemmatized=True)
        left_w2v = _get_phrase_vector(self.sentence, reltuple[0], w2v_model)

        relation = self._relation_to_string(reltuple[1])
        relation_lemmas = self._relation_to_string(reltuple[1], lemmatized=True)

        right_arg = self._arg_to_string(reltuple[2], lemmatized=False)
        right_arg_lemmas = self._arg_to_string(reltuple[2], lemmatized=True)
        right_deprel: str = self.sentence.words[self._get_root(reltuple[2]).id].deprel
        right_w2v = _get_phrase_vector(self.sentence, reltuple[2], w2v_model)

        return Reltuple(
            left_arg,
            left_arg_lemmas,
            left_w2v,
            relation,
            relation_lemmas,
            right_arg,
            right_arg_lemmas,
            right_deprel,
            right_w2v,
        )

    def _relation_to_string(
        self, relation: Union[List[int], str], lemmatized: bool = False
    ):
        if isinstance(relation, list) and not lemmatized:
            string_ = " ".join(self.sentence.words[id_].form for id_ in relation)
        elif isinstance(relation, list) and lemmatized:
            string_ = " ".join(self.sentence.words[id_].lemma for id_ in relation)
        elif isinstance(relation, str):
            string_ = relation
        else:
            raise TypeError
        return self._clean_string(string_)

    def _arg_to_string(self, words_ids: List[int], lemmatized: bool = False):
        if lemmatized:
            string_ = " ".join(
                self.sentence.words[id_].lemma.strip() for id_ in words_ids
            )
        else:
            string_ = " ".join(
                self.sentence.words[id_].form.strip() for id_ in words_ids
            )
        return self._clean_string(string_)

    @staticmethod
    def _clean_string(string_: str):
        res = (
            "".join(
                char
                for char in string_
                if char.isalnum() or char.isspace() or char in ",.;-—_/:%"
            )
            .lower()
            .strip(" .,:;-")
        )
        return res

    def _get_words_ids_tuples(self, additional_relations: bool = False):
        result: List[Tuple[List[int], Union[List[int], str], List[int]]] = []
        for word in self.sentence.words:
            if word.deprel == "cop":
                result += self._get_copula_reltuples(word)
            elif word.upostag == "VERB":
                result += self._get_verb_reltuples(word)
        if additional_relations:
            args = {tuple(left_arg) for left_arg, _, _ in result} | {
                tuple(right_arg) for _, _, right_arg in result
            }
            for arg in args:
                result += self._get_additional_reltuples(list(arg))
        return [
            (left_arg, relation, right_arg)
            for left_arg, relation, right_arg in result
            if not self._is_stopwords(left_arg) and not self._is_stopwords(right_arg)
        ]

    def _get_verb_reltuples(self, verb):
        for child_id in verb.children:
            child = self.sentence.words[child_id]
            if child.deprel == "xcomp":
                return []
        subjects = self._get_subjects(verb)
        right_args = self._get_right_args(verb)
        return [
            (subj, self._get_relation(verb, right_arg=arg), arg)
            for subj in subjects
            for arg in right_args
        ]

    def _get_copula_reltuples(self, copula):
        right_arg = self._get_right_args(copula)[0]
        parent = self.sentence.words[copula.head]
        subjects = self._get_subjects(parent)
        relation = self._get_copula(copula)
        return [(subj, relation, right_arg) for subj in subjects]

    def _get_additional_reltuples(self, words_ids: List[int]):
        result: List[Tuple[List[int], str, List[int]]] = []
        is_a_deprels = ["appos", "flat", "flat:foreign", "flat:name", "conj"]
        relates_to_deprels = ["nmod"]
        main_phrase_ids = words_ids
        root = self._get_root(words_ids)
        children_ids = [id_ for id_ in words_ids if id_ in root.children]

        for child_id in children_ids:
            child = self.sentence.words[child_id]
            if child.deprel in is_a_deprels:
                subtree = self._get_subtree(child)
                descendants_ids = [id_ for id_ in words_ids if id_ in subtree]
                result.append((words_ids, "_is_a_", descendants_ids))
                result += self._get_additional_reltuples(descendants_ids)
                main_phrase_ids = [
                    id_ for id_ in main_phrase_ids if id_ not in descendants_ids
                ]
        if len(words_ids) != len(main_phrase_ids):  # found "is_a" relation?
            result.append((words_ids, "_is_a_", main_phrase_ids))
            result += self._get_additional_reltuples(main_phrase_ids)
            return result

        old_main_phrase_length = len(main_phrase_ids)
        for child_id in children_ids:
            child = self.sentence.words[child_id]
            if child.deprel in relates_to_deprels:
                subtree = self._get_subtree(child)
                descendants_ids = [id_ for id_ in words_ids if id_ in subtree]
                result.append((words_ids, "_relates_to_", descendants_ids))
                result += self._get_additional_reltuples(descendants_ids)
                main_phrase_ids = [
                    id_ for id_ in main_phrase_ids if id_ not in descendants_ids
                ]
        if old_main_phrase_length != len(
            main_phrase_ids
        ):  # found "relates_to" relation?
            result.append((words_ids, "_is_a_", main_phrase_ids))
            result += self._get_additional_reltuples(main_phrase_ids)
        elif len(main_phrase_ids) > 1:
            result.append((main_phrase_ids, "_is_a_", [root.id]))
        return result

    def _get_relation(self, word, right_arg: Optional[List[int]] = None):
        prefix = self._get_relation_prefix(word)
        postfix = self._get_relation_postfix(word, right_arg=right_arg)
        relation: List[int] = prefix + [word.id] + postfix
        return relation

    def _get_relation_prefix(self, relation):
        prefix: List[int] = []
        for child_id in relation.children:
            child = self.sentence.words[child_id]
            if (
                child.deprel == "case"
                or child.deprel == "aux"
                or child.deprel == "aux:pass"
                or child.upostag == "PART"
            ) and child.id < relation.id:
                prefix.append(child.id)
        parent = self.sentence.words[relation.head]
        if relation.deprel == "xcomp":
            prefix = self._get_relation(parent) + prefix
        if self._is_conjunct(relation) and parent.deprel == "xcomp":
            grandparent = self.sentence.words[parent.head]
            prefix = self._get_relation(grandparent) + prefix
        return prefix

    def _get_relation_postfix(self, relation, right_arg: Optional[List[int]] = None):
        postfix: List[int] = []
        for child_id in relation.children:
            child = self.sentence.words[child_id]
            if (
                child.deprel == "case"
                or child.deprel == "aux"
                or child.deprel == "aux:pass"
                or child.upostag == "PART"
            ) and child.id > relation.id:
                postfix.append(child.id)
        if right_arg:
            case_id = self._get_first_case(right_arg)
            if case_id is not None:
                postfix.append(case_id)
                right_arg.remove(case_id)
        return postfix

    def _get_right_args(self, word):
        if word.deprel == "cop":
            args_list = self._get_copula_right_args(word)
        else:
            args_list = self._get_verb_right_args(word)
        return args_list

    def _get_copula_right_args(self, word):
        parent = self.sentence.words[word.head]
        words_ids = self._get_subtree(parent)
        copulas = self._get_all_copulas(parent)
        for copula_words_ids in copulas:
            for id_ in copula_words_ids:
                words_ids.remove(id_)
        subjects = self._get_subjects(parent)
        for subj in subjects:
            for id_to_remove in subj:
                try:
                    words_ids.remove(id_to_remove)
                except ValueError:
                    continue
        return [words_ids]

    def _get_verb_right_args(self, word):
        args_list: List[List[int]] = []
        for child_id in word.children:
            child = self.sentence.words[child_id]
            if self._is_right_arg(child):
                args_list.append(self._get_subtree(child))
        parent = self.sentence.words[word.head]
        if word.deprel == "xcomp":
            args_list += self._get_verb_right_args(parent)
        if self._is_conjunct(word) and parent.deprel == "xcomp":
            grandparent = self.sentence.words[parent.head]
            args_list += self._get_verb_right_args(grandparent)
        return args_list

    def _get_subjects(self, word):
        subj_list: List[List[int]] = []
        for child_id in word.children:
            child = self.sentence.words[child_id]
            if self._is_subject(child):
                subj_list.append(self._get_subtree(child))
        if not subj_list and (word.deprel == "conj" or word.deprel == "xcomp"):
            parent = self.sentence.words[word.head]
            subj_list = self._get_subjects(parent)
        return subj_list

    def _get_subtree(self, word) -> List[int]:
        if not list(word.children):
            return [word.id]
        res_ids = []
        for child_id in (id for id in word.children if id < word.id):
            child = self.sentence.words[child_id]
            res_ids.extend(self._get_subtree(child))
        res_ids.append(word.id)
        for child_id in (id for id in word.children if id > word.id):
            child = self.sentence.words[child_id]
            res_ids.extend(self._get_subtree(child))
        return res_ids

    def _get_first_case(self, words_ids: List[int]):
        root = self._get_root(words_ids)
        for id_ in words_ids:
            word = self.sentence.words[id_]
            if id_ < root.id and word.deprel == "case":
                return id_
        return None

    def _get_copula(self, word):
        parent = self.sentence.words[word.head]
        part_ids: List[int] = []
        for sibling_id in parent.children:
            sibling = self.sentence.words[sibling_id]
            if sibling.id == word.id:
                return part_ids + [sibling.id]
            if sibling.upostag == "PART":
                part_ids.append(sibling.id)
            else:
                part_ids = []
        return []

    def _get_all_copulas(self, word):
        res = []
        for child_id in word.children:
            child = self.sentence.words[child_id]
            if child.deprel == "cop":
                res.append(self._get_copula(child))
        return res

    def _get_root(self, words_ids: List[int]):
        root = None
        for id_ in words_ids:
            word = self.sentence.words[id_]
            if word.head not in words_ids:
                root = word
        return root

    def _is_stopwords(self, words_ids: List[int]) -> bool:
        return {self.sentence.words[id_].lemma for id_ in words_ids}.issubset(
            self._stopwords
        ) or (
            len(words_ids) == 1
            and len(self.sentence.words[words_ids[0]].lemma) == 1
            and self.sentence.words[words_ids[0]].lemma.isalpha()
        )

    @staticmethod
    def _is_subject(word):
        return word.deprel in ("nsubj", "nsubj:pass")

    @staticmethod
    def _is_right_arg(word):
        return word.deprel in ("obj", "iobj", "obl", "obl:agent", "iobl")

    @staticmethod
    def _is_conjunct(word):
        return word.deprel == "conj"


def _get_phrase_vector(
    sentence, words_ids: Union[List[int], Literal["all"]], w2v_model
) -> np.ndarray:
    if words_ids == "all":
        words_ids = list(range(len(sentence.words)))
    vector = np.zeros(300)
    count = 0
    for word_id in words_ids:
        try:
            vector = np.add(
                vector,
                w2v_model[
                    "{}_{}".format(
                        sentence.words[word_id].lemma, sentence.words[word_id].upostag
                    )
                ],
            )
            count += 1
        except KeyError:
            continue
    if count > 0:
        return vector / count
    else:
        return vector
