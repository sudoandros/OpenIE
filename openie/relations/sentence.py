import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Set, Tuple, Union

import numpy as np
from spacy.tokens import Doc, Span, Token


@dataclass(frozen=True)
class Argument:
    phrase: str
    lemmas: str
    vector: np.ndarray
    deprel: Optional[str] = None


@dataclass(frozen=True)
class Relation:
    phrase: str
    lemmas: str


@dataclass(frozen=True)
class Reltuple:
    left_arg: Argument
    relation: Relation
    right_arg: Argument

    def __str__(self) -> str:
        return (
            f"{self.left_arg.phrase}; {self.relation.phrase}; {self.right_arg.phrase}"
        )


class SentenceReltuples:
    def __init__(
        self,
        sentence: Span,
        w2v_model,
        additional_relations: bool = False,
        stopwords: Iterable[str] | None = None,
    ):
        self.sentence = sentence
        self.sentence_vector = _get_phrase_vector(list(sentence), w2v_model)
        self._stopwords: Set[str] = set() if stopwords is None else set(stopwords)
        words_ids_tuples = self._get_words_tuples(
            additional_relations=additional_relations
        )
        self._reltuples = [self._to_reltuple(t, w2v_model) for t in words_ids_tuples]
        self._reltuples = [
            reltuple
            for reltuple in self._reltuples
            if reltuple.left_arg.phrase != reltuple.right_arg.phrase
        ]
        logging.info(
            f"{len(self._reltuples)} relations were extracted from the sentence {self.sentence}:\n"
            + "\n".join(f"({reltuple})" for reltuple in self._reltuples)
        )

    def __getitem__(self, index: int):
        return self._reltuples[index]

    def _to_reltuple(
        self,
        reltuple: Tuple[list[Token], Union[list[Token], str], list[Token]],
        w2v_model,
    ):
        left_arg = Argument(
            phrase=self._arg_to_string(reltuple[0], lemmatized=False),
            lemmas=self._arg_to_string(reltuple[0], lemmatized=True),
            vector=_get_phrase_vector(reltuple[0], w2v_model),
        )
        relation = Relation(
            phrase=self._relation_to_string(reltuple[1]),
            lemmas=self._relation_to_string(reltuple[1], lemmatized=True),
        )
        right_arg = Argument(
            phrase=self._arg_to_string(reltuple[2], lemmatized=False),
            lemmas=self._arg_to_string(reltuple[2], lemmatized=True),
            deprel=self._get_root(reltuple[2]).dep_,
            vector=_get_phrase_vector(reltuple[2], w2v_model),
        )

        return Reltuple(left_arg, relation, right_arg)

    def _relation_to_string(
        self, relation: Union[list[Token], str], lemmatized: bool = False
    ):
        match relation:
            case list() if not lemmatized:
                res = " ".join(str(w) for w in relation)
            case list() if lemmatized:
                res = " ".join(w.lemma_ for w in relation)
            case str():
                res = relation
            case _:
                raise TypeError

        return self._clean_string(res)

    def _arg_to_string(self, words: list[Token], lemmatized: bool = False):
        if lemmatized:
            res = " ".join(w.lemma_.strip() for w in words)
        else:
            res = " ".join(str(w).strip() for w in words)
        return self._clean_string(res)

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

    def _get_words_tuples(self, additional_relations: bool = False):
        result: list[tuple[list[Token], list[Token] | str, list[Token]]] = []
        for token in self.sentence:
            if token.dep_ == "cop":
                result += self._get_copula_reltuples(token)
            elif token.pos_ == "VERB":
                result += self._get_verb_reltuples(token)

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

    def _get_verb_reltuples(self, verb: Token):
        for child in verb.children:
            if child.dep_ == "xcomp":
                return []
        subjects = self._get_subjects(verb)
        right_args = self._get_right_args(verb)
        return [
            (subj, self._get_relation(verb, right_arg=arg), arg)
            for subj in subjects
            for arg in right_args
        ]

    def _get_copula_reltuples(self, copula: Token):
        right_arg = self._get_right_args(copula)[0]
        subjects = self._get_subjects(copula.head)
        relation = self._get_copula(copula)
        return [(subj, relation, right_arg) for subj in subjects]

    def _get_additional_reltuples(self, words: list[Token]):
        result: list[Tuple[list[Token], str, list[Token]]] = []
        is_a_deps = ["appos", "flat", "flat:foreign", "flat:name", "conj"]
        relates_to_deps = ["nmod"]
        main_phrase = words
        root = self._get_root(words)
        children = [w for w in root.children if w in words]

        for child in children:
            if child.dep_ in is_a_deps:
                descendants = [w for w in words if w in child.subtree]
                result.append((words, "_is_a_", descendants))
                result += self._get_additional_reltuples(descendants)
                main_phrase = [w for w in main_phrase if w not in descendants]
        if len(words) != len(main_phrase):  # found "is_a" relation?
            result.append((words, "_is_a_", main_phrase))
            result += self._get_additional_reltuples(main_phrase)
            return result

        old_main_phrase_length = len(main_phrase)
        for child in children:
            if child.dep_ in relates_to_deps:
                descendants = [id_ for id_ in words if id_ in child.subtree]
                result.append((words, "_relates_to_", descendants))
                result += self._get_additional_reltuples(descendants)
                main_phrase = [w for w in main_phrase if w not in descendants]
        if old_main_phrase_length != len(main_phrase):  # found "relates_to" relation?
            result.append((words, "_is_a_", main_phrase))
            result += self._get_additional_reltuples(main_phrase)
        elif len(main_phrase) > 1:
            result.append((main_phrase, "_is_a_", [root]))
        return result

    def _get_relation(self, word: Token, right_arg: list[Token] | None = None):
        prefix = self._get_relation_prefix(word)
        postfix = self._get_relation_postfix(word, right_arg=right_arg)
        relation: list[Token] = [*prefix, word, *postfix]
        return relation

    def _get_relation_prefix(self, relation: Token):
        prefix: list[Token] = []
        for child in relation.children:
            if (
                child.dep_ == "case"
                or child.dep_ == "aux"
                or child.dep_ == "aux:pass"
                or child.pos_ == "PART"
            ) and child.i < relation.i:
                prefix.append(child)
        if relation.dep_ == "xcomp":
            prefix = self._get_relation(relation.head) + prefix
        if relation.dep_ == "conj" and relation.head.dep_ == "xcomp":
            prefix = self._get_relation(relation.head.head) + prefix
        return prefix

    def _get_relation_postfix(
        self, relation: Token, right_arg: list[Token] | None = None
    ):
        postfix: list[Token] = []
        for child in relation.children:
            if (
                child.dep_ == "case"
                or child.dep_ == "aux"
                or child.dep_ == "aux:pass"
                or child.pos_ == "PART"
            ) and child.i > relation.i:
                postfix.append(child)
        if right_arg:
            case = self._get_first_case(right_arg)
            if case:
                postfix.append(case)
                right_arg.remove(case)  # TODO get rid of this shit
        return postfix

    def _get_right_args(self, word: Token):
        if word.dep_ == "cop":
            args_list = self._get_copula_right_args(word)
        else:
            args_list = self._get_verb_right_args(word)
        return args_list

    def _get_copula_right_args(self, word: Token):
        words = list(word.head.subtree)

        copulas = self._get_all_copulas(word.head)
        for copula_words in copulas:
            for word in copula_words:
                words.remove(word)

        subjects = self._get_subjects(word.head)
        for subj in subjects:
            for word in subj:
                try:
                    words.remove(word)
                except ValueError:
                    continue
        return [words]

    def _get_verb_right_args(self, word: Token):
        args_list: list[list[Token]] = []
        for child in word.children:
            if self._is_right_arg(child):
                args_list.append(list(child.subtree))
        if word.dep_ == "xcomp":
            args_list += self._get_verb_right_args(word.head)
        if word.dep_ == "conj" and word.head.dep_ == "xcomp":
            args_list += self._get_verb_right_args(word.head.head)
        return args_list

    def _get_subjects(self, word: Token):
        subjects: list[list[Token]] = []
        for child in word.children:
            if self._is_subject(child):
                subjects.append(list(child.subtree))
        if not subjects and (word.dep_ == "conj" or word.dep_ == "xcomp"):
            subjects = self._get_subjects(word.head)
        return subjects

    def _get_first_case(self, words: list[Token]):
        root = self._get_root(words)
        cases = [
            word
            for left in root.lefts
            for word in left.subtree
            if word in words and word.dep_ == "case"
        ]
        return cases[0] if cases else None

    def _get_copula(self, word: Token):
        parts: list[Token] = []
        for sibling in word.head.children:
            if sibling == word:
                break
            if sibling.pos_ == "PART":
                parts.append(sibling)
            else:
                parts = []
        return parts + [word]

    def _get_all_copulas(self, word: Token):
        return [self._get_copula(w) for w in word.subtree if w.dep_ == "cop"]

    def _get_root(self, words: list[Token]):
        words_sorted = sorted(words, key=lambda x: len(list(x.ancestors)))
        return words_sorted[0]

    def _is_stopwords(self, words: list[Token]) -> bool:
        return {word.lemma_ for word in words}.issubset(self._stopwords) or (
            len(words) == 1 and len(words[0].lemma_) == 1 and words[0].lemma_.isalpha()
        )

    @staticmethod
    def _is_subject(word: Token):
        return word.dep_ in ("nsubj", "nsubj:pass")

    @staticmethod
    def _is_right_arg(word: Token):
        return word.dep_ in ("obj", "iobj", "obl", "obl:agent", "iobl")


def _get_phrase_vector(words: list[Token], w2v_model) -> np.ndarray:
    vector = np.zeros(300)
    count = 0
    for word in words:
        w2v_key = "{}_{}".format(word.lemma_, word.pos_)
        try:
            vector = vector + w2v_model[w2v_key]
            count += 1
        except KeyError:
            continue
    if count > 0:
        return vector / count
    else:
        return vector
