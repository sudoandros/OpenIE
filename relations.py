import argparse
import io
import json
import string
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

from udpipe_model import UDPipeModel


class SentenceReltuples:
    def __init__(self, sentence, additional_relations=False):
        self._reltuples = []
        self._sentence = sentence
        self._string_tuples = None
        self._add_rel = additional_relations
        self._extract_reltuples()

    @property
    def sentence(self):
        return self._sentence

    def string_tuples(self, lemmatize_args=False):
        return tuple(
            self._to_string_tuple(reltuple, lemmatize_args=lemmatize_args)
            for reltuple in self._reltuples
        )

    def _to_string_tuple(self, reltuple, lemmatize_args=False):
        left_arg = self._arg_to_string(reltuple[0], lemmatized=lemmatize_args)
        relation = self._relation_to_string(reltuple[1])
        right_arg = self._arg_to_string(reltuple[2], lemmatized=lemmatize_args)
        return (left_arg, relation, right_arg)

    def _relation_to_string(self, relation):
        if isinstance(relation, list):
            string_ = " ".join(self._sentence.words[id_].form for id_ in relation)
        elif isinstance(relation, str):
            string_ = relation
        else:
            raise TypeError
        return self._clean_string(string_)

    def _arg_to_string(self, words_ids, lemmatized=False):
        if lemmatized:
            string_ = " ".join(self._sentence.words[id_].lemma for id_ in words_ids)
        else:
            string_ = " ".join(self._sentence.words[id_].form for id_ in words_ids)
        return self._clean_string(string_)

    def _extract_reltuples(self):
        for word in self._sentence.words:
            if word.deprel == "cop":
                self._reltuples.extend(self._get_copula_reltuples(word))
            elif word.upostag == "VERB":
                self._reltuples.extend(self._get_verb_reltuples(word))
        if self._add_rel:
            for left_arg, _, right_arg in self._reltuples.copy():
                self._reltuples.extend(self._get_additional_reltuples(left_arg))
                self._reltuples.extend(self._get_additional_reltuples(right_arg))

    def _get_verb_reltuples(self, verb):
        for child_id in verb.children:
            child = self._sentence.words[child_id]
            if child.deprel == "xcomp":
                return ()
        subjects = self._get_subjects(verb)
        right_args = self._get_right_args(verb)
        return tuple(
            (subj, self._get_relation(verb, right_arg=arg), arg)
            for subj in subjects
            for arg in right_args
        )

    def _get_copula_reltuples(self, copula):
        right_arg = self._get_right_args(copula)[0]
        parent = self._sentence.words[copula.head]
        subjects = self._get_subjects(parent)
        relation = self._get_copula(copula)
        return tuple((subj, relation, right_arg) for subj in subjects)

    def _get_additional_reltuples(self, words_ids):
        result = ()
        root = self._get_root(words_ids)
        main_phrase_ids = words_ids
        upper = (
            "appos",
            "flat",
            "flat:foreign",
            "flat:name",
            "nummod",
            "nummod:entity",
            "nummod:gov",
            "conj",
        )
        dependent = ("nmod",)
        children_ids = [id_ for id_ in words_ids if id_ in root.children]
        for child_id in children_ids:
            child = self._sentence.words[child_id]
            descendants_ids = []
            if child.deprel in upper:
                subtree = self._get_subtree(child)
                descendants_ids = [id_ for id_ in words_ids if id_ in subtree]
                result += ((descendants_ids, "выше", words_ids),)
            elif child.deprel in dependent:
                subtree = self._get_subtree(child)
                descendants_ids = [id_ for id_ in words_ids if id_ in subtree]
                result += ((descendants_ids, "часть", words_ids),)
            self._get_additional_reltuples(descendants_ids)
            main_phrase_ids = [
                id_ for id_ in main_phrase_ids if id_ not in descendants_ids
            ]
        if len(words_ids) != len(main_phrase_ids):
            result += ((main_phrase_ids, "выше", words_ids),)
        if len(main_phrase_ids) > 1:
            result += (([root.id], "выше", main_phrase_ids),)
        return result

    def _get_relation(self, word, right_arg=None):
        prefix = self._get_relation_prefix(word)
        postfix = self._get_relation_postfix(word, right_arg=right_arg)
        relation = prefix + [word.id] + postfix
        return relation

    def _get_relation_prefix(self, relation):
        prefix = []
        for child_id in relation.children:
            child = self._sentence.words[child_id]
            if (
                child.deprel == "case"
                or child.deprel == "aux"
                or child.deprel == "aux:pass"
                or child.upostag == "PART"
            ) and child.id < relation.id:
                prefix.append(child.id)
        parent = self._sentence.words[relation.head]
        if relation.deprel == "xcomp":
            prefix = self._get_relation(parent) + prefix
        if self._is_conjunct(relation) and parent.deprel == "xcomp":
            grandparent = self._sentence.words[parent.head]
            prefix = self._get_relation(grandparent) + prefix
        return prefix

    def _get_relation_postfix(self, relation, right_arg=None):
        postfix = []
        for child_id in relation.children:
            child = self._sentence.words[child_id]
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
        parent = self._sentence.words[word.head]
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
        args_list = []
        for child_id in word.children:
            child = self._sentence.words[child_id]
            if self._is_right_arg(child):
                args_list.append(self._get_subtree(child))
        parent = self._sentence.words[word.head]
        if word.deprel == "xcomp":
            args_list += self._get_verb_right_args(parent)
        if self._is_conjunct(word) and parent.deprel == "xcomp":
            grandparent = self._sentence.words[parent.head]
            args_list += self._get_verb_right_args(grandparent)
        return args_list

    def _get_subjects(self, word):
        subj_list = []
        for child_id in word.children:
            child = self._sentence.words[child_id]
            if self._is_subject(child):
                subj_list.append(self._get_subtree(child))
        if not subj_list and (word.deprel == "conj" or word.deprel == "xcomp"):
            parent = self._sentence.words[word.head]
            subj_list = self._get_subjects(parent)
        return subj_list

    def _get_subtree(self, word):
        if not list(word.children):
            return [word.id]
        res_ids = []
        for child_id in (id for id in word.children if id < word.id):
            child = self._sentence.words[child_id]
            res_ids.extend(self._get_subtree(child))
        res_ids.append(word.id)
        for child_id in (id for id in word.children if id > word.id):
            child = self._sentence.words[child_id]
            res_ids.extend(self._get_subtree(child))
        return res_ids

    def _get_first_case(self, words_ids):
        root = self._get_root(words_ids)
        for id_ in words_ids:
            word = self._sentence.words[id_]
            if id_ < root.id and word.deprel == "case":
                return id_
        return None

    def _get_copula(self, word):
        parent = self._sentence.words[word.head]
        part_ids = []
        for sibling_id in parent.children:
            sibling = self._sentence.words[sibling_id]
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
            child = self._sentence.words[child_id]
            if child.deprel == "cop":
                res.append(self._get_copula(child))
        return res

    def _get_root(self, words_ids):
        if not words_ids:
            return None
        for id_ in words_ids:
            word = self._sentence.words[id_]
            if word.head not in words_ids:
                root = word
        return root

    def _is_subject(self, word):
        return word.deprel in ("nsubj", "nsubj:pass")

    def _is_right_arg(self, word):
        return word.deprel in ("obj", "iobj", "obl", "obl:agent", "iobl")

    def _is_conjunct(self, word):
        return word.deprel == "conj"

    def _clean_string(self, string_):
        res = (
            "".join(
                char
                for char in string_
                if char.isalnum() or char.isspace() or char in ",.;-—/:%"
            )
            .lower()
            .strip(" .,:;-")
        )
        return res


class RelGraph:
    def __init__(self, stopwords):
        self._graph = nx.DiGraph()
        self._stopwords = set(stopwords)

    @classmethod
    def from_reltuples_list(cls, stopwords, reltuples_list):
        graph = cls(stopwords)
        for sentence_reltuple in reltuples_list:
            graph.add_sentence_reltuples(sentence_reltuple)

    @property
    def nodes_number(self):
        return self._graph.number_of_nodes()

    @property
    def edges_number(self):
        return self._graph.number_of_edges()

    def add_sentence_reltuples(self, sentence_reltuples):
        sentence_text = sentence_reltuples.sentence.getText()
        for (
            (left_arg, relation, right_arg),
            (left_arg_lemmatized, _, right_arg_lemmatized),
        ) in zip(
            sentence_reltuples.string_tuples(),
            sentence_reltuples.string_tuples(lemmatize_args=True),
        ):
            self._add_node(left_arg_lemmatized, sentence_text, label=left_arg)
            self._add_node(right_arg_lemmatized, sentence_text, label=right_arg)
            self._add_edge(
                left_arg_lemmatized, right_arg_lemmatized, relation, sentence_text
            )

    def _add_edge(self, source, target, label, description):
        if source not in self._graph or target not in self._graph:
            return
        if not self._graph.has_edge(source, target):
            if label == "выше":
                self._graph.add_edge(
                    source,
                    target,
                    label=label,
                    description=description,
                    weight=1,
                    viz={"color": {"b": 255, "g": 0, "r": 0}},
                )
            elif label == "часть":
                self._graph.add_edge(
                    source,
                    target,
                    label=label,
                    description=description,
                    weight=1,
                    viz={"color": {"b": 0, "g": 255, "r": 0}},
                )
            else:
                self._graph.add_edge(
                    source, target, label=label, description=description, weight=1
                )
            return
        # this edge already exists
        if label not in self._graph[source][target]["label"].split(" | "):
            self._graph[source][target]["label"] = "{} | {}".format(
                self._graph[source][target]["label"], label
            )
        if description not in self._graph[source][target]["description"].split(" | "):
            self._graph[source][target]["description"] = "{} | {}".format(
                self._graph[source][target]["description"], description
            )
        self._graph[source][target]["weight"] += 1

    def _add_node(self, name, description, label=None):
        if label is None:
            label = name
        if set(name.split()).issubset(self._stopwords) or (
            len(name) == 1 and name.isalpha()
        ):
            return
        if name not in self._graph:
            self._graph.add_node(name, label=label, description=description, weight=1)
            return
        # this node already exists
        if description not in self._graph.nodes[name]["description"].split(" | "):
            self._graph.nodes[name]["description"] = "{} | {}".format(
                self._graph.nodes[name]["description"], description
            )
        self._graph.nodes[name]["weight"] += 1

    def save(self, path):
        stream_buffer = io.BytesIO()
        nx.write_gexf(self._graph, stream_buffer, encoding="utf-8", version="1.1draft")
        xml_string = stream_buffer.getvalue().decode("utf-8")
        root_element = ET.fromstring(xml_string)
        self._fix_gexf(root_element)
        ET.register_namespace("", "http://www.gexf.net/1.1draft")
        xml_tree = ET.ElementTree(root_element)
        xml_tree.write(path, encoding="utf-8")

    def _fix_gexf(self, root_element):
        graph_node = root_element.find("{http://www.gexf.net/1.1draft}graph")
        attributes_nodes = graph_node.findall(
            "{http://www.gexf.net/1.1draft}attributes"
        )
        edge_attributes = {}
        node_attributes = {}
        for attributes_node in attributes_nodes:
            for attribute_node in attributes_node.findall(
                "{http://www.gexf.net/1.1draft}attribute"
            ):
                attr_id = attribute_node.get("id")
                attr_title = attribute_node.get("title")
                attribute_node.set("id", attr_title)
                if attributes_node.get("class") == "edge":
                    edge_attributes[attr_id] = attr_title
                elif attributes_node.get("class") == "node":
                    node_attributes[attr_id] = attr_title
        nodes_node = graph_node.find("{http://www.gexf.net/1.1draft}nodes")
        for node_node in nodes_node.findall("{http://www.gexf.net/1.1draft}node"):
            attvalues_node = node_node.find("{http://www.gexf.net/1.1draft}attvalues")
            if attvalues_node is not None:
                for attvalue_node in attvalues_node.findall(
                    "{http://www.gexf.net/1.1draft}attvalue"
                ):
                    attr_for = attvalue_node.get("for")
                    attvalue_node.set("for", node_attributes[attr_for])
        edges_node = graph_node.find("{http://www.gexf.net/1.1draft}edges")
        for edge_node in edges_node.findall("{http://www.gexf.net/1.1draft}edge"):
            attvalues_node = edge_node.find("{http://www.gexf.net/1.1draft}attvalues")
            if attvalues_node is not None:
                for attvalue_node in attvalues_node.findall(
                    "{http://www.gexf.net/1.1draft}attvalue"
                ):
                    attr_for = attvalue_node.get("for")
                    if edge_attributes[attr_for] == "label":
                        attr_value = attvalue_node.get("value")
                        edge_node.set("label", attr_value)
                        attvalues_node.remove(attvalue_node)
                    attvalue_node.set("for", edge_attributes[attr_for])


def get_text_relations(
    conllu, udpipe_model, stopwords, additional_relations, nodes_limit
):
    dict_out = {}
    graph = RelGraph(stopwords)
    sentences = udpipe_model.read(conllu, "conllu")

    for s in sentences:
        reltuples = SentenceReltuples(s, additional_relations=additional_relations)
        dict_out[s.getText()] = reltuples.string_tuples()
        graph.add_sentence_reltuples(reltuples)
        if graph.nodes_number > nodes_limit:
            break

    return graph, dict_out


def build_dir_graph(
    conllu_dir, save_dir, udpipe_model, stopwords, additional_relations, nodes_limit
):
    conllu = ""

    for path in tqdm(conllu_dir.iterdir()):
        if not (path.suffix == ".conllu"):
            continue
        with path.open("r", encoding="utf8") as file:
            conllu += "\n" + file.read()

    graph, dict_out = get_text_relations(
        conllu, udpipe_model, stopwords, additional_relations, nodes_limit
    )

    json_path = save_dir / ("reltuples.json")
    with json_path.open("w", encoding="utf8") as file:
        json.dump(dict_out, file, ensure_ascii=False, indent=4)

    graph.save(save_dir / "graph{}.gexf".format(conllu_dir.name))
    print(graph.nodes_number, graph.edges_number)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the UDPipe model")
    parser.add_argument(
        "conllu_dir",
        help="Path to the directory containing parsed text in conllu format",
    )
    parser.add_argument("save_dir", help="Path to the directory to save relations to")
    parser.add_argument(
        "--add", help="Include additional relations", action="store_true"
    )
    parser.add_argument(
        "--nodes-limit",
        help="Stop when after processing of another sentence these number "
        "of nodes will be exceeded",
        type=int,
    )
    args = parser.parse_args()
    conllu_dir = Path(args.conllu_dir)
    save_dir = Path(args.save_dir)
    model = UDPipeModel(args.model_path)
    nodes_limit = args.nodes_limit or float("inf")
    with open("stopwords.txt", mode="r", encoding="utf-8") as file:
        stopwords = list(file.read().split())

    build_dir_graph(conllu_dir, save_dir, model, stopwords, args.add, nodes_limit)

