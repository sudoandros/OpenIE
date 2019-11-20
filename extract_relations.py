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
    def __init__(self, sentence):
        self.reltuples = []
        self.sentence = sentence
        self._extract_reltuples()

    @property
    def string_tuples(self):
        res = []
        for reltuple in self.reltuples:
            res.append(self.to_string_tuple(reltuple))
        return res

    def to_string_tuple(self, reltuple):
        left = self.arg_to_string(reltuple[0])
        center = self.relation_to_string(reltuple[1])
        right = self.arg_to_string(reltuple[2])
        return (left, center, right)

    def _extract_reltuples(self):
        verbs = [word for word in self.sentence.words if word.upostag == "VERB"]
        for verb in verbs:
            self._extract_verb_reltuples(verb)

    def _extract_verb_reltuples(self, verb):
        for child_idx in verb.children:
            child = self.sentence.words[child_idx]
            if child.deprel == "xcomp":
                return
        verb_subjects = self._get_subjects(verb)
        verb_objects = self._get_objects(verb)
        verb_oblique_nominals = self._get_oblique_nominals(verb)
        for subj in verb_subjects:
            for obj in verb_objects:
                self.reltuples.append((subj, verb, obj))
        for subj in verb_subjects:
            for obl in verb_oblique_nominals:
                self.reltuples.append((subj, verb, obl))

    def relation_to_string(self, word):
        prefix = self._get_relation_prefix(word)
        postfix = self._get_relation_postfix(word)
        return prefix + word.form + postfix

    def _get_relation_prefix(self, word):
        prefix = ""
        for child_idx in word.children:
            child = self.sentence.words[child_idx]
            if (
                child.deprel == "case"
                or child.deprel == "aux"
                or child.deprel == "aux:pass"
                or child.upostag == "PART"
            ) and child.id < word.id:
                prefix += child.form + " "
        parent = self.sentence.words[word.head]
        if word.deprel == "xcomp":
            prefix = self.relation_to_string(parent) + " " + prefix
        if self._is_conjunct(word) and parent.deprel == "xcomp":
            grandparent = self.sentence.words[parent.head]
            prefix = self.relation_to_string(grandparent) + " " + prefix
        return prefix

    def _get_relation_postfix(self, word):
        postfix = ""
        for child_idx in word.children:
            child = self.sentence.words[child_idx]
            if (
                child.deprel == "case"
                or child.deprel == "aux"
                or child.deprel == "aux:pass"
                or child.upostag == "PART"
            ) and child.id > word.id:
                postfix += " " + child.form
        return postfix

    def arg_to_string(self, word):
        strings = []
        if not list(word.children):
            return word.form
        for child_idx in (idx for idx in word.children if idx < word.id):
            child = self.sentence.words[child_idx]
            strings.append(self.arg_to_string(child))
        strings.append(word.form)
        for child_idx in (idx for idx in word.children if idx > word.id):
            child = self.sentence.words[child_idx]
            strings.append(self.arg_to_string(child))
        return " ".join(strings)

    def _get_subjects(self, word):
        subj_list = []
        for child_idx in word.children:
            child = self.sentence.words[child_idx]
            if self._is_subject(child):
                subj_list.append(child)
        if not subj_list and (word.deprel == "conj" or word.deprel == "xcomp"):
            parent = self.sentence.words[word.head]
            subj_list = self._get_subjects(parent)
        return subj_list

    def _get_objects(self, word):
        obj_list = []
        for child_idx in word.children:
            child = self.sentence.words[child_idx]
            if self._is_object(child):
                obj_list.append(child)
        parent = self.sentence.words[word.head]
        if word.deprel == "xcomp":
            obj_list += self._get_objects(parent)
        if self._is_conjunct(word) and parent.deprel == "xcomp":
            grandparent = self.sentence.words[parent.head]
            obj_list += self._get_objects(grandparent)
        return obj_list

    def _get_oblique_nominals(self, word):
        obl_list = []
        for child_idx in word.children:
            child = self.sentence.words[child_idx]
            if self._is_oblique_nominal(child):
                obl_list.append(child)
        parent = self.sentence.words[word.head]
        if word.deprel == "xcomp":
            obl_list += self._get_oblique_nominals(parent)
        if self._is_conjunct(word) and parent.deprel == "xcomp":
            grandparent = self.sentence.words[parent.head]
            obl_list += self._get_oblique_nominals(grandparent)
        return obl_list

    def _is_subject(self, word):
        return word.deprel in ["nsubj", "nsubj:pass"]

    def _is_object(self, word):
        return word.deprel in ["obj", "iobj"]

    def _is_oblique_nominal(self, word):
        return word.deprel in ["obl", "obl:agent", "iobl"]

    def _is_conjunct(self, word):
        return word.deprel == "conj"


class RelGraph:
    def __init__(self, stopwords):
        self._graph = nx.DiGraph()
        self._stopwords = set(stopwords)

    @classmethod
    def from_reltuples_list(cls, stopwords, reltuples_list):
        graph = cls(stopwords)
        for sentence_reltuple in reltuples_list:
            graph.add_sentence_reltuples(sentence_reltuple)

    def add_sentence_reltuples(self, sentence_reltuples, include_syntax=False):
        for reltuple in sentence_reltuples.reltuples:
            self._add_reltuple(
                reltuple, sentence_reltuples, include_syntax=include_syntax
            )

    def _add_reltuple(self, reltuple, sentence_reltuples, include_syntax=False):
        source = sentence_reltuples.arg_to_string(reltuple[0])
        target = sentence_reltuples.arg_to_string(reltuple[2])
        relation = sentence_reltuples.relation_to_string(reltuple[1])
        sentence_text = sentence_reltuples.sentence.getText()
        source = self._clean_node(source)
        target = self._clean_node(target)
        relation = self._clean_node(relation)
        self._add_node(source, sentence_text)
        self._add_node(target, sentence_text)
        self._add_edge(source, target, relation, sentence_text)
        if include_syntax:
            self._add_syntax_tree(reltuple[0], sentence_reltuples)
            self._add_syntax_tree(reltuple[2], sentence_reltuples)

    def _add_edge(self, source, target, label, description):
        if source not in self._graph or target not in self._graph:
            return
        if not self._graph.has_edge(source, target):
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

    def _add_node(self, name, description):
        if set(name.split()).issubset(self._stopwords) or (
            len(name) == 1 and name.isalpha()
        ):
            return
        if name not in self._graph:
            self._graph.add_node(name, description=description, weight=1)
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

    def _add_syntax_tree(self, rel_arg, sentence_reltuples):
        full_arg_string = sentence_reltuples.arg_to_string(rel_arg)
        full_arg_string = self._clean_node(full_arg_string)
        self._add_word(rel_arg, sentence_reltuples)
        self._add_edge(
            rel_arg.lemma,
            full_arg_string,
            rel_arg.deprel,
            sentence_reltuples.sentence.getText(),
        )

    def _add_word(self, word, sentence_reltuples):
        self._add_node(word.lemma, sentence_reltuples.sentence.getText())
        parent = sentence_reltuples.sentence.words[word.head]
        self._add_edge(
            word.lemma, parent.lemma, word.deprel, sentence_reltuples.sentence.getText()
        )
        for child_idx in word.children:
            child = sentence_reltuples.sentence.words[child_idx]
            self._add_word(child, sentence_reltuples)

    def _clean_node(self, node_string):
        res = (
            "".join(
                char
                for char in node_string
                if char.isalnum() or char.isspace() or char in ",.;-—/:%"
            )
            .lower()
            .strip(" .,:;")
        )
        return res

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the UDPipe model")
    parser.add_argument(
        "conllu_dir",
        help="Path to the directory containing parsed text in conllu format",
    )
    parser.add_argument("save_dir", help="Path to the directory to save relations to")
    parser.add_argument(
        "--include-syntax",
        help="Include syntax tree of every phrase in the graph",
        action="store_true",
    )
    args = parser.parse_args()
    conllu_dir = Path(args.conllu_dir)
    save_dir = Path(args.save_dir)
    model = UDPipeModel(args.model_path)
    with open("stopwords.txt", mode="r", encoding="utf-8") as file:
        stopwords = list(file.read().split())

    graph = RelGraph(stopwords)
    for path in tqdm(conllu_dir.iterdir()):
        output = {}
        if not (path.suffix == ".conllu"):
            continue
        with path.open("r", encoding="utf8") as file:
            text = file.read()
        sentences = model.read(text, "conllu")
        for s in sentences:
            reltuples = SentenceReltuples(s)
            output[s.getText()] = reltuples.string_tuples
            graph.add_sentence_reltuples(reltuples, include_syntax=args.include_syntax)

        output_path = save_dir / (path.stem + "_reltuples.json")
        with output_path.open("w", encoding="utf8") as file:
            json.dump(output, file, ensure_ascii=False, indent=4)

    graph.save(save_dir / "graph.gexf")

