import argparse
import io
import json
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
        left = self._subtree_to_string(reltuple[0])
        center = self._relation_to_string(reltuple[1])
        right = self._subtree_to_string(reltuple[2])
        return (left, center, right)

    @property
    def graph(self):
        graph = nx.DiGraph()
        for reltuple in self.reltuples:
            graph.add_node(reltuple[0].form, node_type="arg")
            graph.add_node(reltuple[1].form, node_type="arg")
            graph.add_node(reltuple[2].form, node_type="arg")
            graph.add_edge(reltuple[0].form, reltuple[1].form, dependency="relation")
            graph.add_edge(reltuple[1].form, reltuple[2].form, dependency="relation")
            graph_left = self._subtree_to_graph(reltuple[0])
            graph_center = self._relation_to_graph(reltuple[1])
            graph_right = self._subtree_to_graph(reltuple[2])
            for node, attr in graph_left.nodes.items():
                attr["node_type"] = "arg"
                graph.add_node(node, **attr)
            for node, attr in graph_center.nodes.items():
                attr["node_type"] = "rel"
                graph.add_node(node, **attr)
            for node, attr in graph_right.nodes.items():
                attr["node_type"] = "arg"
                graph.add_node(node, **attr)
            for edge, attr in graph_left.edges.items():
                graph.add_edge(edge[0], edge[1], **attr)
            for edge, attr in graph_center.edges.items():
                graph.add_edge(edge[0], edge[1], **attr)
            for edge, attr in graph_right.edges.items():
                graph.add_edge(edge[0], edge[1], **attr)
            for node in graph.nodes:
                graph.nodes[node]["weight"] = len(graph[node]) + 1
        return graph

    def _subtree_to_graph(self, word):
        graph = nx.DiGraph()
        for child_idx in word.children:
            child = self.sentence.words[child_idx]
            graph.add_node(word.form)
            graph.add_node(child.form)
            graph.add_edge(word.form, child.form, dependency="syntax")
            child_graph = self._subtree_to_graph(child)
            for node, attr in child_graph.nodes.items():
                graph.add_node(node, **attr)
            for edge, attr in child_graph.edges.items():
                graph.add_edge(edge[0], edge[1], **attr)
        return graph

    def _relation_to_graph(self, word):
        graph = nx.DiGraph()
        for child_idx in word.children:
            child = self.sentence.words[child_idx]
            if (
                child.deprel == "case"
                or child.deprel == "aux"
                or child.deprel == "aux:pass"
                or child.upostag == "PART"
            ):
                graph.add_node(word.form)
                graph.add_node(child.form)
                graph.add_edge(word.form, child.form, dependency="syntax")
        parent = self.sentence.words[word.head]
        if word.deprel == "xcomp":
            graph.add_node(parent.form)
            graph.add_node(word.form)
            graph.add_edge(parent.form, word.form, dependency="syntax")
            graph_parent = self._relation_to_graph(parent)
            for node, attr in graph_parent.nodes.items():
                graph.add_node(node, **attr)
            for edge, attr in graph_parent.edges.items():
                graph.add_edge(edge[0], edge[1], **attr)
        if self._is_conjunct(word) and parent.deprel == "xcomp":
            grandparent = self.sentence.words[parent.head]
            graph.add_node(grandparent.form)
            graph.add_node(word.form)
            graph.add_edge(grandparent.form, word.form, dependency="syntax")
            graph_grandparent = self._relation_to_graph(grandparent)
            for node, attr in graph_grandparent.nodes.items():
                graph.add_node(node, **attr)
            for edge, attr in graph_grandparent.edges.items():
                graph.add_edge(edge[0], edge[1], **attr)
        return graph

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

    def _relation_to_string(self, word):
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
            prefix = self._relation_to_string(parent) + " " + prefix
        if self._is_conjunct(word) and parent.deprel == "xcomp":
            grandparent = self.sentence.words[parent.head]
            prefix = self._relation_to_string(grandparent) + " " + prefix
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

    def _subtree_to_string(self, word):
        strings = []
        if not list(word.children):
            return word.form
        for child_idx in (idx for idx in word.children if idx < word.id):
            child = self.sentence.words[child_idx]
            strings.append(self._subtree_to_string(child))
        strings.append(word.form)
        for child_idx in (idx for idx in word.children if idx > word.id):
            child = self.sentence.words[child_idx]
            strings.append(self._subtree_to_string(child))
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


def save_graph(graph, path):
    stream_buffer = io.BytesIO()
    nx.write_gexf(graph, stream_buffer, encoding="utf-8", version="1.1draft")
    xml_string = stream_buffer.getvalue().decode("utf-8")
    root_element = ET.fromstring(xml_string)
    fix_gexf(root_element)
    ET.register_namespace("", "http://www.gexf.net/1.1draft")
    xml_tree = ET.ElementTree(root_element)
    xml_tree.write(path, encoding="utf-8")


def fix_gexf(root_element):
    graph_node = root_element.find("{http://www.gexf.net/1.1draft}graph")
    attributes_nodes = graph_node.findall("{http://www.gexf.net/1.1draft}attributes")
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
                for_value = attvalue_node.get("for")
                attvalue_node.set("for", node_attributes[for_value])
    edges_node = graph_node.find("{http://www.gexf.net/1.1draft}edges")
    for edge_node in edges_node.findall("{http://www.gexf.net/1.1draft}edge"):
        attvalues_node = edge_node.find("{http://www.gexf.net/1.1draft}attvalues")
        if attvalues_node is not None:
            for attvalue_node in attvalues_node.findall(
                "{http://www.gexf.net/1.1draft}attvalue"
            ):
                for_value = attvalue_node.get("for")
                attvalue_node.set("for", edge_attributes[for_value])


def simple_test(model):
    sentences = model.tokenize(
        "Андрей пошел в магазин и аптеку, купил куртку и телефон, на улице становилось темно. "
        "Никита определенно не будет бегать в парке, а Андрей, Дима и Федор варили и жарили обед. "
        "Андрей, пока собирался на работу, съел завтрак. "
        "С помощью уязвимости злоумышленник может авторизоваться в уязвимой системе и начать выполнять произвольные команды суперпользователя и творить фигню. "
    )
    graph = nx.DiGraph()
    for s in sentences:
        model.tag(s)
        model.parse(s)
        reltuples = SentenceReltuples(s)
        print(s.getText())
        print("\n".join(str(reltuple) for reltuple in reltuples.string_tuples))
        for edge in graph_sentence.edges:
            graph.add_edge(edge[0], edge[1], **graph_sentence.get_edge_data(*edge))
    conllu = model.write(sentences, "conllu")
    with open("data/test_output/output.conllu", "w", encoding="utf8") as file:
        file.write(conllu)
    nx.write_gexf(graph, "data/test_output/graph.gexf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the UDPipe model")
    parser.add_argument(
        "conllu_dir",
        help="Path to the directory containing parsed text in conllu format",
    )
    parser.add_argument("save_dir", help="Path to the directory to save relations to")
    args = parser.parse_args()
    conllu_dir = Path(args.conllu_dir)
    save_dir = Path(args.save_dir)
    model = UDPipeModel(args.model_path)
    graph = nx.DiGraph()

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
            for node, attr in graph_sentence.nodes.items():
                graph.add_node(node, **attr)
            for edge, attr in graph_sentence.edges.items():
                graph.add_edge(edge[0], edge[1], **attr)

        output_path = save_dir / (path.stem + "_reltuples.json")
        with output_path.open("w", encoding="utf8") as file:
            json.dump(output, file, ensure_ascii=False, indent=4)

    save_graph(graph, save_dir / "graph.gexf")
