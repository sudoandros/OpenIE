import argparse
import io
import json
import string
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import gensim.downloader
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from udpipe_model import UDPipeModel


class SentenceReltuples:
    def __init__(self, sentence, additional_relations=False, stopwords=[]):
        self.sentence = sentence
        self._stopwords = set(stopwords)
        self._reltuples = self._get_reltuples(additional_relations=additional_relations)

    def string_tuples(self, lemmatize_args=False):
        return [
            self._to_string_tuple(reltuple, lemmatize_args=lemmatize_args)
            for reltuple in self._reltuples
        ]

    def _to_string_tuple(self, reltuple, lemmatize_args=False):
        left_arg = self._arg_to_string(reltuple[0], lemmatized=lemmatize_args)
        relation = self._relation_to_string(reltuple[1])
        right_arg = self._arg_to_string(reltuple[2], lemmatized=lemmatize_args)
        return (left_arg, relation, right_arg)

    def _relation_to_string(self, relation):
        if isinstance(relation, list):
            string_ = " ".join(self.sentence.words[id_].form for id_ in relation)
        elif isinstance(relation, str):
            string_ = relation
        else:
            raise TypeError
        return self._clean_string(string_)

    def _arg_to_string(self, words_ids, lemmatized=False):
        if lemmatized:
            string_ = " ".join(self.sentence.words[id_].lemma for id_ in words_ids)
        else:
            string_ = " ".join(self.sentence.words[id_].form for id_ in words_ids)
        return self._clean_string(string_)

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

    def _get_reltuples(self, additional_relations=False):
        result = []
        for word in self.sentence.words:
            if word.deprel == "cop":
                result += self._get_copula_reltuples(word)
            elif word.upostag == "VERB":
                result += self._get_verb_reltuples(word)
        if additional_relations:
            for left_arg, _, right_arg in result.copy():
                result += self._get_additional_reltuples(left_arg)
                result += self._get_additional_reltuples(right_arg)
        return [
            (left_arg, relation, right_arg)
            for left_arg, relation, right_arg in result
            if not self._is_stopwords(left_arg) and not self._is_stopwords(right_arg)
        ]

    def _get_verb_reltuples(self, verb):
        for child_id in verb.children:
            child = self.sentence.words[child_id]
            if child.deprel == "xcomp":
                return ()
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

    def _get_additional_reltuples(self, words_ids):
        result = []
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
            child = self.sentence.words[child_id]
            descendants_ids = []
            if child.deprel in upper:
                subtree = self._get_subtree(child)
                descendants_ids = [id_ for id_ in words_ids if id_ in subtree]
                result.append((descendants_ids, "выше", words_ids))
            elif child.deprel in dependent:
                subtree = self._get_subtree(child)
                descendants_ids = [id_ for id_ in words_ids if id_ in subtree]
                result.append((descendants_ids, "часть", words_ids))
            result += self._get_additional_reltuples(descendants_ids)
            main_phrase_ids = [
                id_ for id_ in main_phrase_ids if id_ not in descendants_ids
            ]
        if len(words_ids) != len(main_phrase_ids):
            result.append((words_ids, "выше", main_phrase_ids))
        if len(main_phrase_ids) > 1:
            result.append((main_phrase_ids, "выше", [root.id]))
        return result

    def _get_relation(self, word, right_arg=None):
        prefix = self._get_relation_prefix(word)
        postfix = self._get_relation_postfix(word, right_arg=right_arg)
        relation = prefix + [word.id] + postfix
        return relation

    def _get_relation_prefix(self, relation):
        prefix = []
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

    def _get_relation_postfix(self, relation, right_arg=None):
        postfix = []
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
        args_list = []
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
        subj_list = []
        for child_id in word.children:
            child = self.sentence.words[child_id]
            if self._is_subject(child):
                subj_list.append(self._get_subtree(child))
        if not subj_list and (word.deprel == "conj" or word.deprel == "xcomp"):
            parent = self.sentence.words[word.head]
            subj_list = self._get_subjects(parent)
        return subj_list

    def _get_subtree(self, word):
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

    def _get_first_case(self, words_ids):
        root = self._get_root(words_ids)
        for id_ in words_ids:
            word = self.sentence.words[id_]
            if id_ < root.id and word.deprel == "case":
                return id_
        return None

    def _get_copula(self, word):
        parent = self.sentence.words[word.head]
        part_ids = []
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

    def _get_root(self, words_ids):
        if not words_ids:
            return None
        for id_ in words_ids:
            word = self.sentence.words[id_]
            if word.head not in words_ids:
                root = word
        return root

    def _is_stopwords(self, words_ids):
        return {self.sentence.words[id_].lemma for id_ in words_ids}.issubset(
            self._stopwords
        ) or (
            len(words_ids) == 1
            and len(self.sentence.words[words_ids[0]].lemma) == 1
            and self.sentence.words[words_ids[0]].lemma.isalpha()
        )

    def _is_subject(self, word):
        return word.deprel in ("nsubj", "nsubj:pass")

    def _is_right_arg(self, word):
        return word.deprel in ("obj", "iobj", "obl", "obl:agent", "iobl")

    def _is_conjunct(self, word):
        return word.deprel == "conj"


class RelGraph:
    def __init__(self):
        self._graph = nx.DiGraph()

    @classmethod
    def from_reltuples_iter(cls, reltuples_iter):
        graph = cls()
        for sentence_reltuple in reltuples_iter:
            graph.add_sentence_reltuples(sentence_reltuple)

    @property
    def nodes_number(self):
        return self._graph.number_of_nodes()

    @property
    def edges_number(self):
        return self._graph.number_of_edges()

    def add_sentence_reltuples(self, sentence_reltuples, cluster=0):
        sentence_text = sentence_reltuples.sentence.getText()
        for (
            (left_arg, relation, right_arg),
            (left_arg_lemmatized, _, right_arg_lemmatized),
        ) in zip(
            sentence_reltuples.string_tuples(),
            sentence_reltuples.string_tuples(lemmatize_args=True),
        ):
            self._add_node(
                left_arg_lemmatized, sentence_text, label=left_arg, type_=cluster
            )
            self._add_node(
                right_arg_lemmatized, sentence_text, label=right_arg, type_=cluster
            )
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

    def _add_node(self, name, description, label=None, type_=0):
        if label is None:
            label = name
        if name not in self._graph:
            self._graph.add_node(
                name, label=label, description=description, weight=1, feat_type=type_
            )
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


class TextReltuples:
    def __init__(
        self,
        conllu,
        udpipe_model,
        w2v_model,
        stopwords,
        additional_relations,
        nodes_limit,
    ):
        sentences = udpipe_model.read(conllu, "conllu")
        labels = self._cluster(sentences, w2v_model)
        self._reltuples = []
        self._dict = {}
        self._graph = RelGraph()
        for s, lbl in zip(sentences, labels):
            sentence_reltuples = SentenceReltuples(
                s, additional_relations=additional_relations, stopwords=stopwords
            )
            self._reltuples.append(sentence_reltuples)
            self._graph.add_sentence_reltuples(sentence_reltuples, cluster=lbl)
            self._dict[
                sentence_reltuples.sentence.getText()
            ] = sentence_reltuples.string_tuples()
            if self._graph.nodes_number > nodes_limit:
                break

    @property
    def graph(self):
        return self._graph

    @property
    def dictionary(self):
        return self._dict

    def _cluster(
        self,
        sentences,
        w2v_model,
        min_cluster_size=20,
        max_cluster_size=100,
        cluster_size_step=10,
    ):
        X = np.array([self._get_sentence_vector(s, w2v_model) for s in sentences])
        max_sil_score = -1
        res_labels = None
        n_sentences = len(sentences)
        for cluster_size in range(
            min_cluster_size, max_cluster_size, cluster_size_step
        ):
            n_clusters = n_sentences // cluster_size
            kmeans = KMeans(n_clusters=n_clusters, n_jobs=1)
            kmeans.fit(X)
            score = silhouette_score(X, kmeans.labels_)
            if score >= max_sil_score:
                max_sil_score = score
                res_labels = kmeans.labels_
        return res_labels

    def _get_sentence_vector(self, sentence, w2v_model):
        vector = np.zeros(300)
        count = 0
        for word in sentence.words:
            try:
                vector = np.add(
                    vector, w2v_model["{}_{}".format(word.lemma, word.upostag)]
                )
                count += 1
            except KeyError:
                continue
        if count > 0:
            return vector / count
        else:
            return vector


def build_dir_graph(
    conllu_dir,
    save_dir,
    udpipe_model,
    stopwords,
    additional_relations,
    nodes_limit,
    w2v_model,
):
    conllu = ""

    for path in tqdm(conllu_dir.iterdir()):
        if not (path.suffix == ".conllu"):
            continue
        with path.open("r", encoding="utf8") as file:
            conllu = "{}\n{}".format(conllu, file.read())

    text_reltuples = TextReltuples(
        conllu, udpipe_model, w2v_model, stopwords, additional_relations, nodes_limit
    )

    json_path = save_dir / ("relations.json")
    with json_path.open("w", encoding="utf8") as file:
        json.dump(text_reltuples.dictionary, file, ensure_ascii=False, indent=4)

    text_reltuples.graph.save(save_dir / "graph{}.gexf".format(conllu_dir.name))
    print(text_reltuples.graph.nodes_number, text_reltuples.graph.edges_number)


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
    udpipe_model = UDPipeModel(args.model_path)
    nodes_limit = args.nodes_limit or float("inf")
    with open("stopwords.txt", mode="r", encoding="utf-8") as file:
        stopwords = list(file.read().split())
    w2v_model = gensim.downloader.load("word2vec-ruscorpora-300")

    build_dir_graph(
        conllu_dir, save_dir, udpipe_model, stopwords, args.add, nodes_limit, w2v_model
    )
