import io
import logging
import xml.etree.ElementTree as ET
from copy import deepcopy
from itertools import chain, groupby, product
from typing import Iterable, List, Sequence, Set

import networkx as nx
import networkx.algorithms.components
import numpy as np
import openie.syntax
from openie.relations.sentence import SentenceReltuples
from scipy.spatial import distance
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids

MIN_CLUSTER_SIZE = 50
NODE_DISTANCE_THRESHOLD = 0.3
SAME_NAME_NODE_DISTANCE_THRESHOLD = 0.5


class RelGraph:
    def __init__(self):
        self._graph = nx.MultiDiGraph()

    @classmethod
    def from_reltuples_iter(cls, reltuples_iter: Sequence[SentenceReltuples]):
        graph = cls()
        for sentence_reltuple in reltuples_iter:
            graph.add_sentence_reltuples(sentence_reltuple)

    @property
    def nodes_number(self):
        return self._graph.number_of_nodes()

    @property
    def edges_number(self):
        return self._graph.number_of_edges()

    def add_sentence_reltuples(
        self, sentence_reltuples: SentenceReltuples, cluster: int = 0
    ):
        sentence_text = sentence_reltuples.sentence.getText()
        for reltuple in sentence_reltuples:
            source = self._add_node(
                reltuple.left_arg_lemmas,
                sentence_text,
                label=reltuple.left_arg,
                vector=reltuple.left_w2v,
                feat_type=cluster,
            )
            target = self._add_node(
                reltuple.right_arg_lemmas,
                sentence_text,
                label=reltuple.right_arg,
                vector=reltuple.right_w2v,
                feat_type=cluster,
            )
            self._add_edge(
                source,
                target,
                reltuple.relation,
                reltuple.relation_lemmas,
                reltuple.right_deprel,
                sentence_text,
                feat_type=cluster,
            )

    def merge_relations(self):
        while True:
            same_name_nodes_to_merge_sets = set(self._find_same_name_nodes_to_merge())
            if len(same_name_nodes_to_merge_sets) > 0:
                for same_name_nodes_to_merge in same_name_nodes_to_merge_sets:
                    self._merge_nodes(same_name_nodes_to_merge)

            same_name_nodes_to_merge_sets = (
                self._find_same_name_nodes_to_merge_weak_rule()
            )
            if len(same_name_nodes_to_merge_sets) > 0:
                for same_name_nodes_to_merge in same_name_nodes_to_merge_sets:
                    self._merge_nodes(same_name_nodes_to_merge)

            nodes_to_merge = []
            edges_to_merge = []

            for source, target, key in self._graph.edges:
                targets_to_merge = self._find_nodes_to_merge(source=source, key=key)
                if len(targets_to_merge) > 1:
                    logging.info(
                        (
                            "Found {n_to_merge} right arguments to merge: \n"
                            "Shared left argument: {left_arg} \n"
                            "Shared relation: {rel} \n"
                            "Values to merge: \n"
                            "{values_to_merge}"
                        ).format(
                            n_to_merge=len(targets_to_merge),
                            left_arg=self._graph.nodes[source]["label"],
                            rel=self._graph[source][next(iter(targets_to_merge))][key][
                                "label"
                            ],
                            values_to_merge="\n".join(
                                self._graph.nodes[node]["label"]
                                for node in targets_to_merge
                            ),
                        )
                    )
                    nodes_to_merge = targets_to_merge
                    break

                sources_to_merge = self._find_nodes_to_merge(target=target, key=key)
                if len(sources_to_merge) > 1:
                    logging.info(
                        (
                            "Found {n_to_merge} left arguments to merge: \n"
                            "Shared right argument: {right_arg} \n"
                            "Shared relation: {rel} \n"
                            "Values to merge: \n"
                            "{values_to_merge}"
                        ).format(
                            n_to_merge=len(sources_to_merge),
                            right_arg=self._graph.nodes[target]["label"],
                            rel=self._graph[next(iter(sources_to_merge))][target][key][
                                "label"
                            ],
                            values_to_merge="\n".join(
                                self._graph.nodes[node]["label"]
                                for node in sources_to_merge
                            ),
                        )
                    )
                    nodes_to_merge = sources_to_merge
                    break

                edges_to_merge = self._find_edges_to_merge(source, target)
                if len(edges_to_merge) > 1:
                    logging.info(
                        (
                            "Found {n_to_merge} relations to merge: \n"
                            "Shared left argument: {left_arg} \n"
                            "Shared right argument: {right_arg} \n"
                            "Values to merge: \n"
                            "{values_to_merge}"
                        ).format(
                            n_to_merge=len(edges_to_merge),
                            left_arg=self._graph.nodes[source]["label"],
                            right_arg=self._graph.nodes[target]["label"],
                            values_to_merge="\n".join(
                                {
                                    self._graph[s][t][key]["label"]
                                    for s, t, key in edges_to_merge
                                }
                            ),
                        )
                    )
                    break

            if len(nodes_to_merge) > 1:
                self._merge_nodes(nodes_to_merge)
            elif len(edges_to_merge) > 1:
                self._merge_edges(edges_to_merge)
            else:
                break

    def filter_nodes(self, n_nodes_to_leave):
        nodes_to_remove = self._find_nodes_to_remove(n_nodes_to_leave)
        self._perform_filtering(nodes_to_remove)

    def _add_edge(
        self, source, target, label, lemmas, deprel, description, weight=1, feat_type=0
    ):
        if label in ["_is_a_", "_relates_to_"]:
            key = label
        else:
            key = "{} + {}".format(lemmas, deprel)
        description = _to_set_if_not_already(description)
        feat_type = _to_set_if_not_already(feat_type)
        if not self._graph.has_edge(source, target, key=key):
            # it's a new edge
            color = {"b": 0, "g": 0, "r": 0}
            if label == "_is_a_":
                color = {"b": 255, "g": 0, "r": 0}
            elif label == "_relates_to_":
                color = {"b": 0, "g": 255, "r": 0}
            self._graph.add_edge(
                source,
                target,
                key=key,
                label=label,
                lemmas=lemmas,
                deprel=deprel,
                description=description,
                weight=weight,
                feat_type=feat_type,
                viz={"color": color},
            )
        else:
            # this edge already exists
            self._graph[source][target][key]["description"] = (
                description | self._graph[source][target][key]["description"]
            )
            self._graph[source][target][key]["feat_type"] = (
                feat_type | self._graph[source][target][key]["feat_type"]
            )
            self._graph[source][target][key]["weight"] += weight

        self._inherit_relation(source, target, key)
        return key

    def _inherit_relation(self, source, target, key):
        if key == "_is_a_":  # it's a "is a" relation
            # inherit all verb relations from up the source to the target
            edges = list(self._graph.in_edges(source, keys=True)) + list(
                self._graph.out_edges(source, keys=True),
            )
            for s, t, key in edges:
                if key in ["_is_a_", "_relates_to_"]:
                    continue
                self._inherit_relation(s, t, key)
        elif key != "_relates_to_":  # it's a verb relation
            # inherit this relation down the "is a" relations for the source
            successors = [
                node
                for node in self._graph.successors(source)
                if self._graph.has_edge(source, node, key="_is_a_")
            ]
            for node in successors:
                self._add_edge(
                    node,
                    target,
                    self._graph[source][target][key]["label"],
                    self._graph[source][target][key]["lemmas"],
                    self._graph[source][target][key]["deprel"],
                    self._graph[source][target][key]["description"],
                    weight=self._graph[source][target][key]["weight"],
                    feat_type=self._graph[source][target][key]["feat_type"],
                )

            # inherit this relation down the "is a" relations for the target
            successors = [
                node
                for node in self._graph.successors(target)
                if self._graph.has_edge(target, node, key="_is_a_")
            ]
            for node in successors:
                self._add_edge(
                    source,
                    node,
                    self._graph[source][target][key]["label"],
                    self._graph[source][target][key]["lemmas"],
                    self._graph[source][target][key]["deprel"],
                    self._graph[source][target][key]["description"],
                    weight=self._graph[source][target][key]["weight"],
                    feat_type=self._graph[source][target][key]["feat_type"],
                )

    def _add_node(self, lemmas, description, label, weight=1, vector=None, feat_type=0):
        description = _to_set_if_not_already(description)
        feat_type = _to_set_if_not_already(feat_type)
        node = "{} + {}".format(lemmas, str(feat_type))
        if node not in self._graph:
            self._graph.add_node(
                node,
                lemmas=lemmas,
                label=label,
                description=description,
                weight=weight,
                vector=vector,
                feat_type=feat_type,
            )
        else:
            # this node already exists
            self._graph.nodes[node]["label"] = " | ".join(
                set(self._graph.nodes[node]["label"].split(" | ") + label.split(" | "))
            )
            self._graph.nodes[node]["description"] = (
                description | self._graph.nodes[node]["description"]
            )
            self._graph.nodes[node]["feat_type"] = (
                feat_type | self._graph.nodes[node]["feat_type"]
            )
            self._graph.nodes[node]["vector"] = (
                self._graph.nodes[node]["vector"] + vector
            ) / 2
            self._graph.nodes[node]["weight"] += weight
        return node

    def _find_target_merge_candidates(self, source, key):
        return {
            target
            for target in self._graph.successors(source)
            if self._graph.has_edge(source, target, key=key)
            and (
                self._graph[source][target][key]["label"]
                not in ["_is_a_", "_relates_to_"]
                or self._graph.nodes[source]["label"]
                == self._graph.nodes[target]["label"]
            )
        }

    def _find_source_merge_candidates(self, target, key):
        return {
            source
            for source in self._graph.predecessors(target)
            if self._graph.has_edge(source, target, key=key)
            and (
                self._graph[source][target][key]["label"]
                not in ["_is_a_", "_relates_to_"]
                or self._graph.nodes[source]["label"]
                == self._graph.nodes[target]["label"]
            )
        }

    def _filter_node_merge_candidates(self, nodes: Set[str]):
        res = nodes.copy()
        for node1 in nodes:
            for node2 in nodes:
                if node1 != node2 and (
                    self._graph.has_edge(node1, node2)
                    or (
                        self._graph.nodes[node1]["description"]
                        & self._graph.nodes[node2]["description"]
                    )
                ):
                    res.discard(node1)
                    res.discard(node2)
        return res

    def _find_nodes_inside_radius(
        self, nodes: Iterable[str], radius: float
    ) -> Set[str]:
        nodes = sorted(
            nodes,
            key=lambda node: (self._graph.nodes[node]["weight"], node),
            reverse=True,
        )
        for central_node in nodes:
            group = {central_node}
            for node in nodes:
                if (
                    self._nodes_distance(central_node, node) <= radius
                    and node != central_node
                ):
                    group.add(node)
            if len(group) > 1:
                return group
        if len(nodes) > 0:
            return {nodes[0]}
        else:
            return set()

    def _nodes_distance(self, node1, node2):
        vector1: np.ndarray = self._graph.nodes[node1]["vector"]
        vector2: np.ndarray = self._graph.nodes[node2]["vector"]
        if not vector1.any() or not vector2.any():
            return float("inf")
        else:
            return float(distance.cosine(vector1, vector2))

    def _find_nodes_to_merge(self, source=None, target=None, key=None):
        if source is not None and key is not None:
            res = self._find_target_merge_candidates(source, key)
        elif target is not None and key is not None:
            res = self._find_source_merge_candidates(target, key)
        else:
            raise ValueError("Wrong set of specified arguments")

        if len(res) < 2:
            return res

        res = self._filter_node_merge_candidates(res)
        res = self._find_nodes_inside_radius(res, NODE_DISTANCE_THRESHOLD)
        return res

    def _find_edges_to_merge(self, source, target):
        keys = [
            (key, cluster, attr["label"])
            for _, _, key, attr in self._graph.out_edges(source, keys=True, data=True)
            if self._graph.has_edge(source, target, key=key)
            and attr["label"] not in ["_is_a_", "_relates_to_"]
            for cluster in attr["feat_type"]
        ]

        keys.sort(key=lambda elem: elem[1:])
        skip_cluster = False
        for cluster_name, cluster_group in groupby(keys, key=lambda elem: elem[1]):
            cluster_group_list = list(cluster_group)
            if len(cluster_group_list) == 1:
                continue
            for _, label_group in groupby(cluster_group_list, key=lambda elem: elem[2]):
                if len(list(label_group)) > 1:
                    skip_cluster = True
                    break
            if skip_cluster:
                skip_cluster = False
                continue
            else:
                keys = set(key for key, *_ in cluster_group_list)
                cluster = cluster_name
                break
        else:  # all clusters have been skipped
            return set()

        edges = set()
        for s, t, key, feat_type in self._graph.edges(keys=True, data="feat_type"):
            if key in keys and cluster in feat_type:
                edges.add((s, t, key))

        # relations from the same sentence are out
        for s1, t1, key1 in edges.copy():
            for s2, t2, key2 in edges.copy():
                if (s1, t1, key1) != (s2, t2, key2) and (
                    self._graph.edges[s1, t1, key1]["description"]
                    & self._graph.edges[s2, t2, key2]["description"]
                ):
                    edges.discard((s1, t1, key1))
                    edges.discard((s2, t2, key2))
        return edges

    def _find_same_name_nodes_to_merge(self):
        labels_edges_dict = {}
        for s, t, k in self._graph.edges:
            if self._graph[s][t][k]["label"] in ["_is_a_", "_relates_to_"]:
                continue
            source_labels = self._graph.nodes[s]["label"].split(" | ")
            edge_labels = self._graph[s][t][k]["label"].split(" | ")
            target_labels = self._graph.nodes[t]["label"].split(" | ")
            for labels in product(source_labels, edge_labels, target_labels):
                if labels not in labels_edges_dict:
                    labels_edges_dict[labels] = [(s, t, k)]
                else:
                    labels_edges_dict[labels].append((s, t, k))

        res = set()
        seen = set()
        for edges in labels_edges_dict.values():
            if len(edges) < 2:
                continue
            sources_to_merge = frozenset(s for s, _, _ in edges if s not in seen)
            if len(sources_to_merge) > 1:
                res.add(sources_to_merge)
                logging.info(
                    f"Found {len(sources_to_merge)} same name sources to merge:\n"
                    + "\n".join(
                        self._graph.nodes[node]["label"] for node in sources_to_merge
                    )
                )
                seen.update(sources_to_merge)
            targets_to_merge = frozenset(t for _, t, _ in edges if t not in seen)
            if len(targets_to_merge) > 1:
                logging.info(
                    f"Found {len(targets_to_merge)} same name targets to merge:\n"
                    + "\n".join(
                        self._graph.nodes[node]["label"] for node in targets_to_merge
                    )
                )
                res.add(targets_to_merge)
                seen.update(targets_to_merge)
        return res

    def _find_same_name_nodes_to_merge_weak_rule(self):
        # TODO merge with `_find_same_name_nodes_to_merge` into a single function
        return (
            self._find_same_name_sources_to_merge_weak_rule()
            | self._find_same_name_targets_to_merge_weak_rule()
        )

    def _find_same_name_sources_to_merge_weak_rule(self):
        # TODO merge with _find_same_name_targets_to_merge_weak_rule into a single function
        labels_edges_dict = {}
        for s, t, k in self._graph.edges:
            if self._graph[s][t][k]["label"] in ["_is_a_", "_relates_to_"]:
                continue
            source_labels = self._graph.nodes[s]["label"].split(" | ")
            edge_labels = self._graph[s][t][k]["label"].split(" | ")
            for labels in product(source_labels, edge_labels):
                if labels not in labels_edges_dict:
                    labels_edges_dict[labels] = {(s, t, k)}
                else:
                    labels_edges_dict[labels].add((s, t, k))

        res = set()
        seen = set()
        for edges in labels_edges_dict.values():
            if len(edges) < 2:
                continue
            targets = {t for _, t, _ in edges}
            targets = self._filter_node_merge_candidates(targets)
            targets = self._find_nodes_inside_radius(
                targets, SAME_NAME_NODE_DISTANCE_THRESHOLD
            )
            sources_to_merge = frozenset(
                s for s, t, _ in edges if t in targets and s not in seen
            )
            if len(sources_to_merge) < 2:
                continue
            logging.info(
                "Found same name sources to merge (weak rule):\n"
                + "\n".join(
                    self._graph.nodes[node]["label"] for node in sources_to_merge
                )
                + "\n"
                + "because of the next similar targets:\n"
                + "\n".join(self._graph.nodes[node]["label"] for node in targets)
            )
            res.add(sources_to_merge)
            seen.update(sources_to_merge)
        return res

    def _find_same_name_targets_to_merge_weak_rule(self):
        labels_edges_dict = {}
        for s, t, k in self._graph.edges:
            if self._graph[s][t][k]["label"] in ["_is_a_", "_relates_to_"]:
                continue
            edge_labels = self._graph[s][t][k]["label"].split(" | ")
            target_labels = self._graph.nodes[t]["label"].split(" | ")
            for labels in product(edge_labels, target_labels):
                if labels not in labels_edges_dict:
                    labels_edges_dict[labels] = {(s, t, k)}
                else:
                    labels_edges_dict[labels].add((s, t, k))

        res = set()
        seen = set()
        for edges in labels_edges_dict.values():
            if len(edges) < 2:
                continue
            sources = {s for s, _, _ in edges}
            sources = self._filter_node_merge_candidates(sources)
            sources = self._find_nodes_inside_radius(
                sources, SAME_NAME_NODE_DISTANCE_THRESHOLD
            )
            targets_to_merge = frozenset(
                t for s, t, _ in edges if s in sources and t not in seen
            )
            if len(targets_to_merge) < 2:
                continue
            logging.info(
                "Found same name sources to merge (weak rule):\n"
                + "\n".join(
                    self._graph.nodes[node]["label"] for node in targets_to_merge
                )
                + "\n"
                + "because of the next similar targets:\n"
                + "\n".join(self._graph.nodes[node]["label"] for node in sources)
            )
            res.add(targets_to_merge)
            seen.update(targets_to_merge)
        return res

    def _merge_nodes(self, nodes):
        def new_set_attr_value(attr_key):
            res = set()
            for node in nodes:
                res |= _to_set_if_not_already(self._graph.nodes[node][attr_key])
            return res

        def new_str_attr_value(attr_key):
            res = set()
            for node in nodes:
                res |= set(self._graph.nodes[node][attr_key].split(" | "))
            return " | ".join(res)

        new_lemmas = new_str_attr_value("lemmas")
        new_label = new_str_attr_value("label")
        new_description = new_set_attr_value("description")
        new_feat_type = new_set_attr_value("feat_type")
        new_weight = sum(self._graph.nodes[node]["weight"] for node in nodes)
        new_vector = sum(self._graph.nodes[node]["vector"] for node in nodes) / len(
            nodes
        )
        new_node = self._add_node(
            new_lemmas,
            new_description,
            new_label,
            new_weight,
            new_vector,
            new_feat_type,
        )

        for source, target, key in self._graph.edges(nodes, keys=True):
            new_source = None
            new_target = None
            if source in nodes:  # "out" edge
                new_source, new_target = (new_node, target)
            elif target in nodes:  # "in" edge
                new_source, new_target = (source, new_node)
            self._add_edge(
                new_source,
                new_target,
                self._graph.edges[source, target, key]["label"],
                self._graph.edges[source, target, key]["lemmas"],
                self._graph.edges[source, target, key]["deprel"],
                self._graph.edges[source, target, key]["description"],
                weight=self._graph.edges[source, target, key]["weight"],
                feat_type=new_feat_type,
            )

        for node in nodes:
            self._graph.remove_node(node)

    def _merge_edges(self, edges):
        def new_str_attr_value(attr_key):
            res = set()
            for source, target, key in edges:
                res |= set(self._graph[source][target][key][attr_key].split(" | "))
            return " | ".join(res)

        new_label = new_str_attr_value("label")
        new_lemmas = new_str_attr_value("lemmas")
        new_deprel = new_str_attr_value("deprel")
        new_weight = sum(
            (
                self._graph[source][target][key]["weight"]
                for source, target, key in edges
            )
        )
        for source, target, key in edges:
            self._add_edge(
                source,
                target,
                new_label,
                new_lemmas,
                new_deprel,
                self._graph[source][target][key]["description"],
                weight=new_weight,
                feat_type=self._graph[source][target][key]["feat_type"],
            )
            self._graph.remove_edge(source, target, key=key)

    def save(self, path):
        self._transform()
        for node in self._graph:
            if self._graph.nodes[node].get("vector") is not None:
                self._graph.nodes[node]["vector"] = str(
                    self._graph.nodes[node]["vector"].tolist()
                )
            self._graph.nodes[node]["description"] = " | ".join(
                self._graph.nodes[node]["description"]
            )
            self._graph.nodes[node]["feat_type"] = " | ".join(
                str(elem) for elem in self._graph.nodes[node]["feat_type"]
            )
        stream_buffer = io.BytesIO()
        nx.write_gexf(self._graph, stream_buffer, encoding="utf-8", version="1.1draft")
        xml_string = stream_buffer.getvalue().decode("utf-8")
        root_element = ET.fromstring(xml_string)
        self._fix_gexf(root_element)
        ET.register_namespace("", "http://www.gexf.net/1.1draft")
        xml_tree = ET.ElementTree(root_element)
        xml_tree.write(path, encoding="utf-8")

    def _find_nodes_to_remove(self, n_nodes_to_leave):
        # include only nodes with verb relations
        nodes_to_leave = set()
        for node in self._graph.nodes:
            if any(
                key not in ["_is_a_", "_relates_to_"]  # is a verb relation
                for _, _, key in chain(
                    self._graph.in_edges(node, keys=True),
                    self._graph.out_edges(node, keys=True),
                )
            ):
                nodes_to_leave.add(node)
        if len(nodes_to_leave) <= n_nodes_to_leave:
            return set(self._graph.nodes) - nodes_to_leave

        # include only the most weighted nodes
        nodes_to_leave = set(
            sorted(
                nodes_to_leave,
                key=lambda node: self._graph.nodes[node]["weight"],
                reverse=True,
            )[:n_nodes_to_leave]
        )

        return set(self._graph.nodes) - nodes_to_leave

    def _perform_filtering(self, nodes_to_remove):
        nodes_to_remove = set(nodes_to_remove)
        for node in nodes_to_remove:
            in_edges = list(self._graph.in_edges(node, keys=True))
            out_edges = list(self._graph.out_edges(node, keys=True))
            # if removing B: A->B->C  ==>  A->C
            for pred, _, key_pred in in_edges:
                for _, succ, key_succ in out_edges:
                    if key_pred != key_succ:
                        continue
                    # FIXME wrong attrs in the new edge?
                    self._add_edge(
                        pred,
                        succ,
                        self._graph[node][succ][key_succ]["label"],
                        self._graph[node][succ][key_succ]["lemmas"],
                        self._graph[node][succ][key_succ]["deprel"],
                        self._graph[node][succ][key_succ]["description"],
                        weight=self._graph[node][succ][key_succ]["weight"],
                        feat_type=self._graph[node][succ][key_succ]["feat_type"],
                    )
            self._graph.remove_node(node)

    def _transform(self):
        # transform relations from edges to nodes with specific node_type and color
        for node in self._graph:
            self._graph.nodes[node]["node_type"] = "argument"
        for source, target, key, attr in list(self._graph.edges(data=True, keys=True)):
            node = "{}({}; {})".format(
                self._graph.edges[source, target, key]["label"], source, target
            )
            new_attr = deepcopy(attr)
            if self._graph.edges[source, target, key]["label"] == "_is_a_":
                new_attr["viz"] = {"color": {"b": 160, "g": 160, "r": 255}}
            elif self._graph.edges[source, target, key]["label"] == "_relates_to_":
                new_attr["viz"] = {"color": {"b": 160, "g": 255, "r": 160}}
            else:
                new_attr["viz"] = {"color": {"b": 255, "g": 0, "r": 0}}
            new_attr["node_type"] = "relation"
            new_attr["weight"] = min(
                self._graph.nodes[source]["weight"], self._graph.nodes[target]["weight"]
            )
            self._graph.add_node(node, **new_attr)
            self._graph.add_edge(source, node)
            self._graph.add_edge(node, target)
            self._graph.remove_edge(source, target, key=key)

    @staticmethod
    def _fix_gexf(root_element):
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
        self, conllu, w2v_model, stopwords, additional_relations, entities_limit,
    ):
        sentences = openie.syntax.read_parsed(conllu, "conllu")
        self._reltuples: Sequence[SentenceReltuples] = []
        self._dict = {}
        self._graph = RelGraph()
        for s in sentences:
            sentence_reltuples = SentenceReltuples(
                s,
                w2v_model,
                additional_relations=additional_relations,
                stopwords=stopwords,
            )
            self._reltuples.append(sentence_reltuples)
        cluster_labels = self._cluster(
            min_cluster_size=MIN_CLUSTER_SIZE, max_cluster_size=MIN_CLUSTER_SIZE + 50,
        )
        for sentence_reltuples, cluster in zip(self._reltuples, cluster_labels):
            self._graph.add_sentence_reltuples(sentence_reltuples, cluster=cluster)
            self._dict[sentence_reltuples.sentence.getText()] = [
                (reltuple.left_arg, reltuple.relation, reltuple.right_arg)
                for reltuple in sentence_reltuples
            ]
        self._graph.merge_relations()
        self._graph.filter_nodes(entities_limit)

    @property
    def graph(self):
        return self._graph

    @property
    def dictionary(self):
        return self._dict

    # TODO iterate over reltuples by __iter__?

    def _cluster(
        self, min_cluster_size=10, max_cluster_size=100, cluster_size_step=10
    ) -> List[int]:
        X = np.array(
            [
                sentence_reltuples.sentence_vector
                for sentence_reltuples in self._reltuples
            ]
        )
        max_sil_score = -1
        n_sentences = len(self._reltuples)
        res_labels = np.zeros(n_sentences)
        for cluster_size in range(
            min_cluster_size, max_cluster_size, cluster_size_step
        ):
            n_clusters = n_sentences // cluster_size
            if n_clusters < 2:
                continue
            clusterer = KMedoids(
                n_clusters=n_clusters, init="k-medoids++", metric="cosine"
            )
            clusterer.fit(X)
            score = silhouette_score(X, clusterer.labels_)
            if score >= max_sil_score:
                max_sil_score = score
                res_labels = clusterer.labels_
        return res_labels.tolist()


def _to_set_if_not_already(attr_key):
    if isinstance(attr_key, (str, int, float)):
        return {attr_key}
    else:
        return set(attr_key)

