import io
from spacy.tokens import Doc
import logging
import xml.etree.ElementTree as ET
from copy import deepcopy
from itertools import chain, combinations, groupby, product, repeat
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

import networkx as nx
import networkx.algorithms.components
import numpy as np
from numba import njit
from openie.relations.sentence import SentenceReltuples
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids

T = TypeVar("T", int, str, float)
MIN_CLUSTER_SIZE = 50
NODE_DISTANCE_THRESHOLD = 0.3
SAME_NAME_NODE_DISTANCE_THRESHOLD = 0.5


class RelGraph:
    def __init__(self):
        self._graph = nx.MultiDiGraph()

    @classmethod
    def from_reltuples(
        cls, reltuples: Iterable[SentenceReltuples], clusters=None, entities_limit=None
    ):
        graph = cls()
        if clusters is None:
            clusters = repeat(0)
        for sentence_reltuple, cluster in zip(reltuples, clusters):
            graph._add_sentence_reltuples(sentence_reltuple, cluster)
        graph.merge_relations()
        if entities_limit is not None:
            graph.filter_nodes(entities_limit)
        return graph

    @property
    def nodes_number(self):
        return self._graph.number_of_nodes()

    @property
    def edges_number(self):
        return self._graph.number_of_edges()

    def _add_sentence_reltuples(
        self, sentence_reltuples: SentenceReltuples, cluster: int
    ):
        sentence_text = str(sentence_reltuples.sentence)
        for reltuple in sentence_reltuples:
            source = self._add_node(
                reltuple.left_arg.lemmas,
                [sentence_text],
                label=[reltuple.left_arg.phrase],
                vector=reltuple.left_arg.vector,
                cluster=[cluster],
            )
            target = self._add_node(
                reltuple.right_arg.lemmas,
                [sentence_text],
                label=[reltuple.right_arg.phrase],
                vector=reltuple.right_arg.vector,
                cluster=[cluster],
            )
            self._add_edge(
                source,
                target,
                reltuple.relation.phrase,
                reltuple.relation.lemmas,
                reltuple.right_arg.deprel,
                [sentence_text],
                cluster=[cluster],
            )

    def merge_relations(self):
        while True:
            self._add_implicit_is_a_relations()
            same_name_nodes_to_merge_sets = self._find_same_name_nodes_to_merge()
            for same_name_nodes_to_merge in same_name_nodes_to_merge_sets:
                self._merge_nodes(same_name_nodes_to_merge)

            nodes_to_merge = []
            edges_to_merge = []

            for source, target, key in self._graph.edges:
                targets_to_merge = self._find_same_neighbor_nodes_to_merge(
                    source=source, key=key
                )
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
                                str(self._graph.nodes[node]["label"])
                                for node in targets_to_merge
                            ),
                        )
                    )
                    nodes_to_merge = targets_to_merge
                    break

                sources_to_merge = self._find_same_neighbor_nodes_to_merge(
                    target=target, key=key
                )
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
                                str(self._graph.nodes[node]["label"])
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
        self,
        source: str,
        target: str,
        label: str,
        lemmas: str,
        deprel: str,
        description: Iterable[str],
        weight: int = 1,
        cluster: Iterable[int] = (0,),
        inherit: bool = True,
    ):
        if label in ["_is_a_", "_relates_to_"]:
            key = label
        else:
            key = f"{lemmas} + {deprel}"
        description = set(description)
        cluster = set(cluster)
        if not self._graph.has_edge(source, target, key=key):  # it's a new edge
            self._graph.add_edge(
                source,
                target,
                key=key,
                label=label,
                lemmas=lemmas,
                deprel=deprel,
                description=description,
                weight=weight,
                cluster=cluster,
            )
        else:  # this edge already exists
            self._graph[source][target][key]["description"] = (
                description | self._graph[source][target][key]["description"]
            )
            self._graph[source][target][key]["cluster"] = (
                cluster | self._graph[source][target][key]["cluster"]
            )
            self._graph[source][target][key]["weight"] += weight

        if inherit:
            self._inherit_relation(source, target, key)
        return key

    def _inherit_relation(self, source: str, target: str, key: str):
        if key == "_is_a_":
            # inherit all verb relations from up the source to the target
            for s, t, k in chain(
                self._graph.in_edges(source, keys=True),
                self._graph.out_edges(source, keys=True),
            ):
                if k in ["_is_a_", "_relates_to_"]:
                    continue
                self._inherit_relation(s, t, k)
        elif key != "_relates_to_":  # it's a verb relation
            # inherit this relation down the "is a" relations for the source
            for node in self._all_successors(source, by_relations=["_is_a_"]):
                self._add_edge(
                    node,
                    target,
                    self._graph[source][target][key]["label"],
                    self._graph[source][target][key]["lemmas"],
                    self._graph[source][target][key]["deprel"],
                    self._graph[source][target][key]["description"],
                    weight=self._graph[source][target][key]["weight"],
                    cluster=self._graph[source][target][key]["cluster"],
                    inherit=False,
                )

            # inherit this relation down the "is a" relations for the target
            for node in self._all_successors(target, by_relations=["_is_a_"]):
                self._add_edge(
                    source,
                    node,
                    self._graph[source][target][key]["label"],
                    self._graph[source][target][key]["lemmas"],
                    self._graph[source][target][key]["deprel"],
                    self._graph[source][target][key]["description"],
                    weight=self._graph[source][target][key]["weight"],
                    cluster=self._graph[source][target][key]["cluster"],
                    inherit=False,
                )

    def _add_node(
        self,
        lemmas: str,
        description: Iterable[str],
        label: Iterable[str],
        weight: int = 1,
        vector: Optional[np.ndarray] = None,
        cluster: Iterable[int] = (0,),
    ):
        description = set(description)
        cluster = set(cluster)
        label = set(label)
        node = f"{lemmas} + {cluster}"
        if node not in self._graph:
            self._graph.add_node(
                node,
                lemmas=lemmas,
                label=label,
                description=description,
                weight=weight,
                vector=vector,
                cluster=cluster,
            )
        else:
            # this node already exists
            self._graph.nodes[node]["label"] = label | self._graph.nodes[node]["label"]
            self._graph.nodes[node]["description"] = (
                description | self._graph.nodes[node]["description"]
            )
            self._graph.nodes[node]["cluster"] = (
                cluster | self._graph.nodes[node]["cluster"]
            )
            self._graph.nodes[node]["vector"] = (
                self._graph.nodes[node]["vector"] + vector
            ) / 2
            self._graph.nodes[node]["weight"] += weight
        return node

    def _find_neighbor_targets_to_merge(self, source, key):
        return {
            target
            for _, target, k, label in self._graph.out_edges(
                source, keys=True, data="label"
            )
            if k == key
            and (
                label not in ["_is_a_", "_relates_to_"]
                or self._graph.nodes[source]["label"]
                == self._graph.nodes[target]["label"]
            )
        }

    def _find_neighbor_sources_to_merge(self, target, key):
        return {
            source
            for source, _, k, label in self._graph.in_edges(
                target, keys=True, data="label"
            )
            if k == key
            and (
                label not in ["_is_a_", "_relates_to_"]
                or self._graph.nodes[source]["label"]
                == self._graph.nodes[target]["label"]
            )
        }

    def _filter_node_merge_candidates(self, nodes: Set[str]):
        to_remove: List[str] = []
        for node1, node2 in combinations(nodes, 2):
            if (
                self._graph.has_edge(node1, node2)
                or self._graph.has_edge(node2, node1)
                or (
                    self._graph.nodes[node1]["description"]
                    & self._graph.nodes[node2]["description"]
                )
            ):
                to_remove.append(node1)
                to_remove.append(node2)
        return nodes - set(to_remove)

    def _find_close_nodes(
        self, nodes: Iterable[str], distance_threshold: float
    ) -> Set[str]:
        nodes = sorted(
            nodes,
            key=lambda node: (self._graph.nodes[node]["weight"], node),
            reverse=True,
        )
        for central_node in nodes:
            group = {
                node
                for node in nodes
                if self._nodes_distance(central_node, node) <= distance_threshold
                or self._graph.nodes[central_node]["label"]
                & self._graph.nodes[node]["label"]
            }  # same labels nodes are considered close too
            if len(group) > 1:
                return group
        if len(nodes) > 0:
            return {nodes[0]}
        else:
            return set()

    def _nodes_distance(self, node1: str, node2: str):
        vector1 = self._graph.nodes[node1]["vector"]
        vector2 = self._graph.nodes[node2]["vector"]
        return cosine_distance(vector1, vector2)

    def _find_same_neighbor_nodes_to_merge(self, source=None, target=None, key=None):
        if source is not None and key is not None:
            res = self._find_neighbor_targets_to_merge(source, key)
        elif target is not None and key is not None:
            res = self._find_neighbor_sources_to_merge(target, key)
        else:
            raise ValueError("Wrong set of specified arguments")

        if len(res) < 2:
            return res

        res = self._filter_node_merge_candidates(res)
        res = self._find_close_nodes(res, NODE_DISTANCE_THRESHOLD)
        return res

    def _find_edges_to_merge(self, source, target):
        keys = [
            (key, cluster, attr["label"])
            for _, _, key, attr in self._graph.out_edges(source, keys=True, data=True)
            if self._graph.has_edge(source, target, key=key)
            and attr["label"] not in ["_is_a_", "_relates_to_"]
            for cluster in attr["cluster"]
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
        for s, t, key, cluster in self._graph.edges(keys=True, data="cluster"):
            if key in keys and cluster in cluster:
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
        res: Set[FrozenSet[str]] = set()
        seen: Set[str] = set()
        for nodes_to_merge in (
            self._find_same_name_sources_to_merge()
            | self._find_same_name_targets_to_merge()
        ):
            new_nodes_to_merge = frozenset(
                node for node in nodes_to_merge if node not in seen
            )
            if len(new_nodes_to_merge) > 1:
                res.add(new_nodes_to_merge)
                seen.update(new_nodes_to_merge)
        return res

    def _find_same_name_sources_to_merge(self):
        # TODO merge with _find_same_name_targets_to_merge into a single function?
        labels_edges_dict: Dict[Tuple[str, str], Set[Tuple[str, str, str]]] = {}
        for s, t, k in self._graph.edges:
            if self._graph[s][t][k]["label"] in ["_is_a_", "_relates_to_"]:
                continue
            source_labels = self._graph.nodes[s]["label"]
            edge_labels = self._graph[s][t][k]["label"].split(" | ")
            for labels in product(source_labels, edge_labels):
                labels_edges_dict.setdefault(labels, set()).add((s, t, k))

        res: Set[FrozenSet[str]] = set()
        for edges in labels_edges_dict.values():
            if len(edges) < 2:
                continue
            targets = {t for _, t, _ in edges}
            targets = self._filter_node_merge_candidates(targets)
            targets = self._find_close_nodes(targets, SAME_NAME_NODE_DISTANCE_THRESHOLD)
            sources_to_merge = frozenset(s for s, t, _ in edges if t in targets)
            if len(sources_to_merge) < 2:
                continue
            logging.info(
                "Found same name sources to merge:\n"
                + "\n".join(
                    " | ".join(self._graph.nodes[node]["label"])
                    for node in sources_to_merge
                )
                + "\n"
                + "because of the next similar targets:\n"
                + "\n".join(
                    " | ".join(self._graph.nodes[node]["label"]) for node in targets
                )
            )
            res.add(sources_to_merge)
        return res

    def _find_same_name_targets_to_merge(self):
        labels_edges_dict: Dict[Tuple[str, str], Set[Tuple[str, str, str]]] = {}
        for s, t, k in self._graph.edges:
            if self._graph[s][t][k]["label"] in ["_is_a_", "_relates_to_"]:
                continue
            edge_labels = self._graph[s][t][k]["label"].split(" | ")
            target_labels = self._graph.nodes[t]["label"]
            for labels in product(edge_labels, target_labels):
                labels_edges_dict.setdefault(labels, set()).add((s, t, k))

        res: Set[FrozenSet[str]] = set()
        for edges in labels_edges_dict.values():
            if len(edges) < 2:
                continue
            sources = {s for s, _, _ in edges}
            sources = self._filter_node_merge_candidates(sources)
            sources = self._find_close_nodes(sources, SAME_NAME_NODE_DISTANCE_THRESHOLD)
            targets_to_merge = frozenset(t for s, t, _ in edges if s in sources)
            if len(targets_to_merge) < 2:
                continue
            logging.info(
                "Found same name targets to merge:\n"
                + "\n".join(
                    " | ".join(self._graph.nodes[node]["label"])
                    for node in targets_to_merge
                )
                + "\n"
                + "because of the next similar targets:\n"
                + "\n".join(
                    " | ".join(self._graph.nodes[node]["label"]) for node in sources
                )
            )
            res.add(targets_to_merge)
        return res

    def _add_implicit_is_a_relations(self):
        have_is_a: List[Tuple[str, str]] = []

        # find nodes with implicit "is a"
        for node in self._graph.nodes:
            all_predecessors_by_is_a = self._all_predecessors(
                node, by_relations=["_is_a_"]
            )
            for node1, node2 in product(all_predecessors_by_is_a, repeat=2):
                if self._has_implicit_is_a(node1, node2):
                    have_is_a.append((node1, node2))

        # connect nodes with implicit "is a"
        for node1, node2 in have_is_a:
            self._add_edge(
                node1,
                node2,
                "_is_a_",
                "_is_a_",
                "",
                self._graph.nodes[node1]["description"]
                | self._graph.nodes[node2]["description"],
                cluster=self._graph.nodes[node1]["cluster"]
                | self._graph.nodes[node2]["cluster"],
            )

        if len(have_is_a) > 0:
            self._add_implicit_is_a_relations()

    def _has_implicit_is_a(self, node1: str, node2: str):
        """
        True if there is an implicit relation (node1; is a; node2)
        """
        if (
            node1 == node2
            or self._graph.has_edge(node1, node2)
            or not (
                self._graph.nodes[node1]["cluster"]
                & self._graph.nodes[node1]["cluster"]
            )
        ):
            return False

        # instantiate with immediate successors by "is a" and "relates to"
        constituents1 = {
            successor
            for successor in self._graph.successors(node1)
            if set(self._graph[node1][successor]) & {"_is_a_", "_relates_to_"}
        }
        constituents2 = {
            successor
            for successor in self._graph.successors(node2)
            if set(self._graph[node2][successor]) & {"_is_a_", "_relates_to_"}
        }
        # one constituent means the phrase is just noun with adjective
        if len(constituents2) < 2:
            return False

        # complement with all successors by "is a"
        constituents1 |= {
            successor
            for node in constituents1
            for successor in self._all_successors(node, by_relations=["_is_a_"])
        }
        constituents2 |= {
            successor
            for node in constituents2
            for successor in self._all_successors(node, by_relations=["_is_a_"])
        }
        if constituents2.issubset(constituents1):
            logging.info(
                (
                    'Found implicit "is a" relation:\n'
                    + "({node1_label}; is a; {node2_label})\n"
                    + "All constituents of {node1_label}:\n"
                    + "{constituents1}\n"
                    + "All constituents of {node2_label}:\n"
                    + "{constituents2}\n"
                ).format(
                    node1_label=" | ".join(self._graph.nodes[node1]["label"]),
                    node2_label=" | ".join(self._graph.nodes[node2]["label"]),
                    constituents1="\n".join(
                        " | ".join(self._graph.nodes[node]["label"])
                        for node in constituents1
                    ),
                    constituents2="\n".join(
                        " | ".join(self._graph.nodes[node]["label"])
                        for node in constituents2
                    ),
                )
            )
            return True
        else:
            return False

    def _all_predecessors(
        self, node: str, by_relations: Optional[List[str]] = None
    ) -> Set[str]:
        if by_relations is None:
            by_relations = []
        acc = {node}
        while True:
            old_size = len(acc)
            acc |= {
                predecessor
                for saved_node in acc
                for predecessor, _, label in self._graph.in_edges(
                    saved_node, data="label"
                )
                if label in by_relations
            }
            if len(acc) == old_size:  # accumulator hasn't changed
                break
        acc.remove(node)
        return acc

    def _all_successors(
        self, node: str, by_relations: Optional[List[str]] = None
    ) -> Set[str]:
        if by_relations is None:
            by_relations = []
        acc = {node}
        while True:
            old_size = len(acc)
            acc |= {
                successor
                for saved_node in acc
                for _, successor, label in self._graph.out_edges(
                    saved_node, data="label"
                )
                if label in by_relations
            }
            if len(acc) == old_size:  # accumulator hasn't changed
                break
        acc.remove(node)
        return acc

    def _merge_nodes(self, nodes: Iterable[str]):
        def new_set_attr_value(attr_key):
            res = set()
            for node in nodes:
                res |= self._graph.nodes[node][attr_key]
            return res

        def new_str_attr_value(attr_key):
            res = set()
            for node in nodes:
                res |= set(self._graph.nodes[node][attr_key].split(" | "))
            return " | ".join(res)

        new_lemmas = new_str_attr_value("lemmas")
        new_label = new_set_attr_value("label")
        new_description = new_set_attr_value("description")
        new_cluster = new_set_attr_value("cluster")
        new_weight: int = sum(self._graph.nodes[node]["weight"] for node in nodes)
        new_vector: np.ndarray = sum(
            self._graph.nodes[node]["vector"] for node in nodes
        ) / len(nodes)
        new_node = self._add_node(
            new_lemmas,
            new_description,
            new_label,
            new_weight,
            new_vector,
            new_cluster,
        )

        for source, target, key in self._graph.edges(nodes, keys=True):
            if source in nodes:  # "out" edge
                new_source, new_target = (new_node, target)
            elif target in nodes:  # "in" edge
                new_source, new_target = (source, new_node)
            else:
                continue
            self._add_edge(
                new_source,
                new_target,
                self._graph.edges[source, target, key]["label"],
                self._graph.edges[source, target, key]["lemmas"],
                self._graph.edges[source, target, key]["deprel"],
                self._graph.edges[source, target, key]["description"],
                weight=self._graph.edges[source, target, key]["weight"],
                cluster=new_cluster,
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
                cluster=self._graph[source][target][key]["cluster"],
            )
            self._graph.remove_edge(source, target, key=key)

    def save(self, path):
        graph = self._transform_graph()
        stream_buffer = io.BytesIO()
        nx.write_gexf(graph, stream_buffer, encoding="utf-8", version="1.1draft")
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

    def _perform_filtering(self, nodes_to_remove: Iterable[str]):
        nodes_to_remove = set(nodes_to_remove)
        for node in nodes_to_remove:
            in_edges = list(self._graph.in_edges(node, keys=True))
            out_edges = list(self._graph.out_edges(node, keys=True))
            # if removing B: A->B->C  ==>  A->C
            for (pred, _, key_pred), (_, succ, key_succ) in product(
                in_edges, out_edges
            ):
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
                    cluster=self._graph[node][succ][key_succ]["cluster"],
                )
            self._graph.remove_node(node)

    def _transform_graph(self):
        """
        Transform graph to better suit for serialization and further visualization
        """
        res = deepcopy(self._graph)
        # transform relations from edges to nodes with specific node_type
        for node in res:
            res.nodes[node]["node_type"] = "argument"
        for source, target, key, attr in list(res.edges(data=True, keys=True)):
            label = res.edges[source, target, key]["label"]
            node = f"{label}({source}; {target})"  # FIXME shouldn't it include cluster number?
            new_attr = deepcopy(attr)
            new_attr["node_type"] = "relation"
            new_attr["weight"] = min(
                res.nodes[source]["weight"], res.nodes[target]["weight"]
            )
            if label == "_is_a_":
                color = {"b": 160, "g": 160, "r": 255}
            elif label == "_relates_to_":
                color = {"b": 160, "g": 255, "r": 160}
            else:
                color = {"b": 255, "g": 0, "r": 0}
            new_attr["viz"] = {"color": color}
            res.add_node(node, **new_attr)
            res.add_edge(source, node)
            res.add_edge(node, target)
            res.remove_edge(source, target, key=key)

        # convert non-serializable attributes to strings
        for node in res:
            if res.nodes[node].get("vector") is not None:
                res.nodes[node]["vector"] = str(res.nodes[node]["vector"].tolist())
            if isinstance(res.nodes[node]["label"], set):
                res.nodes[node]["label"] = " | ".join(res.nodes[node]["label"])
            res.nodes[node]["description"] = " | ".join(res.nodes[node]["description"])
            res.nodes[node]["feat_type"] = " | ".join(
                str(elem) for elem in res.nodes[node]["cluster"]
            )
            del res.nodes[node]["cluster"]
        return res

    @staticmethod
    def _fix_gexf(root_element):
        """
        GraphView is quite choosy about the input files. Not every GEXF file can
        be read by it. This function fixes GEXF XML tree to better match
        GraphView's requirements
        """
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
        parsed_text: Doc,
        w2v_model,
        stopwords,
        additional_relations,
        entities_limit,
    ):
        reltuples = [
            SentenceReltuples(
                s,
                w2v_model,
                additional_relations=additional_relations,
                stopwords=stopwords,
            )
            for s in parsed_text.sents
        ]
        self._dict = {
            str(sentence_reltuples.sentence): [
                (
                    reltuple.left_arg.phrase,
                    reltuple.relation.phrase,
                    reltuple.right_arg.phrase,
                )
                for reltuple in sentence_reltuples
            ]
            for sentence_reltuples in reltuples
        }
        clusters = self._cluster(
            reltuples,
            min_cluster_size=MIN_CLUSTER_SIZE,
            max_cluster_size=MIN_CLUSTER_SIZE + 50,
        )
        self._graph = RelGraph.from_reltuples(
            reltuples, clusters=clusters, entities_limit=entities_limit
        )
        logging.info(
            "Relation tuples have been extracted from texts. "
            "The resulting graph consists of\n"
            f"{self._graph.nodes_number} nodes\n"
            f"{self._graph.edges_number} edges\n"
            "{} connected components".format(
                networkx.algorithms.components.number_weakly_connected_components(
                    self._graph._graph
                )
            )
        )

    @property
    def graph(self):
        return self._graph

    @property
    def dictionary(self):
        return self._dict

    @staticmethod
    def _cluster(
        reltuples: Sequence[SentenceReltuples],
        min_cluster_size=10,
        max_cluster_size=100,
        cluster_size_step=10,
    ) -> List[int]:
        X = np.array(
            [sentence_reltuples.sentence_vector for sentence_reltuples in reltuples]
        )
        max_sil_score = -1
        n_sentences = len(reltuples)
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


@njit()
def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    udotv = np.dot(u, v)
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    if (u_norm == 0) or (v_norm == 0):
        ratio = 0
    else:
        ratio = udotv / (u_norm * v_norm)
    return 1 - ratio
