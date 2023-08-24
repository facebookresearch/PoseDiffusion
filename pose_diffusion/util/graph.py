from typing import List, Dict, Tuple, Union, Set
from collections import defaultdict
import numpy as np

class Match:
    def __init__(self, node_idx: int, sim: float):
        self.node_idx = node_idx
        self.sim = sim

class FeatureNode:
    def __init__(self, image_id: int, feature_idx: int):
        self.image_id = image_id
        self.feature_idx = feature_idx
        self.node_idx = -1
        self.out_matches = []  # outward matches
        self.track_idx = None
        # self.err = None
        self.pt = None
        self.proj_pt = None
        self.pt3d = None
        
    def __repr__(self):
        return f"FeatureNode(node_idx={self.node_idx}, image_id={self.image_id}, feature_idx={self.feature_idx}, num_matches={len(self.out_matches)}, pt={self.pt}, proj_pt={self.proj_pt}, pt3d={self.pt3d}, track_idx={self.track_idx})"


class Graph:
    def __init__(self):
        self.nodes = []
        self.image_name_to_id = {}
        self.image_id_to_name = {}
        self.node_map = {}

    def add_node(self, node) -> int:
        self.nodes.append(node)
        node.node_idx = len(self.nodes) - 1
        return node.node_idx

    def find_or_create_node(self, image_name: str, feature_idx: int):
        image_id = self.image_name_to_id.setdefault(image_name, len(self.image_name_to_id))
        if image_id not in self.image_id_to_name:
            self.image_id_to_name[image_id] = image_name

        key = (image_id, feature_idx)
        if key in self.node_map:
            return self.nodes[self.node_map[key]]
        else:
            node = FeatureNode(image_id, feature_idx)
            node_idx = self.add_node(node)
            self.node_map[key] = node_idx
            return node

    def get_degrees(self) -> List[int]:
        output_degrees = [0]*len(self.nodes)
        for node in self.nodes:
            output_degrees[node.node_idx] += len(node.out_matches)
            for match in node.out_matches:
                output_degrees[match.node_idx] += 1
        return output_degrees

    def get_scores(self) -> List[float]:
        scores = [0.0]*len(self.nodes)
        for node in self.nodes:
            for match in node.out_matches:
                scores[match.node_idx] += match.sim
                scores[node.node_idx] += match.sim
        return scores

    def get_edges(self) -> List[Tuple[int, int, float]]:
        edges = []
        for node in self.nodes:
            for match in node.out_matches:
                edges.append((node.node_idx, match.node_idx, match.sim))
        return edges

    def add_edge(self, node1: FeatureNode, node2: FeatureNode, sim: float):
        match = Match(node2.node_idx, sim)
        node1.out_matches.append(match)

    def register_matches(self, imname1: str, imname2: str, matches: List[int], similarities: List[float] = None):
        n_matches = len(matches)
        if similarities is None:
            similarities = [1.0]*n_matches

        for match_idx in range(n_matches):
            feature_idx1 = matches[match_idx, 0]
            feature_idx2 = matches[match_idx, 1]
                        
            similarity = similarities[match_idx]
            node1 = self.find_or_create_node(imname1, feature_idx1)
            node2 = self.find_or_create_node(imname2, feature_idx2)

            self.add_edge(node1, node2, similarity)


# import numpy as np

def union_find_get_root(node_idx, parent_nodes):
    if parent_nodes[node_idx] == -1:
        return node_idx
    # Union-find path compression heuristic.
    parent_nodes[node_idx] = union_find_get_root(parent_nodes[node_idx], parent_nodes)
    return parent_nodes[node_idx]

def compute_track_labels(graph):
    print("Computing tracks...")
    n_nodes = len(graph.nodes)
    print("# graph nodes: ", n_nodes)
    edges = []
    for node in graph.nodes:
        for match in node.out_matches:
            edges.append((match.sim, node.node_idx, match.node_idx))
    print("# graph edges: ", len(edges))

    # Build the MSF.
    edges.sort(reverse=True)
    parent_nodes = np.full(n_nodes, -1, dtype=int)
    images_in_track = [set() for _ in range(n_nodes)]

    for node_idx in range(n_nodes):
        images_in_track[node_idx].add(graph.nodes[node_idx].image_id)

    for it in edges:
        node_idx1 = it[1]
        node_idx2 = it[2]

        root1 = union_find_get_root(node_idx1, parent_nodes)
        root2 = union_find_get_root(node_idx2, parent_nodes)

        if root1 != root2:
            intersection = images_in_track[root1].intersection(images_in_track[root2])
            if len(intersection) != 0:
                continue
            # Union-find merging heuristic.
            if len(images_in_track[root1]) < len(images_in_track[root2]):
                parent_nodes[root1] = root2
                images_in_track[root2].update(images_in_track[root1])
                images_in_track[root1].clear()
            else:
                parent_nodes[root2] = root1
                images_in_track[root1].update(images_in_track[root2])
                images_in_track[root2].clear()

    # Compute the tracks.
    track_labels = np.full(n_nodes, -1, dtype=int)

    n_tracks = 0
    for node_idx in range(n_nodes):
        if parent_nodes[node_idx] == -1:
            track_labels[node_idx] = n_tracks
            n_tracks += 1
    print("# tracks: ", n_tracks)

    for node_idx in range(n_nodes):
        if track_labels[node_idx] != -1:
            continue
        track_labels[node_idx] = track_labels[union_find_get_root(node_idx, parent_nodes)]

    return track_labels.tolist()



def compute_score_labels(graph, track_labels):
    n_nodes = len(graph.nodes)
    score_labels = [0.0] * n_nodes
    for node_idx in range(n_nodes):
        node = graph.nodes[node_idx]
        for match in node.out_matches:
            if track_labels[node_idx] == track_labels[match.node_idx]:
                score_labels[node_idx] += match.sim
                score_labels[match.node_idx] += match.sim
    return score_labels


def compute_root_labels(graph, track_labels, score_labels):
    n_nodes = len(graph.nodes)
    n_tracks = max(track_labels) + 1
    scores = [(score_labels[node_idx], node_idx) for node_idx in range(n_nodes)]

    scores.sort(reverse=True)
    
    is_root = [False] * n_nodes
    has_root = [False] * n_tracks

    for score, node_idx in scores:
        if has_root[track_labels[node_idx]]:
            continue
        is_root[node_idx] = True
        has_root[track_labels[node_idx]] = True

    return is_root


# def count_edges_ab(graph, track_labels, is_root):
#     n_nodes = len(graph.nodes)
#     track_edge_counts = [(0, 0)] * n_nodes
#     for node in graph.nodes:
#         node_idx1 = node.node_idx
#         for match in node.out_matches:
#             node_idx2 = match.node_idx
#             track_idx1 = track_labels[node_idx1]
#             track_idx2 = track_labels[node_idx2]
#             if track_idx1 == track_idx2:
#                 if is_root[node_idx1] or is_root[node_idx2]:
#                     track_edge_counts[track_idx1] = (track_edge_counts[track_idx1][0] + 1, track_edge_counts[track_idx1][1])
#                 else:
#                     track_edge_counts[track_idx1] = (track_edge_counts[track_idx1][0], track_edge_counts[track_idx1][1] + 1)
#     return track_edge_counts


# def count_track_edges(graph, track_labels):
#     n_tracks = len(set(track_labels))
#     track_edge_counts = [0] * n_tracks
#     for node in graph.nodes:
#         node_idx1 = node.node_idx
#         for match in node.out_matches:
#             node_idx2 = match.node_idx
#             track_idx1 = track_labels[node_idx1]
#             track_idx2 = track_labels[node_idx2]
#             if track_idx1 == track_idx2:
#                 track_edge_counts[track_idx1] += 1

#     return track_edge_counts
