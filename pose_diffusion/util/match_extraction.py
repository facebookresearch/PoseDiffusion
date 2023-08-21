import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pycolmap

from typing import Optional, Iterator, List, Tuple, Dict, Any
import h5py
from hloc.utils.io import find_pair
import h5py
from util.graph import Graph, compute_track_labels, compute_score_labels, compute_root_labels
# from demo import read_points3D_binary
from hloc import reconstruction

from hloc import (
    extract_features,
    logger,
    match_features,
    pairs_from_exhaustive,
    pairs_from_retrieval
)
from hloc.triangulation import (
    import_features,
    import_matches,
    estimation_and_geometric_verification,
    parse_option_args,
    OutputCapture,
)
from hloc.utils.database import (
    COLMAPDatabase,
    image_ids_to_pair_id,
    # pair_id_to_image_ids,
)
from hloc.reconstruction import create_empty_db, import_images, get_image_ids


def extract_match(image_folder_path: str, image_info: Dict, use_graph = False, cfg=None):
    # Now only supports SPSG
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_mapping = os.path.join(tmpdir, "mapping")
        os.makedirs(tmp_mapping)
        for filename in os.listdir(image_folder_path):
            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")
            ):
                shutil.copy(
                    os.path.join(image_folder_path, filename),
                    os.path.join(tmp_mapping, filename),
                )
                
        if use_graph:
            graph, track_labels, root_labels, keypoints, pc_hloc = run_hloc_graph(tmpdir, cfg)
            pc_hloc = points3D_to_nparray(pc_hloc)
            return graph, track_labels, root_labels, keypoints, pc_hloc
        else:        
            matches, keypoints = run_hloc(tmpdir, cfg)


    # From the format of colmap to PyTorch3D
    kp1, kp2, i12 = colmap_keypoint_to_pytorch3d(matches, keypoints, image_info)

    return kp1, kp2, i12

def colmap_keypoint_to_pytorch3d(matches, keypoints, image_info):
    kp1, kp2, i12 = [], [], []
    bbox_xyxy, scale = image_info["bboxes_xyxy"], image_info["resized_scales"]

    for idx in keypoints:
        # coordinate change from COLMAP to OpenCV
        cur_keypoint = keypoints[idx] - 0.5

        # go to the coordiante after cropping
        # use idx - 1 here because the COLMAP format starts from 1 instead of 0
        cur_keypoint = cur_keypoint - [
            bbox_xyxy[idx - 1][0],
            bbox_xyxy[idx - 1][1],
        ]
        cur_keypoint = cur_keypoint * scale[idx - 1]
        keypoints[idx] = cur_keypoint

    for (r_idx, q_idx), pair_match in matches.items():
        if pair_match is not None:
            kp1.append(keypoints[r_idx][pair_match[:, 0]])
            kp2.append(keypoints[q_idx][pair_match[:, 1]])

            i12_pair = np.array([[r_idx - 1, q_idx - 1]])
            i12.append(np.repeat(i12_pair, len(pair_match), axis=0))

    if kp1:
        kp1, kp2, i12 = map(np.concatenate, (kp1, kp2, i12), (0, 0, 0))
    else:
        kp1 = kp2 = i12 = None

    return kp1, kp2, i12


def run_hloc_graph(output_dir: str, cfg=None):
    # learned from
    # https://github.com/cvg/Hierarchical-Localization/blob/master/pipeline_SfM.ipynb

    images = Path(output_dir)
    outputs = Path(os.path.join(output_dir, "output"))
    sfm_pairs = outputs / "pairs-sfm.txt"
    sfm_dir = outputs / "sfm"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    feature_conf = extract_features.confs[
        "superpoint_inloc"
    ]  # or superpoint_max
    matcher_conf = match_features.confs["superglue"]

    references = [
        p.relative_to(images).as_posix()
        for p in (images / "mapping/").iterdir()
    ]

    extract_features.main(
        feature_conf, images, image_list=references, feature_path=features
    )
    
    if cfg.match.exhau_search:
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    else:
        retrieval_conf = extract_features.confs['netvlad']
        retrieval_path = extract_features.main(retrieval_conf, images, outputs)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=cfg.match.num_matched)


    match_features.main(
        matcher_conf, sfm_pairs, features=features, matches=matches
    )

    keypoints = read_keypoints_hloc(features, as_cpp_map=False)
    pairs = read_image_pairs(sfm_pairs)
    pairs.sort()
    matches_scores = read_matches_hloc(matches, pairs)

    graph = build_matching_graph(pairs, matches_scores[0], matches_scores[1], cfg)

    track_labels = compute_track_labels(graph)
    
    score_labels = compute_score_labels(graph, track_labels)
    
    # node within each track with highest score
    root_labels = compute_root_labels(graph, track_labels, score_labels)    

    model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)

    pc_hloc = read_points3D_binary(sfm_dir / "points3D.bin")

    return graph, track_labels, root_labels, keypoints, pc_hloc


def run_hloc(output_dir: str, cfg=None):
    # learned from
    # https://github.com/cvg/Hierarchical-Localization/blob/master/pipeline_SfM.ipynb

    images = Path(output_dir)
    outputs = Path(os.path.join(output_dir, "output"))
    sfm_pairs = outputs / "pairs-sfm.txt"
    sfm_dir = outputs / "sfm"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    feature_conf = extract_features.confs[
        "superpoint_inloc"
    ]  # or superpoint_max
    matcher_conf = match_features.confs["superglue"]

    references = [
        p.relative_to(images).as_posix()
        for p in (images / "mapping/").iterdir()
    ]

    extract_features.main(
        feature_conf, images, image_list=references, feature_path=features
    )
    

    if cfg.match.exhau_search:
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    else:
        retrieval_conf = extract_features.confs['netvlad']
        retrieval_path = extract_features.main(retrieval_conf, images, outputs)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=cfg.match.num_matched)


    match_features.main(
        matcher_conf, sfm_pairs, features=features, matches=matches
    )

    matches, keypoints = compute_matches_and_keypoints(
        sfm_dir, images, sfm_pairs, features, matches, image_list=references
    )

    return matches, keypoints

def read_image_pairs(path) -> List[Tuple[str]]:
    with open(path, "r") as f:
        pairs = [p.split() for p in f.read().rstrip('\n').split('\n')]
    return pairs


def read_keypoints_hloc(path: Path, names: Optional[Iterator[str]] = None,
                        as_cpp_map: bool = False) -> Dict[str, np.ndarray]:
    if as_cpp_map:
        keypoint_dict = Map_NameKeypoints()
    else:
        keypoint_dict = {}
    if names is None:
        names = list_h5_names(path)
    with h5py.File(str(path), "r") as h5f:
        for name in names:
            keypoints = h5f[name]["keypoints"].__array__()[:, :2]
            keypoint_dict[name] = keypoints.astype(np.float64)
    return keypoint_dict





def read_matches_hloc(path: Path, pairs: Iterator[Tuple[str]]
                      ) -> Tuple[List[np.ndarray]]:
    matches = []
    scores = []
    with h5py.File(path, "r") as h5f:
        for k1, k2 in pairs:
            pair, reverse = find_pair(h5f, str(k1), str(k2))
            m = h5f[pair]["matches0"].__array__()
            idx = np.where(m != -1)[0]
            m = np.stack([idx, m[idx]], -1).astype(np.uint64)
            s = h5f[pair]["matching_scores0"].__array__()
            s = s[idx].astype(np.float32)
            if reverse:
                m = np.flip(m, -1)
            matches.append(m)
            scores.append(s)
    return matches, scores


def list_h5_names(path):
    names = []
    with h5py.File(str(path), 'r') as fd:
        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip('/'))
        fd.visititems(visit_fn)
    return list(set(names))


def build_matching_graph(
        pairs: List[Tuple[str]],
        matches: List[np.ndarray],
        scores: Optional[List[np.ndarray]] = None,
        cfg = None,):
    logger.info("Building matching graph...")
    graph = Graph()
    scores = scores if scores is not None else [None for _ in matches]
    for (name1, name2), ma, ss in zip(pairs, matches, scores):
        mask = ss > cfg.match.match_thres
        ma = ma[mask]
        ss = ss[mask]
        graph.register_matches(name1, name2, ma, ss)
    
    return graph


def compute_matches_and_keypoints(
    sfm_dir: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    verbose: bool = False,
    min_match_score: Optional[float] = None,
    image_list: Optional[List[str]] = None,
    image_options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    # learned from
    # https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/reconstruction.py

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"

    create_empty_db(database)
    import_images(image_dir, database, camera_mode, image_list, image_options)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches, min_match_score)
    estimation_and_geometric_verification(database, pairs, verbose)


    
    db = COLMAPDatabase.connect(database)

    matches = dict(
        (
            pair_id_to_image_ids(pair_id),
            _blob_to_array_safe(data, np.uint32, (-1, 2)),
        )
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
    )

    keypoints = dict(
        (image_id, _blob_to_array_safe(data, np.float32, (-1, 2)))
        for image_id, data in db.execute("SELECT image_id, data FROM keypoints")
    )

    db.close()

    return matches, keypoints


def _blob_to_array_safe(blob, dtype, shape=(-1,)):
    if blob is not None:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return blob



def read_image_pairs(path) -> List[Tuple[str]]:
    with open(path, "r") as f:
        pairs = [p.split() for p in f.read().rstrip('\n').split('\n')]
    return pairs



def read_matches_hloc(path: Path, pairs: Iterator[Tuple[str]]
                      ) -> Tuple[List[np.ndarray]]:
    matches = []
    scores = []
    with h5py.File(path, "r") as h5f:
        for k1, k2 in pairs:
            pair, reverse = find_pair(h5f, str(k1), str(k2))
            m = h5f[pair]["matches0"].__array__()
            idx = np.where(m != -1)[0]
            m = np.stack([idx, m[idx]], -1).astype(np.uint64)
            s = h5f[pair]["matching_scores0"].__array__()
            s = s[idx].astype(np.float32)
            if reverse:
                m = np.flip(m, -1)
            matches.append(m)
            scores.append(s)
    return matches, scores

MAX_IMAGE_ID = 2**31 - 1


def pair_id_to_image_ids(pair_id):
    image_id2 = int(pair_id % MAX_IMAGE_ID)
    image_id1 = int((pair_id - image_id2) / MAX_IMAGE_ID)
    return image_id1, image_id2





import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assume your point cloud data is stored in the variable 'point_cloud'
# point_cloud = np.random.rand(100, 3)  # Uncomment this line if you want to generate random point cloud for testing

import os
import collections
import numpy as np
import struct
import argparse


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D

def points3D_to_nparray(points3D):
    # Extract xyz fields from Point3D objects and stack them into a numpy array
    xyz_array = np.stack([point.xyz for point in points3D.values()])
    return xyz_array
