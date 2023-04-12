import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pycolmap
from typing import Optional, List, Dict, Any
from hloc import (
    extract_features,
    logger,
    match_features,
    pairs_from_exhaustive,
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
    pair_id_to_image_ids,
)


def extract_match(image_folder_path: str, image_info: Dict):
    # Now support SPSG
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copytree(image_folder_path, os.path.join(tmpdir, "mapping"))
        matches, keypoints = run_hloc(tmpdir)

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


def run_hloc(output_dir: str):
    # Largely borrowed from hlob
    images = Path(output_dir)
    outputs = Path(os.path.join(output_dir, "output"))
    sfm_pairs = outputs / "pairs-sfm.txt"
    sfm_dir = outputs / "sfm"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    feature_conf = extract_features.confs["superpoint_inloc"]  # or superpoint_max
    matcher_conf = match_features.confs["superglue"]

    references = [
        p.relative_to(images).as_posix() for p in (images / "mapping/").iterdir()
    ]

    extract_features.main(
        feature_conf, images, image_list=references, feature_path=features
    )
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    matches, keypoints = compute_match_and_keypoint(
        sfm_dir, images, sfm_pairs, features, matches, image_list=references
    )

    return matches, keypoints


def compute_match_and_keypoint(
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
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"

    _create_empty_db(database)
    _import_images(image_dir, database, camera_mode, image_list, image_options)
    image_ids = _get_image_ids(database)
    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches, min_match_score)
    estimation_and_geometric_verification(database, pairs, verbose)

    db = COLMAPDatabase.connect(database)

    matches = dict(
        (pair_id_to_image_ids(pair_id), _blob_to_array_safe(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
    )

    keypoints = dict(
        (image_id, _blob_to_array_safe(data, np.float32, (-1, 2)))
        for image_id, data in db.execute("SELECT image_id, data FROM keypoints")
    )

    db.close()

    return matches, keypoints


# helper functions


def _create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()
    logger.info("Creating an empty database...")
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def _import_images(
    image_dir: Path,
    database_path: Path,
    camera_mode: pycolmap.CameraMode,
    image_list: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None,
):
    logger.info("Importing images into the database...")
    if options is None:
        options = {}
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f"No images found in {image_dir}.")
    with pycolmap.ostream():
        pycolmap.import_images(
            database_path,
            image_dir,
            camera_mode,
            image_list=image_list or [],
            options=options,
        )


def _get_image_ids(database_path: Path) -> Dict[str, int]:
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images


def _blob_to_array_safe(blob, dtype, shape=(-1,)):
    if blob is not None:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return blob