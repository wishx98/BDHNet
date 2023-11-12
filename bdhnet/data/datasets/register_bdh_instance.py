# -*- encoding: utf-8 -*-
"""
@File    :   register_bdh_instance.py
@Time    :   2023/08/29 15:17:19
@Author  :   Wenxu Shi
@Version :   1.0
@Contact :   shiwenxu20@mails.ucas.ac.cn
"""

import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog

# from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from .bdh import load_bdh_json
from detectron2.utils.file_io import PathManager

BDH_CATEGORIES = [
    {"id": 1, "name": "building"},
]


_PREDEFINED_SPLITS = {
    # point annotations without masks
    "bdh_train": (
        "BD_Height/bdh_train",
        "BD_Height/annotations/annotations_train.json",
    ),
    "bdh_val": (
        "BD_Height/bdh_val",
        "BD_Height/annotations/annotations_val.json",
    ),
}


def _get_bdh_instances_meta():
    thing_ids = [k["id"] for k in BDH_CATEGORIES]
    # assert len(thing_ids) == 3, len(thing_ids)
    # assert len(thing_ids) == 1, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in BDH_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_bdh_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_bdh_json(json_file, image_root, name, extra_annotation_keys=["bd_height", "offset"]))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(json_file=json_file, image_root=image_root, evaluator_type="bdh_instance_seg", **metadata)


def register_all_bdh_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_bdh_instances(
            key,
            _get_bdh_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_bdh_instance(_root)
