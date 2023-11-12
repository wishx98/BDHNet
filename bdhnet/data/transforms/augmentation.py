# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import inspect
import numpy as np
import pprint
from typing import Any, List, Optional, Tuple, Union
from fvcore.transforms.transform import Transform, TransformList
from detectron2.data.transforms.augmentation import Augmentation

"""
See "Data Augmentation" tutorial for an overview of the system:
https://detectron2.readthedocs.io/tutorials/augmentation.html
"""


__all__ = [
    "Augmentation",
    "AugmentationList",
    "BDH_AugInput",
    "TransformGen",
    "apply_transform_gens",
    "StandardAugInput",
    "apply_augmentations",
]


def _check_img_dtype(img):
    assert isinstance(img, np.ndarray), "[Augmentation] Needs an numpy array, but got a {}!".format(type(img))
    assert not isinstance(img.dtype, np.integer) or (
        img.dtype == np.uint8
    ), "[Augmentation] Got image of type {}, use uint8 or floating points instead!".format(img.dtype)
    assert img.ndim in [2, 3], img.ndim


def _get_aug_input_args(aug, aug_input) -> List[Any]:
    """
    Get the arguments to be passed to ``aug.get_transform`` from the input ``aug_input``.
    """
    if aug.input_args is None:
        # Decide what attributes are needed automatically
        prms = list(inspect.signature(aug.get_transform).parameters.items())
        # The default behavior is: if there is one parameter, then its "image"
        # (work automatically for majority of use cases, and also avoid BC breaking),
        # Otherwise, use the argument names.
        if len(prms) == 1:
            names = ("image",)
        else:
            names = []
            for name, prm in prms:
                if prm.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    raise TypeError(
                        f""" \
The default implementation of `{type(aug)}.__call__` does not allow \
`{type(aug)}.get_transform` to use variable-length arguments (*args, **kwargs)! \
If arguments are unknown, reimplement `__call__` instead. \
"""
                    )
                names.append(name)
        aug.input_args = tuple(names)

    args = []
    for f in aug.input_args:
        try:
            args.append(getattr(aug_input, f))
        except AttributeError as e:
            raise AttributeError(
                f"{type(aug)}.get_transform needs input attribute '{f}', " f"but it is not an attribute of {type(aug_input)}!"
            ) from e
    return args


def _transform_to_aug(tfm_or_aug):
    """
    Wrap Transform into Augmentation.
    Private, used internally to implement augmentations.
    """
    assert isinstance(tfm_or_aug, (Transform, Augmentation)), tfm_or_aug
    if isinstance(tfm_or_aug, Augmentation):
        return tfm_or_aug
    else:

        class _TransformToAug(Augmentation):
            def __init__(self, tfm: Transform):
                self.tfm = tfm

            def get_transform(self, *args):
                return self.tfm

            def __repr__(self):
                return repr(self.tfm)

            __str__ = __repr__

        return _TransformToAug(tfm_or_aug)


class AugmentationList(Augmentation):
    """
    Apply a sequence of augmentations.

    It has ``__call__`` method to apply the augmentations.

    Note that :meth:`get_transform` method is impossible (will throw error if called)
    for :class:`AugmentationList`, because in order to apply a sequence of augmentations,
    the kth augmentation must be applied first, to provide inputs needed by the (k+1)th
    augmentation.
    """

    def __init__(self, augs):
        """
        Args:
            augs (list[Augmentation or Transform]):
        """
        super().__init__()
        self.augs = [_transform_to_aug(x) for x in augs]

    def __call__(self, aug_input) -> Transform:
        tfms = []
        scale = 1.0
        for x in self.augs:
            if "Resize" in str(x.__class__):
                tfm, scale = x(aug_input)
            else:
                tfm = x(aug_input)
            tfms.append(tfm)
        return TransformList(tfms), scale

    def __repr__(self):
        msgs = [str(x) for x in self.augs]
        return "AugmentationList[{}]".format(", ".join(msgs))

    __str__ = __repr__


class BDH_AugInput:
    """
    Input that can be used with :meth:`Augmentation.__call__`.
    This is a standard implementation for the majority of use cases.
    This class provides the standard attributes **"image", "boxes", "sem_seg"**
    defined in :meth:`__init__` and they may be needed by different augmentations.
    Most augmentation policies do not need attributes beyond these three.

    After applying augmentations to these attributes (using :meth:`AugInput.transform`),
    the returned transforms can then be used to transform other data structures that users have.

    Examples:
    ::
        input = AugInput(image, boxes=boxes)
        tfms = augmentation(input)
        transformed_image = input.image
        transformed_boxes = input.boxes
        transformed_other_data = tfms.apply_other(other_data)

    An extended project that works with new data types may implement augmentation policies
    that need other inputs. An algorithm may need to transform inputs in a way different
    from the standard approach defined in this class. In those rare situations, users can
    implement a class similar to this class, that satify the following condition:

    * The input must provide access to these data in the form of attribute access
      (``getattr``).  For example, if an :class:`Augmentation` to be applied needs "image"
      and "sem_seg" arguments, its input must have the attribute "image" and "sem_seg".
    * The input must have a ``transform(tfm: Transform) -> None`` method which
      in-place transforms all its attributes.
    """

    # TODO maybe should support more builtin data types here
    def __init__(
        self,
        image: np.ndarray,
        *,
        boxes: Optional[np.ndarray] = None,
        sem_seg: Optional[np.ndarray] = None,
        bd_height: np.float16 = None,
        offset: Optional[np.ndarray] = None,
    ):
        """
        Args:
            image (ndarray): (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
                floating point in range [0, 1] or [0, 255]. The meaning of C is up
                to users.
            boxes (ndarray or None): Nx4 float32 boxes in XYXY_ABS mode
            sem_seg (ndarray or None): HxW uint8 semantic segmentation mask. Each element
                is an integer label of pixel.
            bd_height (float16 or None): height of the building instance
            offset (ndarray or None): 2x1 float32 offset of the building instance
        """
        _check_img_dtype(image)
        self.image = image
        self.boxes = boxes
        self.sem_seg = sem_seg
        self.bd_height = bd_height
        self.offset = offset

    def transform(self, tfm: Transform) -> None:
        """
        In-place transform all attributes of this class.

        By "in-place", it means after calling this method, accessing an attribute such
        as ``self.image`` will return transformed data.
        """
        self.image = tfm.apply_image(self.image)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)
        if self.bd_height is not None:
            self.bd_height = self.bd_height
        if self.offset is not None:
            self.offset = tfm.apply_segmentation(self.offset)

    def apply_augmentations(self, augmentations: List[Union[Augmentation, Transform]]) -> TransformList:
        """
        Equivalent of ``AugmentationList(augmentations)(self)``
        """
        return AugmentationList(augmentations)(self)


def apply_augmentations(augmentations: List[Union[Transform, Augmentation]], inputs):
    """
    Use ``T.AugmentationList(augmentations)(inputs)`` instead.
    """
    if isinstance(inputs, np.ndarray):
        # handle the common case of image-only Augmentation, also for backward compatibility
        image_only = True
        inputs = BDH_AugInput(inputs)
    else:
        image_only = False
    tfms, scale = inputs.apply_augmentations(augmentations)
    return inputs.image if image_only else inputs, tfms, scale


apply_transform_gens = apply_augmentations
"""
Alias for backward-compatibility.
"""

TransformGen = Augmentation
"""
Alias for Augmentation, since it is something that generates :class:`Transform`s
"""

StandardAugInput = BDH_AugInput
"""
Alias for compatibility. It's not worth the complexity to have two classes.
"""
