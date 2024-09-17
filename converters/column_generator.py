"""Functions for creating dataset columns."""

from ..utils import plotting

def add_binarized_obj_mask(examples):
    """Functions that adds a binarized object mask using the conditioning image.
    """
    examples['obj_bbox_mask'] = [
        plotting.binarize_to_object(image) for image in examples["conditioning_image"]]
    return examples

def add_binarized_masks(examples):
    """Functions that generates layout binarized masks using the conditioning image.
    """
    examples["mask_image"] = [
        plotting.binarize(image) for image in examples["conditioning_image"]]
    return examples
