"""Contains functions used to generate prompts for Layout to Image Dataset."""

import sys
from typing import List

import numpy as np
from . import constants



def get_basic_prompt(obj_class: str) -> str:
    """Basic image generation prompt with object class name."""
    return (
        f"A segmented image of a {obj_class} on a black "
        "background"
    )

def get_styled_prompt(obj_class: str) -> str:
    """Image generation prompt with object class and stylization keywords."""

    return (
        f"A segmented sprite of a real {obj_class} with "
        "photo-realistic features and body structure on a "
        "black background."
    )

def get_prompt_wt_parts(obj_class: str, layout: np.matrix) -> str:
    """Advanced prompt with """

    prompt = (
        f"A segmented sprite of a real {obj_class} with "
        "photo-realistic details and body structure on a black "
        "background."
    )

    parts = np.where(layout[:, 0]==1)[0]

    part_mapping = list(constants.ALL_PART_MAPPING[obj_class].keys())

    part_list = [part_mapping[part] for part in parts]

    if part_list:
        prompt += " Image shows: "
        for part in part_list:
            prompt += f"{obj_class}'s {part}, "
        prompt = prompt[:-2]+"."

    return prompt
