"""Doc string"""

import numpy as np


_COLORS = [(1, 0, 0),
           (0.737, 0.561, 0.561),
           (0.255, 0.412, 0.882),
           (0.545, 0.271, 0.0745),
           (0.98, 0.502, 0.447),
           (0.98, 0.643, 0.376),
           (0.18, 0.545, 0.341),
           (0.502, 0, 0.502),
           (0.627, 0.322, 0.176),
           (0.753, 0.753, 0.753),
           (0.529, 0.808, 0.922),
           (0.416, 0.353, 0.804),
           (0.439, 0.502, 0.565),
           (0.784, 0.302, 0.565),
           (0.867, 0.627, 0.867),
           (0, 1, 0.498),
           (0.275, 0.51, 0.706),
           (0.824, 0.706, 0.549),
           (0, 0.502, 0.502),
           (0.847, 0.749, 0.847),
           (1, 0.388, 0.278),
           (0.251, 0.878, 0.816),
           (0.933, 0.51, 0.933),
           (0.961, 0.871, 0.702)]
COLORS = (np.asarray(_COLORS)*255).astype(int)
CANVAS_SIZE = 660
CLASS_DICT = {
    0: 'cow',
    1: 'sheep',
    2: 'bird',
    3: 'person',
    4: 'cat',
    5: 'dog',
    6: 'horse',
    7: 'aeroplane',
    8: 'motorbike',
    9: 'bicycle',
    10: 'car',
}
SET_NAME = 'scaled_filled_boxes'

bird_part_labels = {
    'head': 0, 'torso': 1, 'neck': 2, 'left wing': 3, 'right wing': 4,
    'left leg': 5, 'left foot': 6, 'right leg': 7, 'right foot': 8, 'tail': 9
}

cat_part_labels = {
    'head': 0, 'torso': 1, 'neck': 2, 'left front leg': 3, 'left front paw': 4,
    'right front leg': 5, 'right front paw': 6, 'left back leg': 7,
    'left back paw': 8, 'right back leg': 9, 'right back paw': 10, 'tail': 11
}

cow_part_labels = {
    'head': 0, 'left horn': 1, 'right horn': 2, 'torso': 3, 'neck': 4,
    'left front upper leg': 5, 'left front lower leg': 6,
    'right front upper leg': 7, 'right front lower leg': 8,
    'left back upper leg': 9, 'left back lower leg': 10,
    'right back upper leg': 11, 'right back lower leg': 12, 'tail': 13
}

dog_part_labels = {
    'head': 0, 'torso': 1, 'neck': 2, 'left front leg': 3, 'left front paw': 4,
    'right front leg': 5, 'right front paw': 6, 'left back leg': 7,
    'left back paw': 8, 'right back leg': 9, 'right back paw': 10,
    'tail': 11, 'muzzle': 12
}

horse_part_labels = {
    'head': 0, 'left front hoof': 1, 'right front hoof': 2, 'torso': 3,
    'neck': 4, 'left front upper leg': 5, 'left front lower leg': 6,
    'right front upper leg': 7, 'right front lower leg': 8,
    'left back upper leg': 9, 'left back lower leg': 10,
    'right back upper leg': 11, 'right back lower leg': 12, 'tail': 13,
    'left back hoof': 14, 'right back hoof': 15}

person_part_labels = {
    'head': 0, 'torso': 1, 'neck': 2, 'left lower arm': 3, 'left upper arm': 4,
    'left hand': 5, 'right lower arm': 6, 'right upper arm': 7,
    'right hand': 8, 'left lower leg': 9, 'left upper leg': 10,
    'left foot': 11, 'right lower leg': 12, 'right upper leg': 13,
    'right foot': 14
}


ALL_PART_MAPPING = {
    "bird": bird_part_labels,
    "cat": cat_part_labels,
    "cow": cow_part_labels,
    "sheep": cow_part_labels,
    "dog": dog_part_labels,
    "horse": horse_part_labels,
    "person": person_part_labels,
}

_bird_part_labels = {
    'head': 0, 'torso': 1, 'neck': 2, 'lwing': 3, 'rwing': 4,
    'lleg': 5, 'lfoot': 6, 'rleg': 7, 'rfoot': 8, 'tail': 9
}

_cat_part_labels = {
    'head': 0, 'torso': 1, 'neck': 2, 'lfleg': 3, 'lfpa': 4, 'rfleg': 5,
    'rfpa': 6, 'lbleg': 7, 'lbpa': 8, 'rbleg': 9, 'rbpa': 10, 'tail': 11}

_cow_part_labels = {
    'head': 0, 'lhorn': 1, 'rhorn': 2, 'torso': 3, 'neck': 4, 'lfuleg': 5,
    'lflleg': 6,'rfuleg': 7, 'rflleg': 8, 'lbuleg': 9, 'lblleg': 10,
    'rbuleg': 11, 'rblleg': 12, 'tail': 13}

_dog_part_labels = {
    'head': 0, 'torso': 1, 'neck': 2, 'lfleg': 3, 'lfpa': 4, 'rfleg': 5,
    'rfpa': 6, 'lbleg': 7, 'lbpa': 8, 'rbleg': 9, 'rbpa': 10, 'tail': 11, 'muzzle': 12}

_horse_part_labels = {
    'head': 0, 'lfho': 1, 'rfho': 2, 'torso': 3, 'neck': 4, 'lfuleg': 5,
    'lflleg': 6, 'rfuleg': 7, 'rflleg': 8, 'lbuleg': 9, 'lblleg': 10,
    'rbuleg': 11, 'rblleg': 12, 'tail': 13, 'lbho': 14, 'rbho': 15}

_person_part_labels = {
    'head': 0, 'torso': 1, 'neck': 2, 'llarm': 3, 'luarm': 4, 'lhand': 5, 
    'rlarm': 6,'ruarm': 7, 'rhand': 8, 'llleg': 9, 'luleg': 10, 'lfoot': 11, 
    'rlleg': 12, 'ruleg': 13, 'rfoot': 14}
