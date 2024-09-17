"""Contains plotting functions for generating conditioning images."""

from typing import List
import sys

from PIL import Image
import cv2
import numpy as np

sys.path.append("..")

from . import constants


def plot_bbx(
        bbx: np.matrix,
        scale_to_canvas:bool=False,
        invert_order: bool=False
    ) -> np.matrix:
    """Create a layout bounding box conditioning image from input bounding 
    boxes.

    Args:
        bbx (np.matrix): Layout bounding box
        scale_to_canvas (bool, optional): If bounding boxes need to be scale
            up to the canvas size. Defaults to False.
        invert_order (bool, optional): Should the order of plotting bounding 
            boxes be reversed

    Returns:
        np.matrix: matrix containing the image of layout bounding boxes.
    """
    if scale_to_canvas:
        bbx = bbx*constants.CANVAS_SIZE

    canvas = np.zeros(
        (constants.CANVAS_SIZE, constants.CANVAS_SIZE, 3), np.uint8) * 255
    obj_x_min = obj_y_min = constants.CANVAS_SIZE
    obj_x_max = obj_y_max = 0

    if invert_order:
        bbx = bbx[::-1]

    for i, coord in enumerate(bbx):

        has_part, x_minp, y_minp, x_maxp, y_maxp = coord
        if has_part:
            obj_x_min = min(obj_x_min, x_minp)
            obj_y_min = min(obj_y_min, y_minp)
            obj_x_max = max(obj_x_max, x_maxp)
            obj_y_max = max(obj_y_max, y_maxp)
            cv2.rectangle(
                canvas,
                (int(x_minp), int(y_minp)),
                (int(x_maxp), int(y_maxp)),
                constants.COLORS[i].tolist(),
                -1
            )

    return canvas[
        int(obj_y_min):int(obj_y_max+1),
        int(obj_x_min):int(obj_x_max+1)
    ]


def plot_ellipse(
        bbx: np.matrix,
        scale_to_canvas:bool=False,
        invert_order: bool=False
) -> np.matrix:
    """Create a layout of ellipses as conditioning image from input bounding 
    boxes.

    Args:
        bbx (np.matrix): Layout bounding box
        scale_to_canvas (bool, optional): If bounding boxes need to be scale
            up to the canvas size. Defaults to False.
        invert_order (bool, optional): Should the order of plotting bounding 
            boxes be reversed

    Returns:
        np.matrix: matrix containing the image of layout bounding boxes.
    """
    if scale_to_canvas:
        bbx = bbx*constants.CANVAS_SIZE

    if invert_order:
        bbx = bbx[::-1]

    canvas = np.zeros(
        (constants.CANVAS_SIZE, constants.CANVAS_SIZE, 3), np.uint8) * 255
    obj_x_min = obj_y_min = constants.CANVAS_SIZE
    obj_x_max = obj_y_max = 0

    for i, coord in enumerate(bbx):

        has_part, x_minp, y_minp, x_maxp, y_maxp = coord
        if has_part:
            obj_x_min = min(obj_x_min, x_minp)
            obj_y_min = min(obj_y_min, y_minp)
            obj_x_max = max(obj_x_max, x_maxp)
            obj_y_max = max(obj_y_max, y_maxp)

            x_center = (x_minp + x_maxp)/2
            y_center = (y_minp + y_maxp)/2
            x_len = max(x_maxp - x_minp, 1)
            y_len = max(y_maxp - y_minp, 1)

            cv2.ellipse(
                canvas,
                (int(x_center), int(y_center)),
                (int(x_len/2), int(y_len/2)),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=constants.COLORS[i].tolist(),
                thickness=-1)

    return canvas[
        int(obj_y_min):int(obj_y_max+1),
        int(obj_x_min):int(obj_x_max+1)]


def plot_distributions(
        bbx: np.matrix,
        scale_to_canvas:bool=False,
        invert_order: bool=False
) -> np.matrix:
    """Create a layout as 2D distributions as conditioning image from input
    bounding boxes dimensions.

    Each distribution is centered at the bounding box center and the covariance 
    is both the directions is propoostional to the bounding box width in that direction.

    Args:
        bbx (np.matrix): Layout bounding box
        scale_to_canvas (bool, optional): If bounding boxes need to be scale
            up to the canvas size. Defaults to False.
        invert_order (bool, optional): Should the order of plotting bounding 
            boxes be reversed

    Returns:
        np.matrix: matrix containing the image of layout of distributions.
    """
    if scale_to_canvas:
        bbx = bbx*constants.CANVAS_SIZE

    if invert_order:
        bbx = bbx[::-1]

    strength = 0.02
    canvas = np.zeros(
        (constants.CANVAS_SIZE, constants.CANVAS_SIZE, 3), np.uint8) * 255
    obj_x_min = obj_y_min = constants.CANVAS_SIZE
    obj_x_max = obj_y_max = 0

    for i, coord in enumerate(bbx):

        has_part, x_min, y_min, x_max, y_max = coord
        if has_part:
            obj_x_min = min(obj_x_min, x_min)
            obj_y_min = min(obj_y_min, y_min)
            obj_x_max = max(obj_x_max, x_max)
            obj_y_max = max(obj_y_max, y_max)

            x_center = (x_min + x_max)/2
            y_center = (y_min + y_max)/2
            x_len = max(x_max - x_min, 1)
            y_len = max(y_max - y_min, 1)

            temp = np.random.multivariate_normal(
                [y_center, x_center],
                [
                    [(y_len**2) * strength, 0],
                    [0, (x_len**2 * strength)]
                ],
                int(120 * x_len * y_len)
                )
            temp_filtered = temp[(
                (temp[:, 0] >= y_min)
                & (temp[:, 0] <= y_max)
                & (temp[:, 1] >= x_min)
                & (temp[:, 1] <= x_max)
            )]

            canvas[
                (temp_filtered[:, 0]).astype(int),
                (temp_filtered[:, 1]).astype(int)] = constants.COLORS[i]

    return canvas[
        int(obj_y_min):int(obj_y_max+1),
        int(obj_x_min):int(obj_x_max+1)]


def plot_obj_bbx(bbx:np.matrix) -> np.matrix:
    """Generates an image of object bounding box.

    Args:
        bbx (np.matrix): NumPy matrix containing the boundig box cooardinates.

    Returns:
        np.matrix: Numpy matrix representing the image of plotted bounding box.
    """
    bbx = bbx*constants.CANVAS_SIZE
    canvas = np.ones(
        (constants.CANVAS_SIZE, constants.CANVAS_SIZE, 3), np.uint8) * 255
    x_minp, y_minp, x_maxp, y_maxp = bbx
    cv2.rectangle(canvas, (int(x_minp), int(y_minp)),
                  (int(x_maxp), int(y_maxp)), constants.COLORS[0], -1)
    return canvas


def centre_object(bbx: np.matrix, canvas_size: int) -> np.matrix:
    """Centers the object on the canvas image.

    The input bounding boxes are expected to be scaled to the canvas size

    Args:
        bbx (np.matrix): Numpy matrix containing all the layout bounding boxes.
            canvas_size (int): Size of the canvas in pixel. Generally expected 
            to be 660.

    Returns:
        np.matrix: Numpy matrix contiaing centered layout bounding boxes.
    """

    pos = np.repeat(bbx[:, :, :1] == 1, 5, axis=-1)
    bx = np.copy(bbx)
    h, w = canvas_size
    bbx_min = bbx
    bbx_min[pos] = h
    h_o = (np.max(bbx[:, :, 4], axis=0) + np.min(bbx_min[:, :, 2], axis=0))
    w_o = (np.max(bbx[:, :, 3], axis=0) + np.min(bbx_min[:, :, 1], axis=0))

    h_shift = (h/2 - h_o/2).astype(int)
    w_shift = (w/2 - w_o/2).astype(int)

    bx[:, :, 1] = (bx[:, :, 1]+w_shift)
    bx[:, :, 2] = (bx[:, :, 2]+h_shift)
    bx[:, :, 3] = (bx[:, :, 3]+w_shift)
    bx[:, :, 4] = (bx[:, :, 4]+h_shift)

    return bx*pos


def scale(bbx: np.matrix, scaling_factor: float) -> List[np.matrix]:
    """Generates upscaled and downscaled versions of the input layout.

    In total returns 4 variations. Upscaled and downscaled version at specified 
    scale factor and half scale factor.

    Args:
        bbx (np.matrix): _description_
        scaling_factor (float): _description_

    Returns:
        np.matrix: _description_
    """

    pos = bbx[:, :, :1]

    fold_a = np.copy(bbx)
    fold_b = np.copy(bbx)
    fold_c = np.copy(bbx)
    fold_d = np.copy(bbx)

    center_shift_pos = constants.CANVAS_SIZE*(scaling_factor)
    center_shift_neg = constants.CANVAS_SIZE*(-scaling_factor)

    fold_a[:, :, 1:] = fold_a[:, :, 1:]*(1-scaling_factor) + center_shift_pos
    fold_a = centre_object(
        fold_a*pos, (constants.CANVAS_SIZE, constants.CANVAS_SIZE))
    fold_b[:, :, 1:] = fold_b[:, :, 1:]*(1+scaling_factor) + center_shift_neg
    fold_b = centre_object(
        fold_b*pos, (constants.CANVAS_SIZE, constants.CANVAS_SIZE))

    center_shift_pos = constants.CANVAS_SIZE*(scaling_factor*0.5)
    center_shift_neg = constants.CANVAS_SIZE*(-scaling_factor*0.5)

    fold_c[:, :, 1:] = fold_c[:, :, 1:] * \
        (1-scaling_factor*0.5) + center_shift_pos
    fold_c = centre_object(
        fold_c*pos, (constants.CANVAS_SIZE, constants.CANVAS_SIZE))
    fold_d[:, :, 1:] = fold_d[:, :, 1:] * \
        (1+scaling_factor*0.5) + center_shift_neg
    fold_d = centre_object(
        fold_d*pos, (constants.CANVAS_SIZE, constants.CANVAS_SIZE))

    return fold_a*pos, fold_b*pos, fold_c*pos, fold_d*pos


def binarize(img: Image) -> Image:
    """Generates a binarized image from the input.

    Args:
        img (PIL.Image): Given Image

    Returns:
        PIL.Image: Binarized image
    """

    img = np.mean(img, axis=-1)
    return np.where(img>1, 255, 0)


def binarize_to_object(img: Image) -> Image:
    """
    Generates a binarized image where the light portion represents the smallest 
    bounding box enclosing all non zero pixels.

    Args:
        img (PIL.Image): Input colour or grascale image.

    Returns:
        PIL.Image: Binarized image of the enclosing bounding box.
    """
    # img = img.convert("L")
    # gray = img.point(lambda x: 0 if x < 1 else 255, '1')
    # gray_matrix = np.where(np.array(gray))
    # x_max, y_max = np.max(gray_matrix, axis=1)
    # x_min, y_min = np.min(gray_matrix, axis=1)
    # obj_mask = np.zeros(gray.size)
    # obj_mask[x_min: x_max+1, y_min: y_max+1] = 255
    # return Image.fromarray(obj_mask)

    img = np.mean(img, axis=-1)
    gray = np.where(img > 1, 255, 0)
    gray_matrix = np.where(np.array(gray))
    x_max, y_max = np.max(gray_matrix, axis=1)
    x_min, y_min = np.min(gray_matrix, axis=1)
    obj_mask = np.zeros(gray.shape)
    obj_mask[x_min: x_max+1, y_min: y_max+1] = 255
    return obj_mask


def pad_cnet_old(image: Image) -> Image:
    """Pads the object or layout image to controlnet size of 512.

    Args:
        image (Image): Input image arbitary size

    Returns:
        Image: Padded image of size 512, 512
    
    Raises:
        Expection when image overflows 512X512 size.
    """
    result = Image.new(
        image.mode,
        (constants.CANVAS_SIZE, constants.CANVAS_SIZE),
        'black'
    )
    try:
        result.paste(image,
                     (int((constants.CANVAS_SIZE-image.shape[0])//2),
                      int((constants.CANVAS_SIZE-image.shape[1])//2)))
        return result.resize((512, 512))
    except Exception as e:
        print(e,
            "Image size: ",
            image.shape, int((constants.CANVAS_SIZE-image.shape[0])//2),
            int((constants.CANVAS_SIZE-image.shape[1])//2)
        )
        return

def pad_cnet(image: np.matrix) -> Image:
    """Pads the object or layout image to controlnet size of 512.

    Args:
        image (np.matrix): Input image arbitary size

    Returns:
        np.matrix: Padded image of size 512, 512
    
    Raises:
        Expection when image overflows 512X512 size.
    """
    old_image_height, old_image_width, channels = image.shape

    # create new image of desired size and color (blue) for padding
    new_image_width = constants.CANVAS_SIZE
    new_image_height = constants.CANVAS_SIZE
    color = (0,0,0) # Black background
    result = np.full(
        (new_image_height, new_image_width, channels),
        color,
        dtype=np.uint8
    )

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height,
        x_center:x_center+old_image_width] = image

    return result
