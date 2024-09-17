"""Executable for creating conditioning images for Controlnet Fine-tuning."""
import argparse

import gc
import os
import sys
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd

sys.path.append("..")

from utils import constants, plotting, prompts


def parse_args(input_args=None):

    parser = argparse.ArgumentParser(
        description="Data converter for creating a Layout to Image finetuning dataset.",
    )

    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        required=True,
        help="Path to dataset already updated to huggingface."
    )

    parser.add_argument(
        "--add_parts_in_prompt",
        type=bool,
        default=False,
        required=None,
        help="Whether to add part names in the prompt."
    )

    parser.add_argument(
        "--add_layout_as_bbx",
        type=bool,
        required=False,
        help="Whether to create conditioning image with bounding boxes in layout."
    )

    parser.add_argument(
        "--add_layout_as_ellipse",
        type=bool,
        required=False,
        help="Whether to creating conditioning image with ellipses in layout"
    )

    parser.add_argument(
        "--add_layout_as_distribution",
        type=bool,
        required=False,
        help="Whether to creating conditioning image with a distribution in layout"
    )

    parser.add_argument(
        "--output_file_path",
        type=str,
        required=True,
    )

    if input_args is not None:
        return parser.parse_args(input_args)
    else:
        return parser.parse_args()

    # TODO: Add flag validations


def plot_layout_and_save(
        layout, plotter, idx, tag, output_file_path):        

    layout_plot = plotter(layout)

    if (layout_plot.shape[0]==0) or layout_plot.shape[1]==0:
        return
    try:
        cv2.imwrite(
            f"{output_file_path}/{idx}{tag}.png",
            cv2.cvtColor(layout_plot, cv2.COLOR_RGB2BGR)
        )
        del layout_plot
        gc.collect()

    except Exception as e:
        print(e, layout_plot.shape)


def write_data(data_specs, args):

    scale, tag, layouts, split, obj_classes, obj_images = data_specs

    os.makedirs(f"{args.output_file_path}image/{split}/", exist_ok=True)

    files = [f"{idx}{tag}.jpg" for idx in range(len(obj_classes))]
    captions_basic = [
        prompts.get_basic_prompt(constants.CLASS_DICT[np.argmax(obj_class)])
        for obj_class in obj_classes
    ]
    captions_artsy = [
        prompts.get_styled_prompt(
            constants.CLASS_DICT[np.argmax(obj_class)]
        )
        for obj_class in obj_classes
    ]

    captions_with_parts = [
        prompts.get_prompt_wt_parts(
            constants.CLASS_DICT[np.argmax(obj_class)],
            layout
        )
        for obj_class, layout in zip(obj_classes, layouts)
    ]

    class_data = pd.DataFrame({
        'file_name': files,
        'caption_basic': captions_basic,
        'caption_artsy': captions_artsy,
        'caption_wt_parts': captions_with_parts,
    })
    class_data.to_csv(
        f"{args.output_file_path}{split}_metadata{tag}.csv"
    )
    del class_data
    del files
    del captions_basic
    del captions_artsy

    for idx, (layout, image) in tqdm(enumerate(
        zip(layouts, obj_images))):

        image = image.astype(np.uint8)
        if scale != 0:
            height, width, _ = image.shape
            scaled_height, scaled_width = (
                int(height*(1+scale)), int(width*(1+scale))
            )
            if ((scaled_height > constants.CANVAS_SIZE) 
                or (scaled_width > constants.CANVAS_SIZE)):
                return

            image = cv2.resize(
                image,
                (scaled_height, scaled_height),
                interpolation=cv2.INTER_AREA
            )

        cv2.imwrite(
            f"{args.output_file_path}image/{split}/{idx}{tag}.jpg",
            image
        )
        del image

        if args.add_layout_as_bbx:
            output_file_path = (
                f"{args.output_file_path}layout_bbx_image/{split}/"
            )
            os.makedirs(output_file_path, exist_ok=True)
            plotter = plotting.plot_bbx
            plot_layout_and_save(
                layout=layout,
                plotter=plotter,
                idx=idx,
                tag=tag,
                output_file_path=output_file_path,
            )
        if args.add_layout_as_ellipse:
            output_file_path = (
                f"{args.output_file_path}layout_ellipse_image/{split}/"
            )
            os.makedirs(output_file_path, exist_ok=True)
            plotter = plotting.plot_ellipse
            plot_layout_and_save(
                layout=layout,
                plotter=plotter,
                idx=idx,
                tag=tag,
                output_file_path=output_file_path,
            )
        if args.add_layout_as_distribution:
            output_file_path = (
                f"{args.output_file_path}"
                f"layout_distribution_image/{split}/"
            )
            os.makedirs(output_file_path, exist_ok=True)
            plotter = plotting.plot_distributions
            plot_layout_and_save(
                layout=layout,
                plotter=plotter,
                idx=idx,
                tag=tag,
                output_file_path=output_file_path,
            )
        gc.collect()


def main(args):

    data_root = args.base_path

    x_train_path = "X_train_combined_mask_data.np"
    obj_class_train_path = "class_v_combined_mask_data.np"
    images_train_path = "img_train_combined_mask_data.np"

    x_test_path = "X_test_combined_mask_data.np"
    obj_class_test_path = "class_v_test_combined_mask_data.np"
    images_test_path = "img_test_combined_mask_data.np"

    x_val_path = "X_val_combined_mask_data.np"
    obj_class_val_path = "class_v_val_combined_mask_data.np"
    images_val_path = "img_val_combined_mask_data.np"

    x_train = np.load(data_root + x_train_path, allow_pickle=True)
    obj_class_train = np.load(data_root + obj_class_train_path,
                              allow_pickle=True)
    images_train = np.load(data_root + images_train_path,
                           allow_pickle=True)

    x_test = np.load(data_root + x_test_path,
                     allow_pickle=True)
    obj_class_test = np.load(data_root + obj_class_test_path,
                             allow_pickle=True)
    images_test = np.load(data_root + images_test_path,
                          allow_pickle=True)

    x_val = np.load(data_root + x_val_path, allow_pickle=True)
    obj_class_val = np.load(data_root + obj_class_val_path,
                            allow_pickle=True)
    images_val = np.load(data_root + images_val_path,
                         allow_pickle=True)

    x_train[:, :, 1:] *= constants.CANVAS_SIZE
    train_bbxs = [x_train] #= list(plotting.scale(x_train, 0.2))
    train_bbxs.append(x_train.copy())

    x_test[:, :, 1:] *= constants.CANVAS_SIZE
    test_bbxs = [x_test] # list(plotting.scale(x_test, 0.2))
    test_bbxs.append(x_test.copy())

    x_val[:, :, 1:] *= constants.CANVAS_SIZE
    val_bbxs = [x_val] # list(plotting.scale(x_val, 0.2))
    val_bbxs.append(x_val.copy())

    scales = [0.0, 0.0, 0.0] # [0.2, -0.2, 0.1, -0.1]
    tags =  ["", "", ""] # ["1000", "2000", "3000", "4000"]
    bbxs = [train_bbxs[-1], test_bbxs[-1], val_bbxs[-1]]
    splits = ["train", "test", "val"]

    obj_classes = [obj_class_train, obj_class_test, obj_class_val]
    obj_images = [images_train, images_test, images_val]

    data_tuples = list(
        zip(scales, tags, bbxs, splits, obj_classes, obj_images))

    for data_spec in list(data_tuples):
        write_data(data_spec, args)


if __name__=="__main__":
    args = parse_args()
    main(args)