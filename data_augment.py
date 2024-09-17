"""Data augmentation script with multiple augmentation options."""

import argparse

import gc
import os
import sys
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd


def parse_args(input_args=None):

    parser = argparse.ArgumentParser(
        description="Data converter for augmenting Layout to Image finetuning dataset.",
    )

    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        required=True,
        help="Path to dataset that has to be augmented."

    )

    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        required=True,
        help="Splits to augment."
    )

    parser.add_argument(
        "--image_folders",
        type=str,
        nargs="+",
        default=None,
        required=False,
        help="List of image_folders to augment."
    )

    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=True,
        help=("List of factor by which the image has to be "
              "upscaled or downscaled")
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help=("Base path where the augmented data is stored. "
        "This path will")
    )


    if input_args is not None:
        return parser.parse_args(input_args)
    else:
        return parser.parse_args()


def augment_and_save(
        image,
        scale,
        output_path,
        pad=True,
        add_mask=False,
        mask_path=None,
    ):

    # Due to change in input size for controlnet
    scaling_constant = 512/660

    old_height, old_width, channels = image.shape
    scaling_factor = 1 + scale
    new_width = int(old_width * scaling_factor * scaling_constant)
    new_height = int(old_height * scaling_factor * scaling_constant)

    if scaling_factor != 0.0:
        resized_image = cv2.resize(image, (new_width, new_height))
    else:
        resized_image = image

    if pad:
        controlnet_image_height = 512
        controlnet_image_width = 512
        result = np.full(
            (controlnet_image_height,
             controlnet_image_width,
             channels),
            (0, 0, 0),
            dtype=np.uint8
        )

        # compute center offset
        x_center = (controlnet_image_width - new_width) // 2
        y_center = (controlnet_image_height - new_height) // 2

        # copy img image into center of result image
        result[y_center:y_center+new_height,
            x_center:x_center+new_width] = resized_image

        if add_mask:
            mask = np.full(
                (controlnet_image_height,
                controlnet_image_width,
                channels),
                (0, 0, 0),
                dtype=np.uint8
            )
            mask[y_center:y_center+new_height,
                x_center:x_center+new_width] =  np.full(
                    (new_height, new_width, channels),
                    (255, 255, 255),
                    dtype=np.uint8
            )

            
            cv2.imwrite(mask_path, mask)


    else:
        result =  resized_image

    cv2.imwrite(output_path, result)


def main(input_args):

    base_path = input_args.base_path
    output_path = input_args.output_path
    scales = [0.0, ] + input_args.scales

    print(f"Reading images from {base_path}")
    print(f"For splits: {input_args.splits}")
    print(f"From these subfolders: {input_args.image_folders}")
    print("Augmentation will be done for these scales: "
          f"{input_args.scales}")

    os.makedirs(output_path, exist_ok=True)

    caption_files = [
        f for f in os.listdir(base_path) if (
            os.path.isfile(os.path.join(base_path, f)) 
            and (f != ".DS_Store"))]

    for caption_file in caption_files:
        caption_data = pd.read_csv(
            os.path.join(base_path, caption_file)
        )
        all_augmented_captions = []
        for idx in range(len(scales)):
            caption_data_scaled = caption_data[
                ['file_name',
                    'caption_basic',
                    'caption_artsy',
                    'caption_wt_parts']].copy()
            caption_data_scaled[
                'file_name'
            ] = (
                caption_data_scaled[
                'file_name'].str.split('.', expand=True)[0]
                + str(idx) + "000."
                + 'jpg')
            all_augmented_captions.append(
                caption_data_scaled)
        combined_captions = pd.concat(
            all_augmented_captions).reset_index(drop=True)
        combined_captions['caption_artsy'] = combined_captions[
            'caption_artsy'].str.replace('andbody', 'and body')
        combined_captions.to_csv(
            os.path.join(output_path, caption_file))       

    for image_folder in input_args.image_folders:

        for split in input_args.splits:

            current_output_path = os.path.join(
                output_path,
                image_folder,
                split
            )

            os.makedirs(current_output_path, exist_ok=True)
            source_path =  os.path.join(
                base_path, image_folder, split)
            all_images = os.listdir(source_path)

            for image_path in tqdm(all_images):

                img_idx, _ = image_path.split(".")

                for idx, scale in enumerate(scales):

                    new_filename = (
                        img_idx + str(idx) + "000." + "jpg")
                    output_file_name = os.path.join(
                        current_output_path, new_filename)

                    if image_folder == "image":
                        mask_path = current_output_path.replace(
                            "/image/", "/mask/")
                        os.makedirs(mask_path, exist_ok=True)
                        mask_filename = os.path.join(
                            mask_path,
                            new_filename)
                        add_mask = True
                    else:
                        add_mask = False
                        mask_filename=None
                    image = cv2.imread(os.path.join(
                        source_path, image_path))
                    augment_and_save(
                        image,
                        scale,
                        output_path=output_file_name,
                        mask_path=mask_filename,
                        add_mask=add_mask
                    )


if __name__=="__main__":
    args = parse_args()
    main(args)
