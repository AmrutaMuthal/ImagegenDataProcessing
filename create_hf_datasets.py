"""Creats conditioning datasets for training."""

import argparse

from datasets import load_dataset, concatenate_datasets
from sympy import false


def parse_args(input_args=None):

    parser = argparse.ArgumentParser(
        description="Data converter for augmenting Layout to Image finetuning dataset.",
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        required=True,
        help="Path to images folder that has to be augmented."
    )

    parser.add_argument(
        "--conditioning_path",
        type=str,
        default=None,
        required=False,
        help="Path to conditioning folder that has to be augmented."
    )

    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        required=False,
        help="Path to masks folder that has to be augmented."
    )

    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        required=True,
        help="Name of splits to process."
    )

    parser.add_argument(
        "--hf_write_auth_token",
        type=str,
        required=True,
        help="Auth token to enable writing the dataset to HuggingFace."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=False,
        help=("Base path where the augmented data is stored. This path should "
              "end with the prefix for the final dataset name ")
    )


    if input_args is not None:
        return parser.parse_args(input_args)
    else:
        return parser.parse_args()


def create_conditoning_dataset(
        image_path,
        conditioning_path=None,
        mask_path=None,
        splits=None,
        output_path=None,
        hf_write_auth_token=None,
):

    controlnet_image = load_dataset(path = image_path).remove_columns(
        'Unnamed: 0')
    controlnet_conditioning_image = load_dataset(
        path = conditioning_path).rename_column(
            'image', 'conditioning_image').remove_columns('label')
    controlnet_mask_image = load_dataset(path = mask_path).rename_column(
        'image', 'mask_image').remove_columns('label')
    print(controlnet_image.column_names)
    print(controlnet_image['train'][0])
    print(controlnet_conditioning_image.column_names)
    print(controlnet_conditioning_image['train'][0])
    print(controlnet_mask_image.column_names)
    print(controlnet_mask_image['train'][0])

    all_sets = {}

    for split in splits:
        complete_controlnet_dataset = concatenate_datasets(
            [
                controlnet_image[split],
                controlnet_conditioning_image[split],
                controlnet_mask_image[split]
            ],
            axis=1,
            split=split
        )
        print(complete_controlnet_dataset[0])

        all_sets[split] = complete_controlnet_dataset

        complete_controlnet_dataset.push_to_hub(f"{output_path}_{split}",
                                                  token=hf_write_auth_token)

    if ('train' in splits) and ('test' in splits):
        complete_controlnet_dataset = concatenate_datasets(
            [all_sets['train'], all_sets['test']],
            axis=0,
            split='train'
        )
    
        complete_controlnet_dataset.push_to_hub(f"{output_path}_test_train",
                                                token=hf_write_auth_token)


def main(input_args):

    create_conditoning_dataset(
        input_args.image_path,
        input_args.conditioning_path,
        input_args.mask_path,
        input_args.splits,
        input_args.output_path,
        input_args.hf_write_auth_token,
    ) 


if __name__=="__main__":
    args = parse_args()
    main(args)