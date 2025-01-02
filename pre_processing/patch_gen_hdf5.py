# This script generates pathes from  whole slide images (e.g. from TCGA) and save the extracted patches into a hdf5 file.
# This will save all the paches but keeps the file number small.

import pandas as pd
import numpy as np
from openslide import OpenSlide
from multiprocessing import Pool, Value, Lock
import os
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.io import imsave, imread
from skimage.exposure.exposure import is_low_contrast
from skimage.transform import resize
from scipy.ndimage import binary_dilation, binary_erosion
import argparse
import logging
import h5py
from tqdm import tqdm

import pickle
import re
import pdb
import pandas as pd

def get_mask_image(img_RGB, RGB_min=50):
    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return mask

def get_mask(slide, level='max', RGB_min=50):
    #read svs image at a certain level  and compute the otsu mask
    if level == 'max':
        level = len(slide.level_dimensions) - 1
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_RGB = np.transpose(np.array(slide.read_region((0, 0),level,slide.level_dimensions[level]).convert('RGB')),
                           axes=[1, 0, 2])

    tissue_mask = get_mask_image(img_RGB, RGB_min)
    return tissue_mask, level

def extract_patches(slide_path, mask_path, patch_size, patches_output_dir, slide_id, max_patches_per_slide=2000):

    patch_folder = os.path.join(patches_output_dir, slide_id)
    if not os.path.isdir(patch_folder):
        os.makedirs(patch_folder)

    patch_folder_mask = os.path.join(mask_path, slide_id)
    if not os.path.isdir(patch_folder_mask):
        os.makedirs(patch_folder_mask)

    if os.path.exists(os.path.join(patch_folder, "complete.txt")):
        print(f'{slide_id}: patches have already been extreacted')
        return

    path_hdf5 = os.path.join(patch_folder, f"{slide_id}.hdf5")
    hdf = h5py.File(path_hdf5, 'w')

    slide = OpenSlide(slide_path)
    mask, mask_level = get_mask(slide)
    mask = binary_dilation(mask, iterations=3)
    mask = binary_erosion(mask, iterations=3)
    np.save(os.path.join(patch_folder_mask, "mask.npy"), mask)

    mask_level = len(slide.level_dimensions) - 1

    PATCH_LEVEL = 0
    BACKGROUND_THRESHOLD = .2

    try:
        ratio_x = slide.level_dimensions[PATCH_LEVEL][0] / slide.level_dimensions[mask_level][0]
        ratio_y = slide.level_dimensions[PATCH_LEVEL][1] / slide.level_dimensions[mask_level][1]

        xmax, ymax = slide.level_dimensions[PATCH_LEVEL]

        # handle slides with 40 magnification at base level
        resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
        if not slide.properties.get('aperio.AppMag', 20): print(f"magnifications for {slide_id} is not found, using default magnification 20X")

        patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))
        print(f"patch size for {slide_id}: {patch_size_resized}")

        i = 0
        indices = [(x, y) for x in range(0, xmax, patch_size_resized[0]) for y in
                    range(0, ymax, patch_size_resized[0])]

        # here, we generate all the pathes with valid mask
        if max_patches_per_slide is None:
            max_patches_per_slide = len(indices)

        np.random.seed(5)
        np.random.shuffle(indices)

        for x, y in indices:
            # check if in background mask
            x_mask = int(x / ratio_x)
            y_mask = int(y / ratio_y)
            if mask[x_mask, y_mask] == 1:
                patch = slide.read_region((x, y), PATCH_LEVEL, patch_size_resized).convert('RGB')
                try:
                    mask_patch = get_mask_image(np.array(patch))
                    mask_patch = binary_dilation(mask_patch, iterations=3)
                except Exception as e:
                    print("error with slide id {} patch {}".format(slide_id, i))
                    print(e)
                if (mask_patch.sum() > BACKGROUND_THRESHOLD * mask_patch.size) and not (is_low_contrast(patch)):
                    if resize_factor != 1.0:
                        patch = patch.resize(patch_size)
                    patch = np.array(patch)
                    tile_name = f"{x}_{y}"
                    hdf.create_dataset(tile_name, data=patch)
                    i = i + 1
            if i >= max_patches_per_slide:
                break

        hdf.close()

        if i == 0:
            print("no patch extracted for slide {}".format(slide_id))
        else:
            with open(os.path.join(patch_folder, "complete.txt"), 'w') as f:
                f.write('Process complete!\n')
                f.write(f"Total n patch = {i}")
                print(f"{slide_id} complete, total n patch = {i}")

    except Exception as e:
        print("error with slide id {} patch {}".format(slide_id, i))
        print(e)

def get_slide_id(slide_name):
    return slide_name.split('.')[0]

def process(opts):
    slide_path, patch_size, patches_output_dir, mask_path, slide_id, max_patches_per_slide = opts
    extract_patches(slide_path, mask_path, patch_size,
                    patches_output_dir, slide_id, max_patches_per_slide)


parser = argparse.ArgumentParser(description='Generate patches from a given folder of images')
parser.add_argument('--ref_file', default="examples/ref_file.csv", required=False, metavar='ref_file', type=str,
                    help='Path to the ref_file, if provided, only the WSIs in the ref file will be processed')
parser.add_argument('--wsi_path', default="examples/HE", metavar='WSI_PATH', type=str,
                    help='Path to the input directory of WSI files')
parser.add_argument('--patch_path', default="examples/Patches_hdf5" ,metavar='PATCH_PATH', type=str,
                    help='Path to the output directory of patch images')
parser.add_argument('--mask_path', default="examples/Patches_hdf5", metavar='MASK_PATH', type=str,
                    help='Path to the  directory of numpy masks')
parser.add_argument('--patch_size', default=256, type=int, help='patch size, '
                                                                'default 256')
parser.add_argument('--start', type=int, default=0,
                    help='Start slide index for parallelization')
parser.add_argument('--end', type=int, default=None,
                    help='End slide index for parallelization')
parser.add_argument('--max_patches_per_slide', default=None, type=int)
parser.add_argument('--debug', default=0, type=int,
                    help='whether to use debug mode')
parser.add_argument('--parallel', default=1, type=int,
                    help='whether to use parallel computation')


if __name__ == '__main__':

    args = parser.parse_args()
    slide_list = os.listdir(args.wsi_path)
    slide_list = [s for s in slide_list if s.endswith('.svs') or s.endswith('.tiff')]

    if args.ref_file:
        ref_file = pd.read_csv(args.ref_file)
        selected_slides = list(ref_file['wsi_file_name'])
        wsi_files = [f'{s}.svs' for s in selected_slides]
        slide_list = list(set(slide_list) & set(wsi_files))
        slide_list = sorted(slide_list)

    if args.start is not None and args.end is not None:
        slide_list = slide_list[args.start:args.end]
    elif args.start is not None:
        slide_list = slide_list[args.start:]
    elif args.end is not None:
        slide_list = slide_list[:args.end]

    if args.debug:
        slide_list = slide_list[0:5]
        args.max_patches_per_slide = 20

    print(f"Found {len(slide_list)} slides")

    opts = [
        (os.path.join(args.wsi_path, s), (args.patch_size, args.patch_size), args.patch_path, args.mask_path,
        get_slide_id(s), args.max_patches_per_slide) for
        (i, s) in enumerate(slide_list)]

    if args.parallel:
        pool = Pool(processes=4)
        pool.map(process, opts)
    else:
        for opt in opts:
            process(opt)


