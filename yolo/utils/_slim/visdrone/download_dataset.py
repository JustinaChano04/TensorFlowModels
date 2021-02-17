"""
This file is the main file that handles downloading and processing the dataset.
The rest of the folder is a TensorFlow wrapper for this file.

Documentation of VisDrone annotation format: http://aiskyeye.com/evaluate/results-format/
Documentation of DarkNet annotation format: https://github.com/AlexeyAB/Yolo_mark/issues/60#issuecomment-401854885

NOTE: On Colab, this conversion takes about 2-3 days of CPU execution. This is
more than Colab allows for running a cell. Leave the browser open and restart
the cell every 12 hours to bypass this limit.

You may also get I/O timeout errors because the file list is not yet loaded onto
the Colab server from the drive. Run the following command a couple of times
until it no longer gives an error and then you can proceed to process the
dataset or train the model.

!find VisDrone2018-DET-train/data -name '*.txt' | head -n 10

Loading the dataset onto the Colab server usually takes 30 minutes.
"""

import os
import glob
import pathlib
import shutil
import zipfile

import cv2
import gdown

try:
    from tqdm import tqdm
except:
    def tqdm(iterator):
        return iterator


# Recieved from https://www.jianshu.com/p/62e827306fca
FOLDERS = [
    ['VisDrone2018-DET-train', 'https://drive.google.com/uc?id=1DfD2EgzIj_ZiL1rRwr1TuG1Od_CwFtGs'],
    ['VisDrone2018-DET-val', 'https://drive.google.com/uc?id=1-WvQAfB0TdrLqMnop5fXHe24YtRlUsmb'],
    ['VisDrone2018-DET-test-dev', 'https://drive.google.com/uc?id=1VzRQGVHt0aF4rn5xYjrNddYfIurrnGp1']
]


def convert_labels_to_darknet(img_width: int, img_height: int, visdrone_annotations: tuple):
    """
    This function handles differences in labeling between VisDrone and the
    SlimYOLOv3 model.

    The annotations that are specified in my augmentation of the dataset are all
    one off from the annotations provided by VisDrone. This is not explicitly
    stated in the paper or shown in the code, but are assumed by the
    configurations that they set in their repository. As such these are the following:

    VisDrone2018    SlimYOLOv3      Name
    1               0               pedestrian
    2               1               people
    3               2               bicycle
    4               3               car
    5               4               van
    6               5               truck
    7               6               tricycle
    8               7               awning-tricycle
    9               8               bus
    10              9               motor

    The ignored regions (0) and others (11) classes in the original dataset are
    ignored and removed from the dataset. In addition, all of the classes are
    shifted by one.
    """
    for bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion in visdrone_annotations:
        dn_class = int(object_category) - 1

        if dn_class == -1 or dn_class == 10:
            continue

        x_center = (int(bbox_left) + (int(bbox_width) / 2)) / img_width
        y_center = (int(bbox_top) + (int(bbox_height) / 2)) / img_height
        yield f'{dn_class} {x_center} {y_center} {int(bbox_width) / float(img_width)} {int(bbox_height)/float(img_height)}\n'


def convert_folder_to_darknet(folder: str):
    """
    Restructures the annotations in the dataset folder to match the format
    outlined in the DarkNet documentation.
    """
    source_directory = f'{folder}/annotations'
    destination_directory = f'{folder}/data'

    os.makedirs(destination_directory, exist_ok=True)

    for annotation_file in tqdm(glob.glob(os.path.join(source_directory, '*.txt'))):
        image_file = annotation_file.replace('.txt', '.jpg').replace('annotations', 'images')
        darknet_annotation_file = annotation_file.replace(source_directory, destination_directory)

        # Skip over existing files. Used if colab stops in the middle
        if os.path.exists(darknet_annotation_file):
            continue

        # Yes. The height is first (num rows, num columns, ...)
        # https://docs.opencv.org/master/d3/df2/tutorial_py_basic_ops.html
        img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        height, width, channels = img.shape

        # Read the annotations from the VisDrone dataset
        with open(annotation_file, 'r') as file:
            vis_annotations = file.read().strip().split('\n')
            vis_annotations = [bbox.split(',') for bbox in vis_annotations]

        # Save the annotations in the DarkNet format
        with open(darknet_annotation_file, 'w') as f:
            for darknet_annotation in convert_labels_to_darknet(width, height, vis_annotations):
                f.write(darknet_annotation)

        # Copy over the images
        shutil.copyfile(image_file, image_file.replace('images', 'data'))


def download_dataset(download_dir: str, extract_dir: str):
    """
    Main function to download and process the entire VisDrone2018 dataset.
    """
    destinations = []
    for folder, url in FOLDERS:
        destination = os.path.join(download_dir, folder)
        complete = pathlib.Path(download_dir, folder, 'complete')
        ziplocation = pathlib.Path(extract_dir, folder + '.zip')

        # See if the entire folder was processed
        if complete.exists():
            continue

        # Download the dataset
        gdown.cached_download(url, str(ziplocation), postprocess=gdown.extractall)

        # Unzip the dataset
        with zipfile.ZipFile(ziplocation) as zip:
            zip.extractall(folder)

        # Process the dataset
        convert_folder_to_darknet(folder)

        # Mark file as processed
        complete.touch()
        destinations.add(destination)
    return destinations

if __name__ == '__main__':
    download_dataset('.', '.')
