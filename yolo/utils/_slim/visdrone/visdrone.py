"""visdrone dataset."""

import os, glob

import tensorflow as tf
import tensorflow_datasets as tfds

from . import download_dataset

_NAMES_FILE = os.path.join(os.path.dirname(__file__), 'labels.txt')

# TODO(visdrone): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """Dataset for the VisDrone competition in {year}
"""

_CITATION = """
@article{zhuvisdrone2018,
title={Vision Meets Drones: A Challenge},
author={Zhu, Pengfei and Wen, Longyin and Bian, Xiao and Haibin, Ling and Hu, Qinghua},
journal={arXiv preprint:1804.07437},
year={2018} }
""".strip()

class VisdroneConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Visdrone."""

  def __init__(self, splits, **kwargs):
    super().__init__(version=tfds.core.Version('1.0.0'), **kwargs)
    self.splits = splits


class Visdrone(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for visdrone dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial.',
  }

  BUILDER_CONFIGS = [
      VisdroneConfig(
          name='2018',
          description=_DESCRIPTION.format(year=2018),
          splits=[
              tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST
          ],
      ),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # Adapted from COCO style features from: https://www.tensorflow.org/datasets/catalog/coco
            'image': tfds.features.Image(shape=(1920, 1080, 3), dtype=tf.uint8),
            'image/filename': tfds.features.Text(),
            'objects': tfds.features.Sequence({
                'bbox': tfds.features.BBoxFeature(),
                'label': tfds.features.ClassLabel(names_file=_NAMES_FILE),
            }),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'objects'),
        homepage='http://aiskyeye.com/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    train_path, validation_path, test_path = download_dataset.download_dataset(
        dl_manager._download_dir, dl_manager._extract_dir)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={'path': train_path}
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={'path': validation_path}
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={'path': test_path}
        )
    ]

  def _generate_examples(self, path):
    """Yields examples."""
    files = glob.glob(os.path.join(path, 'data', '*.txt'))
    for annotation_file in files:
        image_id = annotation_file.rsplit('/', 1)[1].rsplit('.', 1)[0]
        image_file = annotation_file.replace('.txt', '.jpg')

        # Skip over occasional errors in the dataset
        if not os.path.exists(image_file):
            continue

        # Read annotations
        annotations = []
        with open(annotation_file) as file:
            for line in file.readlines():
                class_id, box_cx, box_cy, box_width, box_height = line.split()

                class_id = int(class_id)
                box_cx = float(box_cx)
                box_cy = float(box_cy)
                box_width = float(box_width)
                box_height = float(box_height)

                xmin = max(0.0, float(box_cx - 0.5*box_width))
                assert(xmin >= 0 and xmin <= 1)

                xmax = min(1.0, float(box_cx + 0.5*box_width))
                assert(xmax >= 0 and xmax <= 1)

                ymin = max(0.0, float(box_cy - 0.5*box_height))
                assert(ymin >= 0 and ymin <= 1)

                ymax = min(1.0, float(box_cy + 0.5*box_height))
                assert(ymax >= 0 and ymax <= 1)

                annotations.append({
                    'bbox': tfds.features.BBox(ymin, xmin, ymax, xmax),
                    'label': class_id
                })

        yield image_id, {
            'image': image_file,
            'image/filename': image_id + '.jpg',
            'objects': annotations,
        }
