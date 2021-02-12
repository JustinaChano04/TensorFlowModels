import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from official.vision.beta.ops import box_ops, preprocess_ops
from yolo.ops import preprocessing_ops as po
from yolo.utils.demos import utils, coco


def preprocess(point):
  image = tf.cast(point['image'], tf.float32)
  image = image / 255.
  image = tf.image.resize(image, (416, 416))
  obj = point['objects']
  classes = obj['label']
  boxes = obj['bbox']

  boxes = preprocess_ops.clip_or_pad_to_fixed_size(boxes, 200, 0)
  classes = preprocess_ops.clip_or_pad_to_fixed_size(classes, 200, -1)
  label = {
      'bbox': tf.cast(boxes, tf.float32),
      'classes': tf.cast(classes, tf.float32)
  }
  return image, label


ds = tfds.load('coco/2017', split='train')
ds = ds.map(preprocess)
ds = ds.batch(4)
drawer = utils.DrawBoxes(labels=coco.get_coco_names(), thickness=1)
ds = ds.shuffle(50)
i = 0
for images, label in ds:
  image, boxes, classes = po.mosaic_four(images, label['bbox'],
                                         label['classes'])
  sample = {
      'bbox': tf.expand_dims(boxes, axis=0),
      'classes': tf.expand_dims(classes, axis=0)
  }

  print(tf.shape(image))
  print(tf.shape(boxes))
  print(tf.shape(classes))

  image = drawer(tf.expand_dims(image, axis=0), sample)

  for i in range(4):
    plt.imshow(image[0, i])
    plt.show()
  print(f'saved image{i}')
  i += 1
  if i == 10:
    break
