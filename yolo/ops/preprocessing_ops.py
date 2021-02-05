import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from yolo.ops import box_ops
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.ops import box_ops as bo


def shift_zeros(data, mask, axis = -2):

  zeros = tf.zeros_like(data)
  
  data_flat = tf.boolean_mask(data, mask)
  nonzero_lens = tf.reduce_sum(tf.cast(mask, dtype=tf.int32), axis=-2)
  nonzero_mask = tf.sequence_mask(nonzero_lens, maxlen=tf.shape(mask)[-2])
  perm1 = tf.range(0, tf.shape(tf.shape(data))[0] - 2)
  perm2 = tf.roll(tf.range(tf.shape(tf.shape(data))[0] - 2, tf.shape(tf.shape(data))[0]), 1, axis = -1)

  perm = tf.concat([perm1, perm2], axis = -1)
  nonzero_mask = tf.transpose(nonzero_mask, perm = perm)
  inds = tf.cast(tf.where(nonzero_mask), dtype=tf.int32)
  nonzero_data = tf.tensor_scatter_nd_update(zeros, tf.cast(tf.where(nonzero_mask), dtype=tf.int32),  data_flat)

  return nonzero_data 

def scale_image(image, resize=False, w=None, h=None):
  """Image Normalization.
    Args:
        image(tensorflow.python.framework.ops.Tensor): The image.
    Returns:
        A Normalized Function.
    """
  with tf.name_scope('scale_image'):
    image = tf.convert_to_tensor(image)
    if resize:
      image = tf.image.resize(image, size=(w, h))
    image = image / 255
  return image

def random_translate(image, box, classes, t, seed=10):
  t_x = tf.random.uniform(minval=-t, maxval=t, shape=(), dtype=tf.float32)
  t_y = tf.random.uniform(minval=-t, maxval=t, shape=(), dtype=tf.float32)
  box, classes = translate_boxes(box, classes, t_x, t_y)
  image = translate_image(image, t_x, t_y)
  return image, box, classes


def translate_boxes(box, classes, translate_x, translate_y):
  with tf.name_scope('translate_boxs'):
    box = box_ops.yxyx_to_xcycwh(box)
    x, y, w, h = tf.split(box, 4, axis = -1)
    x = x + translate_x
    y = y + translate_y

    x_mask_lower = x >= 0
    y_mask_lower = y >= 0
    x_mask_upper = x < 1
    y_mask_upper = y < 1

    x_mask = tf.math.logical_and(x_mask_lower, x_mask_upper)
    y_mask = tf.math.logical_and(y_mask_lower, y_mask_upper)
    mask = tf.math.logical_and(x_mask, y_mask)

    x = shift_zeros(x, mask) #tf.boolean_mask(x, mask)
    y = shift_zeros(y, mask) #tf.boolean_mask(y, mask)
    w = shift_zeros(w, mask) #tf.boolean_mask(w, mask)
    h = shift_zeros(h, mask) #tf.boolean_mask(h, mask)
    classes = shift_zeros(tf.expand_dims(classes, axis = -1), mask)
    classes = tf.squeeze(classes, axis = -1)

    box = tf.concat([x, y, w, h], axis=-1)
    box = box_ops.xcycwh_to_yxyx(box)
  return box, classes


def translate_image(image, translate_x, translate_y):
  with tf.name_scope('translate_image'):
    if (translate_x != 0 and translate_y != 0):
      image_jitter = tf.convert_to_tensor([translate_x, translate_y])
      image_jitter.set_shape([2])
      image = tfa.image.translate(
          image, image_jitter * tf.cast(tf.shape(image)[1], tf.float32))
  return image


def pad_max_instances(value, instances, pad_value=0, pad_axis=0):
  shape = tf.shape(value)
  dim1 = shape[pad_axis]
  take = tf.math.reduce_min([instances, dim1])
  value, _ = tf.split(
      value, [take, -1], axis=pad_axis)  # value[:instances, ...]
  pad = tf.convert_to_tensor([tf.math.reduce_max([instances - dim1, 0])])
  nshape = tf.concat([shape[:pad_axis], pad, shape[(pad_axis + 1):]], axis=0)
  pad_tensor = tf.fill(nshape, tf.cast(pad_value, dtype=value.dtype))
  value = tf.concat([value, pad_tensor], axis=pad_axis)
  return value

def resize_crop_filter(image, boxes, classes, default_width, default_height, target_width, target_height, randomize = False):
  with tf.name_scope('resize_crop_filter'):
    image = tf.image.resize(image, (target_width, target_height))

    if default_width > target_width:
      dx = (default_width - target_width)//2
      dy = (default_height - target_height)//2

      if randomize:
        dx = tf.random.uniform([], minval = 0, maxval = dx * 2, dtype = tf.int32)
        dy = tf.random.uniform([], minval = 0, maxval = dy * 2, dtype = tf.int32)

      image, boxes, classes = pad_filter_to_bbox(image, boxes, classes, default_width, default_height, dx, dy)
    elif default_width < target_width:
      dx = (target_width - default_width)//2
      dy = (target_height - default_height)//2

      if randomize:
        dx = tf.random.uniform([], minval = 0, maxval = dx * 2, dtype = tf.int32)
        dy = tf.random.uniform([], minval = 0, maxval = dy * 2, dtype = tf.int32)

      image, boxes, classes = crop_filter_to_bbox(image, boxes, classes, default_width, default_height, dx, dy, fix = False)
  return image, boxes, classes




def crop_filter_to_bbox(image, boxes, classes, target_width, target_height, offset_width, offset_height, fix = False):
  with tf.name_scope('resize_crop_filter'):
    shape = tf.shape(image)

    if tf.shape(shape)[0] == 4:
      height = shape[1]
      width = shape[2]
    else: # tf.shape(shape)[0] == 3:
      height = shape[0]
      width = shape[1] 
    
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
    if fix: 
      image = tf.image.pad_to_bounding_box(image, offset_height, offset_width, height, width)

    x_lower_bound = offset_width/width
    y_lower_bound = offset_height/height

    x_upper_bound = (offset_width + target_width)/width
    y_upper_bound = (offset_height + target_height)/height


    boxes = box_ops.yxyx_to_xcycwh(boxes)
    x, y, w, h = tf.split(tf.cast(boxes, x_lower_bound.dtype), 4, axis = -1)

    x_mask_lower = x > x_lower_bound
    y_mask_lower = y > y_lower_bound
    x_mask_upper = x < x_upper_bound
    y_mask_upper = y < y_upper_bound

    x_mask = tf.math.logical_and(x_mask_lower, x_mask_upper)
    y_mask = tf.math.logical_and(y_mask_lower, y_mask_upper)
    
    mask = tf.math.logical_and(x_mask, y_mask)

    x = shift_zeros(x, mask) #tf.boolean_mask(x, mask)
    y = shift_zeros(y, mask) #tf.boolean_mask(y, mask)
    w = shift_zeros(w, mask) #tf.boolean_mask(w, mask)
    h = shift_zeros(h, mask) #tf.boolean_mask(h, mask)
    classes = shift_zeros(tf.expand_dims(classes, axis = -1), mask)
    classes = tf.squeeze(classes, axis = -1)


    if not fix:
      x = (x - x_lower_bound) * tf.cast(width/target_width, x.dtype) 
      y = (y - y_lower_bound) * tf.cast(height/target_height, y.dtype) 
      w = w * tf.cast(width/target_width, w.dtype)
      h = h * tf.cast(height/target_height, h.dtype)

    boxes = tf.cast(tf.concat([x, y, w, h], axis = -1), boxes.dtype)
    boxes = box_ops.xcycwh_to_yxyx(boxes)
  return image, boxes, classes

def cut_out(image_full, boxes, classes, target_width, target_height, offset_width, offset_height):
  shape = tf.shape(image_full)

  if tf.shape(shape)[0] == 4:
    width = shape[1]
    height = shape[2]
  else: # tf.shape(shape)[0] == 3:
    width = shape[0]
    height = shape[1] 
  
  image_crop = tf.image.crop_to_bounding_box(image_full, offset_height, offset_width, target_height, target_width) + 1
  image_crop = tf.ones_like(image_crop)
  image_crop = tf.image.pad_to_bounding_box(image_crop, offset_height, offset_width, height, width)
  image_crop = 1 - image_crop
  
  x_lower_bound = offset_width/width
  y_lower_bound = offset_height/height

  x_upper_bound = (offset_width + target_width)/width
  y_upper_bound = (offset_height + target_height)/height

  boxes = box_ops.yxyx_to_xcycwh(boxes)
  
  x, y, w, h = tf.split(tf.cast(boxes, x_lower_bound.dtype), 4, axis = -1)

  x_mask_lower = x > x_lower_bound
  y_mask_lower = y > y_lower_bound
  x_mask_upper = x < x_upper_bound
  y_mask_upper = y < y_upper_bound

  x_mask = tf.math.logical_and(x_mask_lower, x_mask_upper)
  y_mask = tf.math.logical_and(y_mask_lower, y_mask_upper)
  mask = tf.math.logical_not(tf.math.logical_and(x_mask, y_mask))

  x = shift_zeros(x, mask) #tf.boolean_mask(x, mask)
  y = shift_zeros(y, mask) #tf.boolean_mask(y, mask)
  w = shift_zeros(w, mask) #tf.boolean_mask(w, mask)
  h = shift_zeros(h, mask) #tf.boolean_mask(h, mask)
  classes = shift_zeros(tf.expand_dims(classes, axis = -1), mask)
  classes = tf.squeeze(classes, axis = -1)

  boxes = tf.cast(tf.concat([x, y, w, h], axis = -1), boxes.dtype)
  boxes = box_ops.xcycwh_to_yxyx(boxes)

  image_full *= image_crop
  return image_full, boxes, classes

def cutmix_1(image_to_crop, boxes1, classes1, image_mask, boxes2, classes2, target_width, target_height, offset_width, offset_height):
  with tf.name_scope('cutmix'):
    image, boxes, classes = cut_out(image_mask, boxes2, classes2, target_width, target_height, offset_width, offset_height)
    image_, boxes_, classes_ = crop_filter_to_bbox(image_to_crop, boxes1, classes1, target_width, target_height, offset_width, offset_height, fix=True)
    image += image_
    boxes = tf.concat([boxes, boxes_], axis = -2)
    classes = tf.concat([classes, classes_], axis = -1)

    boxes = box_ops.yxyx_to_xcycwh(boxes)
    x, y, w, h = tf.split(boxes, 4, axis = -1)
    
    mask = x > 0
    x = shift_zeros(x, mask) #tf.boolean_mask(x, mask)
    y = shift_zeros(y, mask) #tf.boolean_mask(y, mask)
    w = shift_zeros(w, mask) #tf.boolean_mask(w, mask)
    h = shift_zeros(h, mask) #tf.boolean_mask(h, mask)
    classes = shift_zeros(tf.expand_dims(classes, axis = -1), mask)
    classes = tf.squeeze(classes, axis = -1)

    boxes = tf.cast(tf.concat([x, y, w, h], axis = -1), boxes.dtype)
    boxes = box_ops.xcycwh_to_yxyx(boxes)

  return image, boxes, classes

def cutmix_batch(image, boxes, classes, target_width, target_height, offset_width, offset_height):
  with tf.name_scope('cutmix_batch'):

    image_, boxes_, classes_ = cut_out(image, boxes, classes, target_width, target_height, offset_width, offset_height)
    image__, boxes__, classes__ = crop_filter_to_bbox(image, boxes, classes, target_width, target_height, offset_width, offset_height, fix=True)

    mix = tf.random.uniform([], minval = 0, maxval = 1)
    if mix > 0.5: 
      i_split1, i_split2 = tf.split(image__, 2, axis = 0)
      b_split1, b_split2 = tf.split(boxes__, 2, axis = 0)
      c_split1, c_split2 = tf.split(classes__, 2, axis = 0)

      image__ = tf.concat([i_split2, i_split1], axis = 0)
      boxes__ = tf.concat([b_split2, b_split1], axis = 0)
      classes__ = tf.concat([c_split2, c_split1], axis = 0)

    image = image_ + image__
    boxes = tf.concat([boxes_, boxes__], axis = -2)
    classes = tf.concat([classes_, classes__], axis = -1)

    boxes = box_ops.yxyx_to_xcycwh(boxes)
    x, y, w, h = tf.split(boxes, 4, axis = -1)
  
    mask = x > 0
    x = shift_zeros(x, mask) #tf.boolean_mask(x, mask)
    y = shift_zeros(y, mask) #tf.boolean_mask(y, mask)
    w = shift_zeros(w, mask) #tf.boolean_mask(w, mask)
    h = shift_zeros(h, mask) #tf.boolean_mask(h, mask)
    classes = shift_zeros(tf.expand_dims(classes, axis = -1), mask)
    classes = tf.squeeze(classes, axis = -1)

    boxes = tf.cast(tf.concat([x, y, w, h], axis = -1), boxes.dtype)
    boxes = box_ops.xcycwh_to_yxyx(boxes)

    x = tf.squeeze(x, axis = -1)
    classes = tf.where(x == 0, -1, classes)

    num_detections = tf.reduce_sum(tf.cast(x > 0, tf.int32), axis = -1)

  return image, boxes, classes, num_detections

def randomized_cutmix_batch(image, boxes, classes):
  shape = tf.shape(image)

  width = shape[1]
  height = shape[2]

  w_limit = 3 * width//4
  h_limit = 3 * height//4

  twidth = tf.random.uniform([], minval = width//4, maxval = w_limit, dtype = tf.int32)
  theight = tf.random.uniform([], minval = height//4, maxval = h_limit, dtype = tf.int32)

  owidth = tf.random.uniform([], minval = 0, maxval = width - twidth, dtype = tf.int32)
  oheight = tf.random.uniform([], minval = 0, maxval = height - theight, dtype = tf.int32)

  image, boxes, classes, num_detections = cutmix_batch(image, boxes, classes, twidth, theight, owidth, oheight)
  return image, boxes, classes, num_detections

def randomized_cutmix_split(image, boxes, classes):
  # this is not how it is really done
  mix = tf.random.uniform([], maxval = 1, dtype = tf.int32)
  if mix == 1: 
    i1, i2, i3, i4 = tf.split(image, 4, axis = 0)
    b1, b2, b3, b4 = tf.split(boxes, 2, axis = 0)
    c1, c2, c3, c4 = tf.split(classes, 2, axis = 0)

    image = tf.concat([i1, i3, i2, i4], axis = 0)
    boxes = tf.concat([b1, b3, b2, b4], axis = 0)
    classes = tf.concat([b1, b3, b2, b4], axis = 0)    

  i_split1, i_split2 = tf.split(image, 2, axis = 0)
  b_split1, b_split2 = tf.split(boxes, 2, axis = 0)
  c_split1, c_split2 = tf.split(classes, 2, axis = 0)

  
  i_split1, b_split1, c_split1, num_dets1 = randomized_cutmix_batch(i_split1, b_split1, c_split1)
  i_split2, b_split2, c_split2, num_dets2 = randomized_cutmix_batch(i_split2, b_split2, c_split2)
  image = tf.concat([i_split2, i_split1], axis = 0)
  boxes = tf.concat([b_split2, b_split1], axis = 0)
  classes = tf.concat([c_split2, c_split1], axis = 0)
  num_detections = tf.concat([num_dets2, num_dets1], axis = 0)
  #image, boxes, classes, num_detections = randomized_cutmix_batch(image, boxes, classes)

  return image, boxes, classes, num_detections

def pad_filter_to_bbox(image, boxes, classes, target_width, target_height, offset_width, offset_height):
  with tf.name_scope('resize_crop_filter'):
    shape = tf.shape(image)

    if tf.shape(shape)[0] == 4:
      height = shape[1]
      width = shape[2]
    else: # tf.shape(shape)[0] == 3:
      height = shape[0]
      width = shape[1] 
    
    image = tf.image.pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width)

    x_lower_bound = tf.cast(offset_width/width, tf.float32)
    y_lower_bound = tf.cast(offset_height/height, tf.float32)

    boxes = box_ops.yxyx_to_xcycwh(boxes)
    x, y, w, h = tf.split(tf.cast(boxes, x_lower_bound.dtype), 4, axis = -1)

    x = (x + x_lower_bound) * tf.cast(width/target_width, x.dtype) 
    y = (y + y_lower_bound) * tf.cast(height/target_height, y.dtype) 
    w = w * tf.cast(width/target_width, w.dtype)
    h = h * tf.cast(height/target_height, h.dtype)

    boxes = tf.cast(tf.concat([x, y, w, h], axis = -1), boxes.dtype)
    boxes = box_ops.xcycwh_to_yxyx(boxes)
  return image, boxes, classes



def fit_preserve_aspect_ratio(image,
                              boxes,
                              width=None,
                              height=None,
                              target_dim=None):
  if width is None or height is None:
    shape = tf.shape(data['image'])
    if tf.shape(shape)[0] == 4:
      width = shape[1]
      height = shape[2]
    else:
      width = shape[0]
      height = shape[1]

  clipper = tf.math.maximum(width, height)
  if target_dim is None:
    target_dim = clipper

  pad_width = clipper - width
  pad_height = clipper - height
  image = tf.image.pad_to_bounding_box(image, pad_width // 2, pad_height // 2,
                                       clipper, clipper)

  boxes = box_ops.yxyx_to_xcycwh(boxes)
  x, y, w, h = tf.split(boxes, 4, axis=-1)

  y *= tf.cast(width / clipper, tf.float32)
  x *= tf.cast(height / clipper, tf.float32)

  y += tf.cast((pad_width / clipper) / 2, tf.float32)
  x += tf.cast((pad_height / clipper) / 2, tf.float32)

  h *= tf.cast(width / clipper, tf.float32)
  w *= tf.cast(height / clipper, tf.float32)

  boxes = tf.concat([x, y, w, h], axis=-1)

  boxes = box_ops.xcycwh_to_yxyx(boxes)
  image = tf.image.resize(image, (target_dim, target_dim))
  return image, boxes


def get_best_anchor(y_true, anchors, width=1, height=1):
  """
    get the correct anchor that is assoiciated with each box using IOU betwenn input anchors and gt
    Args:
        y_true: tf.Tensor[] for the list of bounding boxes in the yolo format
        anchors: list or tensor for the anchor boxes to be used in prediction found via Kmeans
        size: size of the image that the bounding boxes were selected at 416 is the default for the original YOLO model
    return:
        tf.Tensor: y_true with the anchor associated with each ground truth box known
    """
  with tf.name_scope('get_anchor'):
    width = tf.cast(width, dtype=tf.float32)
    height = tf.cast(height, dtype=tf.float32)

    anchor_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]

    # scale thhe boxes
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
    anchors_x = anchors[..., 0] / width
    anchors_y = anchors[..., 1] / height
    anchors = tf.stack([anchors_x, anchors_y], axis=-1)

    # build a matrix of anchor boxes
    anchors = tf.transpose(anchors, perm=[1, 0])
    anchor_xy = tf.tile(
        tf.expand_dims(anchor_xy, axis=-1), [1, 1, tf.shape(anchors)[-1]])
    anchors = tf.tile(
        tf.expand_dims(anchors, axis=0), [tf.shape(anchor_xy)[0], 1, 1])

    # stack the xy so, each anchor is asscoaited once with each center from the ground truth input
    anchors = K.concatenate([anchor_xy, anchors], axis=1)
    anchors = tf.transpose(anchors, perm=[2, 0, 1])

    # copy the gt n times so that each anchor from above can be compared to input ground truth
    truth_comp = tf.tile(
        tf.expand_dims(y_true[..., 0:4], axis=-1),
        [1, 1, tf.shape(anchors)[0]])
    truth_comp = tf.transpose(truth_comp, perm=[2, 0, 1])

    # compute intersection over union of the boxes, and take the argmax of comuted iou for each box.
    # thus each box is associated with the largest interection over union
    iou_raw = box_ops.compute_iou(truth_comp, anchors)

    gt_mask = tf.cast(iou_raw > 0.213, dtype=iou_raw.dtype)

    num_k = tf.reduce_max(
        tf.reduce_sum(tf.transpose(gt_mask, perm=[1, 0]), axis=1))
    if num_k <= 0:
      num_k = 1.0

    values, indexes = tf.math.top_k(
        tf.transpose(iou_raw, perm=[1, 0]),
        k=tf.cast(num_k, dtype=tf.int32),
        sorted=True)
    ind_mask = tf.cast(values > 0.213, dtype=indexes.dtype)
    iou_index = tf.concat([
        K.expand_dims(indexes[..., 0], axis=-1),
        ((indexes[..., 1:] + 1) * ind_mask[..., 1:]) - 1
    ],
                          axis=-1)

    stack = tf.zeros(
        [tf.shape(iou_index)[0],
         tf.cast(1, dtype=iou_index.dtype)],
        dtype=iou_index.dtype) - 1
    while num_k < 5:
      iou_index = tf.concat([iou_index, stack], axis=-1)
      num_k += 1
    iou_index = iou_index[..., :5]

    values = tf.concat([
        K.expand_dims(values[..., 0], axis=-1),
        ((values[..., 1:]) * tf.cast(ind_mask[..., 1:], dtype=tf.float32))
    ],
                       axis=-1)
  return tf.cast(iou_index, dtype=tf.float32)

# add better documentation 
def build_grided_gt(y_true, mask, size, num_classes, dtype, use_tie_breaker):
  """
    convert ground truth for use in loss functions
    Args:
        y_true: tf.Tensor[] ground truth [box coords[0:4], classes_onehot[0:-1], best_fit_anchor_box]
        mask: list of the anchor boxes choresponding to the output, ex. [1, 2, 3] tells this layer to predict only the first 3 anchors in the total.
        size: the dimensions of this output, for regular, it progresses from 13, to 26, to 52

    Return:
        tf.Tensor[] of shape [size, size, #of_anchors, 4, 1, num_classes]
    """
  boxes = tf.cast(y_true['bbox'], dtype)
  classes = tf.one_hot(
      tf.cast(y_true['classes'], dtype=tf.int32),
      depth=num_classes,
      dtype=dtype)
  anchors = tf.cast(y_true['best_anchors'], dtype)

  num_boxes = tf.shape(boxes)[0]
  len_masks = tf.shape(mask)[0]

  full = tf.zeros([size, size, len_masks, num_classes + 4 + 1], dtype=dtype)
  depth_track = tf.zeros((size, size, len_masks), dtype=tf.int32)

  x = tf.cast(boxes[..., 0] * tf.cast(size, dtype=dtype), dtype=tf.int32)
  y = tf.cast(boxes[..., 1] * tf.cast(size, dtype=dtype), dtype=tf.int32)

  anchors = tf.repeat(tf.expand_dims(anchors, axis=-1), len_masks, axis=-1)

  update_index = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  update = tf.TensorArray(dtype, size=0, dynamic_size=True)
  const = tf.cast(tf.convert_to_tensor([1.]), dtype=dtype)
  mask = tf.cast(mask, dtype=dtype)

  i = 0
  anchor_id = 0
  for box_id in range(num_boxes):
    if K.all(tf.math.equal(boxes[box_id, 2:4], 0)):
      continue
    if K.any(tf.math.less(boxes[box_id, 0:2], 0.0)) or K.any(
        tf.math.greater_equal(boxes[box_id, 0:2], 1.0)):
      continue
    if use_tie_breaker:
      for anchor_id in range(tf.shape(anchors)[-1]):
        index = tf.math.equal(anchors[box_id, anchor_id], mask)
        if K.any(index):
          p = tf.cast(K.argmax(tf.cast(index, dtype=tf.int32)), dtype=tf.int32)
          uid = 1
          used = depth_track[y[box_id], x[box_id], p]

          if anchor_id == 0:
            # write the box to the update list
            # the boxes output from yolo are for some reason have the x and y indexes swapped for some reason, I am not sure why
            """peculiar"""
            update_index = update_index.write(i, [y[box_id], x[box_id], p])
            value = K.concatenate([boxes[box_id], const, classes[box_id]])
            update = update.write(i, value)
          elif tf.math.equal(used, 2) or tf.math.equal(used, 0):
            uid = 2
            # write the box to the update list
            # the boxes output from yolo are for some reason have the x and y indexes swapped for some reason, I am not sure why
            """peculiar"""
            update_index = update_index.write(i, [y[box_id], x[box_id], p])
            value = K.concatenate([boxes[box_id], const, classes[box_id]])
            update = update.write(i, value)

          depth_track = tf.tensor_scatter_nd_update(depth_track,
                                                    [(y[box_id], x[box_id], p)],
                                                    [uid])
          i += 1
    else:
      index = tf.math.equal(anchors[box_id, 0], mask)
      if K.any(index):
        # tf.(0, anchors[ box_id, 0])
        p = tf.cast(K.argmax(tf.cast(index, dtype=tf.int32)), dtype=tf.int32)
        update_index = update_index.write(i, [y[box_id], x[box_id], p])
        value = K.concatenate([boxes[box_id], const, classes[box_id]])
        update = update.write(i, value)
        i += 1

  # if the size of the update list is not 0, do an update, other wise, no boxes and pass an empty grid
  if tf.math.greater(update_index.size(), 0):
    update_index = update_index.stack()
    update = update.stack()
    full = tf.tensor_scatter_nd_update(full, update_index, update)
  return full


def build_batch_grided_gt(y_true, mask, size, num_classes, dtype,
                          use_tie_breaker):
  """
    convert ground truth for use in loss functions
    Args:
        y_true: tf.Tensor[] ground truth [box coords[0:4], classes_onehot[0:-1], best_fit_anchor_box]
        mask: list of the anchor boxes choresponding to the output, ex. [1, 2, 3] tells this layer to predict only the first 3 anchors in the total.
        size: the dimensions of this output, for regular, it progresses from 13, to 26, to 52

    Return:
        tf.Tensor[] of shape [batch, size, size, #of_anchors, 4, 1, num_classes]
    """
  boxes = tf.cast(y_true['bbox'], dtype)
  classes = tf.one_hot(
      tf.cast(y_true['classes'], dtype=tf.int32),
      depth=num_classes,
      dtype=dtype)
  anchors = tf.cast(y_true['best_anchors'], dtype)

  batches = tf.shape(boxes)[0]
  num_boxes = tf.shape(boxes)[1]
  len_masks = tf.shape(mask)[0]

  full = tf.zeros([batches, size, size, len_masks, num_classes + 4 + 1],
                  dtype=dtype)
  depth_track = tf.zeros((batches, size, size, len_masks), dtype=tf.int32)

  x = tf.cast(boxes[..., 0] * tf.cast(size, dtype=dtype), dtype=tf.int32)
  y = tf.cast(boxes[..., 1] * tf.cast(size, dtype=dtype), dtype=tf.int32)

  anchors = tf.repeat(tf.expand_dims(anchors, axis=-1), len_masks, axis=-1)

  update_index = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  update = tf.TensorArray(dtype, size=0, dynamic_size=True)
  const = tf.cast(tf.convert_to_tensor([1.]), dtype=dtype)
  mask = tf.cast(mask, dtype=dtype)

  i = 0
  anchor_id = 0
  for batch in range(batches):
    for box_id in range(num_boxes):
      if K.all(tf.math.equal(boxes[batch, box_id, 2:4], 0)):
        continue
      if K.any(tf.math.less(boxes[batch, box_id, 0:2], 0.0)) or K.any(
          tf.math.greater_equal(boxes[batch, box_id, 0:2], 1.0)):
        continue
      if use_tie_breaker:
        for anchor_id in range(tf.shape(anchors)[-1]):
          index = tf.math.equal(anchors[batch, box_id, anchor_id], mask)
          if K.any(index):
            p = tf.cast(
                K.argmax(tf.cast(index, dtype=tf.int32)), dtype=tf.int32)
            uid = 1

            used = depth_track[batch, y[batch, box_id], x[batch, box_id], p]
            if anchor_id == 0:
              # write the box to the update list
              # the boxes output from yolo are for some reason have the x and y indexes swapped for some reason, I am not sure why
              """peculiar"""
              update_index = update_index.write(
                  i, [batch, y[batch, box_id], x[batch, box_id], p])
              value = K.concatenate(
                  [boxes[batch, box_id], const, classes[batch, box_id]])
              update = update.write(i, value)
            elif tf.math.equal(used, 2) or tf.math.equal(used, 0):
              uid = 2
              # write the box to the update list
              # the boxes output from yolo are for some reason have the x and y indexes swapped for some reason, I am not sure why
              """peculiar"""
              update_index = update_index.write(
                  i, [batch, y[batch, box_id], x[batch, box_id], p])
              value = K.concatenate(
                  [boxes[batch, box_id], const, classes[batch, box_id]])
              update = update.write(i, value)

            depth_track = tf.tensor_scatter_nd_update(
                depth_track, [(batch, y[batch, box_id], x[batch, box_id], p)],
                [uid])
            i += 1
      else:
        index = tf.math.equal(anchors[batch, box_id, 0], mask)
        if K.any(index):
          # tf.(0, anchors[batch, box_id, 0])
          p = tf.cast(K.argmax(tf.cast(index, dtype=tf.int32)), dtype=tf.int32)
          update_index = update_index.write(
              i, [batch, y[batch, box_id], x[batch, box_id], p])
          value = K.concatenate(
              [boxes[batch, box_id], const, classes[batch, box_id]])
          update = update.write(i, value)
          i += 1

  # if the size of the update list is not 0, do an update, other wise, no boxes and pass an empty grid
  if tf.math.greater(update_index.size(), 0):
    update_index = update_index.stack()
    update = update.stack()
    full = tf.tensor_scatter_nd_update(full, update_index, update)
  return full


def patch_four(images, boxes, classes):
  image1, image2, image3, image4 = tf.split(images, 4, axis = 0)
  patch1 = tf.concat([image1,image2], axis = -2)
  patch2 = tf.concat([image3, image4], axis = -2)
  full_image = tf.concat([patch1, patch2], axis = -3)
  num_instances = tf.shape(boxes)[-2]

  #assume box xywh
  box1, box2, box3, box4 = tf.split(boxes * 0.5, 4, axis = 0)
  class1, class2, class3, class4 = tf.split(classes, 4, axis = 0)

  #unpad the tensors
  box1 = unpad_tensor(box1)
  box2 = unpad_tensor(box2)
  box3 = unpad_tensor(box3)
  box4 = unpad_tensor(box4)
  class1 = unpad_tensor(class1, -1)
  class2 = unpad_tensor(class2, -1)
  class3 = unpad_tensor(class3, -1)
  class4 = unpad_tensor(class4, -1)

  #translate boxes
  box2, class2 = translate_boxes(box2, class2, .5, 0)
  box3, class3 = translate_boxes(box3, class3, 0, .5)
  box4, class4 = translate_boxes(box4, class4, .5, .5)

  full_boxes = tf.concat([box1, box2, box3, box4], axis = -2)
  full_class = tf.concat([class1, class2, class3, class4], axis = -1)

  full_boxes = preprocess_ops.clip_or_pad_to_fixed_size(full_boxes, num_instances, 0)
  full_class = preprocess_ops.clip_or_pad_to_fixed_size(full_class, num_instances, -1)
  return full_image, tf.expand_dims(full_boxes, axis = 0), tf.expand_dims(full_class, axis = 0)

def random_crop_contain_center(image, boxes, classes, crop_fraction, seed=10):
  with tf.name_scope('random_crop_contain_center'):
    image_size = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    crop_size = (crop_fraction * tf.math.minimum(image_size[0], image_size[1]))
    crop_offset = tf.cast((image_size - crop_size) / 2.0, dtype=tf.int32)
    crop_size = tf.cast(crop_size, dtype=tf.int32)
    jitter = tf.random.uniform([],
                               minval=-crop_offset[0],
                               maxval= crop_offset[0],
                               seed=seed,
                               dtype=tf.int32)
    crop_offset += jitter
    cropped_image = image[
        crop_offset[0]:crop_offset[0] + crop_size,
        crop_offset[1]:crop_offset[1] + crop_size, :]
    y_scale = tf.cast(crop_size, tf.float32) / image_size[0]
    x_scale = tf.cast(crop_size, tf.float32) / image_size[1]
    image_scale = tf.convert_to_tensor([y_scale, x_scale])
    cropped_shape = tf.convert_to_tensor([crop_size, crop_size])

    boxes = bo.denormalize_boxes(boxes, image_size)
    boxes -= tf.tile(tf.expand_dims(tf.cast(crop_offset, tf.float32), axis=0), [1, 2])
    boxes = bo.clip_boxes(boxes, cropped_shape)
    boxes = bo.normalize_boxes(boxes, cropped_shape)
    indices = bo.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)
    return cropped_image, boxes, classes

def unpad_tensor(input_tensor, padding_value = 0):
  if tf.rank(input_tensor) == 3:
    abs_sum_tensor = tf.reduce_sum(tf.abs(input_tensor), -1)
    padding_vector = tf.ones(shape = (1, 1)) * padding_value
    mask = abs_sum_tensor != padding_vector
    return input_tensor[mask]
  elif tf.rank(input_tensor) == 2:
    return input_tensor[input_tensor != padding_value]

