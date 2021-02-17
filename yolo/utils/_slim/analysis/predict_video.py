# Adapted from https://github.com/PurdueCAM2Project/TensorFlowModelGardeners/blob/5a5c87/yolo/demos/video_detect_cpu.py

import cv2
import time
import os
import sys

import numpy as np
import datetime
import colorsys

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

'''Video Buffer using cv2'''


def DEFAULT(colors, label_names, display_name):
    def draw_box_name(image, box, classes, conf):
        if box[3] == 0:
            return False
        cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]),
                      colors[classes], 1)
        cv2.putText(image, "%s, %0.3f" % (label_names[classes], conf),
                    (box[0], box[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[classes], 1)
        return True

    def draw_box(image, box, classes, conf):
        if box[3] == 0:
            return False
        cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]),
                      colors[classes], 1)
        return True

    if display_name:
        return draw_box_name
    else:
        return draw_box


def get_coco_names(path="yolo/dataloaders/dataset_specs/coco.names"):
    with open(path, "r") as f:
        data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].strip()
    return data


def gen_colors(max_classes):
    hue = np.linspace(start=0, stop=1, num=max_classes)
    np.random.shuffle(hue)
    colors = []
    for val in hue:
        colors.append(colorsys.hsv_to_rgb(val, 0.75, 1.0))
    return colors


def int_scale_boxes(boxes, classes, width, height):
    boxes = K.stack([
        tf.cast(boxes[..., 1] * width, dtype=tf.int32),
        tf.cast(boxes[..., 3] * width, dtype=tf.int32),
        tf.cast(boxes[..., 0] * height, dtype=tf.int32),
        tf.cast(boxes[..., 2] * height, dtype=tf.int32)
    ],
                    axis=-1)
    classes = tf.cast(classes, dtype=tf.int32)
    return boxes, classes


def draw_box(image, boxes, classes, conf, draw_fn):
    i = 0
    for i in range(boxes.shape[0]):
        if draw_fn(image, boxes[i], classes[i], conf[i]):
            i += 1
        else:
            return i
    return i


def video_processor(model, vidpath, device="/CPU:0", get_draw_fn=DEFAULT):
    img_array = []

    i = 0
    t = 0
    start = time.time()
    tick = 0
    e, f, a, b, c, d = 0, 0, 0, 0, 0, 0
    if isinstance(model, str):
        with tf.device(device):
            model = ks.models.load_model(model)
            model.make_predict_function()

    if hasattr(model, "predict"):
        predfunc = model.predict
        print("using pred function")
    else:
        predfunc = model
        print("using call function")

    colors = gen_colors(10)
    label_names = get_coco_names(path="datasets/visdrone/labels.txt")
    print(label_names)

    # output_writer = cv2.VideoWriter('yolo_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_count, (480, 640))  # change output file name if needed
    pred = None
    cap = cv2.VideoCapture(vidpath)
    assert cap.isOpened()

    width = int(cap.get(3))
    height = int(cap.get(4))
    print('width, height, fps:', width, height, int(cap.get(5)))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        success, image = cap.read()

        #with tf.device(device):
        e = datetime.datetime.now()
        image = tf.cast(image, dtype=tf.float32)
        image = image / 255
        f = datetime.datetime.now()

        if t % 1 == 0:
            a = datetime.datetime.now()
            #with tf.device(device):
            pimage = tf.expand_dims(image, axis=0)
            pimage = tf.image.resize(pimage, (608, 608))
            pred = predfunc(pimage)
            b = datetime.datetime.now()

        image = image.numpy()
        if pred != None:
            c = datetime.datetime.now()
            boxes, classes = int_scale_boxes(pred["bbox"], pred["classes"],
                                             width, height)
            draw = get_draw_fn(colors, label_names, 'YOLO')
            draw_box(image, boxes[0].numpy(), classes[0].numpy(),
                     pred["confidence"][0], draw)
            d = datetime.datetime.now()

        cv2.imshow('frame', image)
        i += 1
        t += 1

        if time.time() - start - tick >= 1:
            tick += 1
            print_opt((((f - e) + (b - a) + (d - c))), i)
            i = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return


def print_opt(latency, fps):
    print(
        f"                                \rlatency:, \033[1;32;40m{latency * 1000} \033[0m ms",
        end="\n")
    print("                                 \rfps: \033[1;34;40m%d\033[0m " %
          (fps),
          end="\n")
    print("\033[F\033[F\033[F", end="\n")
    return


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage: python3 -m analysis.predict_video <model_path> [video_path].")
        exit()

    model = sys.argv[1]
    if len(sys.argv) > 2:
        vidpath = sys.argv[2]
        if not os.path.exists(vidpath):
            print("Input video path doesn't exist.")
            exit()
    else:
        vidpath = 0

    video_processor(model, vidpath)
