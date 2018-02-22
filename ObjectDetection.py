import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util as label
from object_detection.utils import visualization_utils as vis_util
from PIL import Image

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def detect_objects(image_np, sess, detect_object_util):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detect_object_util.get_tensor_by_name('image_tensor:0')
    boxes = detect_object_util.get_tensor_by_name('detection_boxes:0')
    scores = detect_object_util.get_tensor_by_name('detection_scores:0')
    classes = detect_object_util.get_tensor_by_name('detection_classes:0')
    num_detections = detect_object_util.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np


MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_MODEL = os.path.join('/Users/gbala/tf/TensorFlow', MODEL_NAME, 'frozen_inference_graph.pb')

#loaidng the coco data set
PATH_TO_DATA_SET = os.path.join('/Users/gbala/Library/Python/2.7/lib/python/site-packages/tensorflow/models/', 'object_detection', 'data', 'mscoco_label_map.pbtxt')

labels = label.load_labelmap(PATH_TO_DATA_SET)
categories = label.convert_label_map_to_categories(labels, max_num_classes=20, use_display_name=True)
category_index = label.create_category_index(categories)

image_path = '/Users/gbala/tf/TensorFlow/image3.jpg'

image = Image.open(image_path)
image_array = load_image_into_numpy_array(image)
#plt.interactive(False)
#plt.imshow(image_array)
#plt.show(block=True)

detect_object_util = tf.Graph()
with detect_object_util.as_default():
    frozen_inference_graph = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as model:
        model_graph = model.read()
        frozen_inference_graph.ParseFromString(model_graph)
        tf.import_graph_def(frozen_inference_graph, name='')

with tf.Session(graph=detect_object_util) as session:
    image = Image.open(image_path)
    image_array = load_image_into_numpy_array(image)
    image_process = detect_objects(image_array, session, detect_object_util)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_process)
    plt.show(block=True)

