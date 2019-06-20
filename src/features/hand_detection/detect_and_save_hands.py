#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:23:57 2019

@author: hitansh
"""
import numpy as np
import os
import sys
# import tarfile
import tensorflow as tf
# import zipfile
# from distutils.version import StrictVersion
# from collections import defaultdict
# from io import StringIO
# from matplotlib import pyplot as plt
from PIL import Image
import cv2
# from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# print('pwd: ' + os.getcwd())


'''
## Variables

Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.
By default we use an "SSD with Mobilenet" model here.
See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
'''

os.chdir(r'../../../data/TensorFlow/workspace/training_demo/')
# print('Changed to: ' + os.getcwd())

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_FROZEN_GRAPH = 'trained-inference-graphs/output_inference_graph_v1.pb/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = 'annotations/label_map.pbtxt'

# Loading graph into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
'''
rh: right hand
lh: left hand
ll: left label
rl: right label
'''

# Detection
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = 'test_images'
PATH_TO_TEST_IMAGES_DIR = '../../../train/XR_HAND_CROPPED'
TEST_IMAGE_PATHS = []

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# loading files list
for r, d, f in os.walk(PATH_TO_TEST_IMAGES_DIR):
    for file in f:
        if '.png' in file:
            # os.remove(os.path.join(r,file))
            TEST_IMAGE_PATHS.append(os.path.join(r, file))
total_files = len(TEST_IMAGE_PATHS)


# This is a fucntion which detects the stuff (like hands etc and then returns a dictionary)
def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict


# Looping through all images
log = open('../../../../../hand_detection_script_log.txt', 'w')
# this file is one folder behind x_ray folder

j = 0
count = 0

SAVE_PATH = r'../../../train/XR_HAND_CENTRED_NEW'

if not os.path.exists(os.path.exists(SAVE_PATH)):
    os.mkdir(SAVE_PATH)

for image_path in TEST_IMAGE_PATHS[:30]:
    count += 1
    # print(count,end='\r')
    log.write(str(count) + ' ')
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = cv2.imread(image_path, 1)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    # Visualization of the results of a detection.
    boxes = output_dict['detection_boxes']
    bool_anything_found = 0
    detection_number = 0
    for i in range(output_dict['num_detections']):
        if(output_dict['detection_scores'][i] > 0.70):
            j += 1
            detection_number += 1
            image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
            im_width, im_height = image_pil.size
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            # plt.figure(j,figsize=IMAGE_SIZE)
            # plt.plot([left,right],[bottom,top],linewidth=1.0)
            # plt.imshow(image_np)
            # check if it is a label
            if(output_dict['detection_classes'][i] == 3 or output_dict['detection_classes'][i] == 4):
                '''
                This code can be used to paint labels, however, it is not implemented
                mask=np.zeros(image_np.shape,dtype='uint8')
                mask[int(top):int(bottom+top), int(left):int(left+right)]=image_np[int(top):int(bottom+top), int(left):int(left+right)]
                mask[:int(top)]
                '''
                # j+=1
                # plt.figure(j,figsize=IMAGE_SIZE)
                # plt.imshow(mask)
                # inpainted_image=cv2.inpaint(image_np,mask,3,cv2.INPAINT_TELEA)
                # cv2.imshow(inpainted_image)
                # print('Label', end='\r')
                pass
            # if it is not a label
            # will only come here if score>70% and not a label
        else:
            bool_anything_found = 1
            j = j + 1
            crop_img = image_np[int(top):int(bottom + top), int(left):int(left + right)]
            # plt.figure(j,figsize=IMAGE_SIZE)
            # plt.imshow(crop_img)
            IMAGE_PATH_DIR = os.path.join(SAVE_PATH, image_path.split('/')[-3], image_path.split('/')[-2])
            if not os.path.exists(IMAGE_PATH_DIR):
                os.makedirs(IMAGE_PATH_DIR)
            IMAGE_PATH_NEW = IMAGE_PATH_DIR + '/' + image_path.split('/')[-1][:-4] + r'_cropped_' + str(detection_number) + '.png'
            cv2.imwrite(IMAGE_PATH_NEW, crop_img)
            log.flush()
    if(not bool_anything_found):
        # print('Nothing found in this image')
        # save the image as it is
        IMAGE_PATH_DIR = os.path.join(SAVE_PATH, image_path.split('/')[-3], image_path.split('/')[-2])
        if not os.path.exists(IMAGE_PATH_DIR):
            os.makedirs(IMAGE_PATH_DIR)
        IMAGE_PATH_NEW = IMAGE_PATH_DIR + '/' + image_path.split('/')[-1][:-4] + r'_undetected.png'
        cv2.imwrite(IMAGE_PATH_NEW, image_np)
        # plt.figure(j,figsize=IMAGE_SIZE)
        # plt.imshow(image_np)
        pass

log.write('\nFertig.')
log.close()
