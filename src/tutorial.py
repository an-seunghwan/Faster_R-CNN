#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(False)
#%%
import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
# !pip install opencv-python
import cv2
import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
#%%
dataset_path = '/Users/anseunghwan/Downloads/VOCdevkit/VOC2007'
# dataset_path = r'D:\VOCdevkit\VOC2007'

IMAGE_FOLDER = "JPEGImages"
ANNOTATIONS_FOLDER = "Annotations"

ann_root, _, ann_files = next(os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))
img_root, _, img_files = next(os.walk(os.path.join(dataset_path, IMAGE_FOLDER)))

for f in ann_files[:3]:

    # XML파일와 이미지파일은 이름이 같으므로, 확장자만 맞춰서 찾습니다.
    img_name = img_files[img_files.index(".".join([f.split(".")[0], "jpg"]))]
    img_file = os.path.join(img_root, img_name)
    image = Image.open(img_file)
    print("Image size: ", np.array(image).shape)
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    xml = open(os.path.join(ann_root, f), "r")
    tree = Et.parse(xml)
    root = tree.getroot()

    size = root.find("size")

    width = size.find("width").text
    height = size.find("height").text
    channels = size.find("depth").text

    objects = root.findall("object")

    for _object in objects:
        name = _object.find("name").text
        bndbox = _object.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Box를 그릴 때, 왼쪽 상단 점과, 오른쪽 하단 점의 좌표를 입력으로 주면 됩니다.
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red")
        draw.text((xmin, ymin), name)

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.show()
    plt.close()
#%%
# parameters
image_height = 224
image_width  = 224
image_depth  = 3 # RGB
RPN_kernel_size = 3 # 3x3
subsampling_ratio = 8 # Pooling 3 times
anchor_sizes = [32, 64, 128]     
anchor_aspect_ratio = [[1, 1],[1/math.sqrt(2), math.sqrt(2)],[math.sqrt(2), 1/math.sqrt(2)]]
num_per_anchors = len(anchor_sizes) * len(anchor_aspect_ratio)
neg_threshold = 0.3
pos_threshold = 0.7
anchor_sampling_amount = 128 # 128 for each positive, negative sampling
#%%
'''
Input : dataset's annotations
Output : class label
'''
classes = []

for xml_file in ann_files: 

    xml = open(os.path.join(ann_root, xml_file), "r")
    tree = Et.parse(xml)
    root = tree.getroot()
    objects = root.findall("object")
    name = [object_.find("name").text.lower() for object_ in objects]
    classes.extend(name) 

classes = list(set(classes)) 
classes.sort()
classdict = {c:i for i,c in enumerate(classes)}
numclass = len(classes)
#%%
def get_labels_from_xml(xml_file, numclass = numclass):
    '''
    Input : 1 xml file
    (Get class label, get box coordinates)
    Because images are resized to 224x224, coordinates also need to be resized.
    Output: Existing class label, ground truth box coordinates
    '''

    xml = open(os.path.join(ann_root, xml_file), "r")
    tree = Et.parse(xml)
    root = tree.getroot()
    size = root.find("size")

    width = float(size.find("width").text)
    height = float(size.find("height").text)

    class_label = [] 
    bbox_label  = [] 
    
    objects = root.findall("object")
    
    for object_ in objects:
        ebj_class = object_.find("name").text.lower()
         # Get bounding box coordinates
        x_min = float(object_.find('bndbox').find('xmin').text) # top left x-axis coordinate.
        x_max = float(object_.find('bndbox').find('xmax').text) # bottom right x-axis coordinate.
        y_min = float(object_.find('bndbox').find('ymin').text) # top left y-axis coordinate.
        y_max = float(object_.find('bndbox').find('ymax').text) # bottom right y-axis coordinate.
        
        # Images resized to 224x224. So resize the coordinates
        x_min = float((image_width / width) * x_min)
        y_min = float((image_height / height) * y_min)
        x_max = float((image_width / width) * x_max)
        y_max = float((image_height / height) * y_max)
        box_info = [x_min, y_min, x_max, y_max] # [top-left, bottom-right]
        
        class_label.append(classdict.get(ebj_class))
        bbox_label.append(np.asarray(box_info, dtype='float32'))

    return class_label, np.asarray(bbox_label)

class_label, bbox_label = get_labels_from_xml(ann_files[0], numclass = numclass)
print(class_label)
print(bbox_label)
#%%
def generate_anchors(rpn_kernel_size=rpn_kernel_size, subsampled_ratio=subsampled_ratio,
                     anchor_sizes=anchor_sizes, anchor_aspect_ratio=anchor_aspect_ratio):
    '''
    Input : subsample_ratio (= Pooled ratio)
    generate anchor in feature map. Then project it to original image.
    Output : list of anchors (x,y,w,h) and anchor_boolean (ignore anchor if value equals 0)
    '''

    list_of_anchors = []
    anchor_booleans = [] # This is to keep track of an anchor's status. Anchors that are out of boundary are meant to be ignored.

    starting_center = divmod(rpn_kernel_size, 2)[0] # rpn kernel's starting center in feature map
    
    anchor_center = [starting_center - 1, starting_center] # -1 on the x-coor because the increment comes first in the while loop
    
    subsampled_height = image_height / subsampled_ratio # = 28
    subsampled_width = image_width / subsampled_ratio # = 28

    while (anchor_center != [subsampled_width - (1 + starting_center), subsampled_height - (1 + starting_center)]):  # != [26, 26]

        anchor_center[0] += 1 #Increment x-axis

        # If sliding window reached last center, increase y-axis
        if anchor_center[0] > subsampled_width - (1 + starting_center):
            anchor_center[1] += 1
            anchor_center[0] = starting_center

        # anchors are referenced to the original image. 
        # Therefore, multiply downsampling ratio to obtain input image's center 
        anchor_center_on_image = [anchor_center[0] * subsampled_ratio, anchor_center[1] * subsampled_ratio]

        for size in anchor_sizes:
            
            for a_ratio in anchor_aspect_ratio:
                # [x, y, w, h]
                anchor_info = [anchor_center_on_image[0], anchor_center_on_image[1], size*a_ratio[0], size*a_ratio[1]]
                # check whether anchor crosses the boundary of the image or not
                if (anchor_info[0] - anchor_info[2]/2 < 0 or anchor_info[0] + anchor_info[2]/2 > image_width or 
                                        anchor_info[1] - anchor_info[3]/2 < 0 or anchor_info[1] + anchor_info[3]/2 > image_height) :
                    anchor_booleans.append([0.0]) # if anchor crosses boundary, anchor_booleans = 0
                else:
                    anchor_booleans.append([1.0])

                list_of_anchors.append(anchor_info)
    
    return list_of_anchors, anchor_booleans
anchors, anchor_booleans = generate_anchors()
#%%
def generate_label(class_label, ground_truth_boxes, anchors, anchor_booleans, numclass=numclass,
                    neg_threshold = neg_threshold, pos_threshold = pos_threshold):
    '''
    Input : classes, ground truth box (top-left, bottom-right), all of anchors, anchor booleans.
    Compute IoU to get positive, negative samples.
    if IoU > 0.7, positive 
        IoU < 0.3, negative
        Otherwise, ignore
    Output : anchor booleans (to know which anchor to ignore), objectness label, regression coordinate in one image
    '''

    num_anchors = len(anchors) # Get the total number of anchors.
    anchor_boolean  = np.reshape(np.asarray(anchor_booleans), (num_anchors, 1))
    
    # IoU is more than threshold or not.
    objectness = np.zeros((num_anchors, 2), dtype=np.float32)
    # delta(x, y, w, h)
    box_regression = np.zeros((num_anchors, 4), dtype=np.float32)
    # belongs to which object for every anchor
    class_array = np.zeros((num_anchors, numclass), dtype=np.float32)
    
    for j in range(ground_truth_boxes.shape[0]):

        #Get the ground truth box's coordinates.
        gt_box_top_left_x = ground_truth_boxes[j][0]
        gt_box_top_left_y = ground_truth_boxes[j][1]
        gt_box_btm_rght_x = ground_truth_boxes[j][2]
        gt_box_btm_rght_y = ground_truth_boxes[j][3]

        # Calculate the area of the original bounding box
        gt_box_area = (gt_box_btm_rght_x - gt_box_top_left_x + 1) * (gt_box_btm_rght_y - gt_box_top_left_y + 1)
    
        for i in range(num_anchors):

            ######### Compute IoU #########

            # Check if the anchor should be ignored or not. If it is to be ignored, it crosses boundary of image.
            if int(anchor_boolean[i][0]) == 0:
                continue

            anchor = anchors[i] # Select the i-th anchor [x,y,w,h]

            # anchors are in [x,y,w,h] format, convert them to the [top-left-x, top-left-y, btm-right-x, btm-right-y]
            anchor_top_left_x = anchor[0] - anchor[2]/2
            anchor_top_left_y = anchor[1] - anchor[3]/2
            anchor_btm_rght_x = anchor[0] + anchor[2]/2
            anchor_btm_rght_y = anchor[1] + anchor[3]/2

            # Get the area of the bounding box.
            anchor_box_area = (anchor_btm_rght_x - anchor_top_left_x + 1) * (anchor_btm_rght_y - anchor_top_left_y + 1)

            # Determine the intersection rectangle.
            int_rect_top_left_x = max(gt_box_top_left_x, anchor_top_left_x)
            int_rect_top_left_y = max(gt_box_top_left_y, anchor_top_left_y)
            int_rect_btm_rght_x = min(gt_box_btm_rght_x, anchor_btm_rght_x)
            int_rect_btm_rght_y = min(gt_box_btm_rght_y, anchor_btm_rght_y)

            # if the boxes do not intersect, difference = 0
            int_rect_area = max(0, int_rect_btm_rght_x - int_rect_top_left_x + 1) * max(0, int_rect_btm_rght_y - int_rect_top_left_y)

            # Calculate the IoU
            intersect_over_union = float(int_rect_area / (gt_box_area + anchor_box_area - int_rect_area))
            
            if intersect_over_union >= pos_threshold:
                objectness[i][0] = 1.0 
                objectness[i][1] = 0.0 
                
                # get the class label
                class_array[i][class_label[j]] = 1.0 # Denote the label of the class in the array.
                
                # Get the ground-truth box's [x,y,w,h]
                gt_box_center_x = ground_truth_boxes[j][0] + ground_truth_boxes[j][2]/2
                gt_box_center_y = ground_truth_boxes[j][1] + ground_truth_boxes[j][3]/2
                gt_box_width    = ground_truth_boxes[j][2] - ground_truth_boxes[j][0]
                gt_box_height   = ground_truth_boxes[j][3] - ground_truth_boxes[j][1]

                # true value for Regression
                delta_x = (gt_box_center_x - anchor[0])/anchor[2]
                delta_y = (gt_box_center_y - anchor[1])/anchor[3]
                delta_w = math.log(gt_box_width / anchor[2])
                delta_h = math.log(gt_box_height / anchor[3])

                box_regression[i][0] = delta_x
                box_regression[i][1] = delta_y
                box_regression[i][2] = delta_w
                box_regression[i][3] = delta_h

            if intersect_over_union <= neg_threshold:
                if int(objectness[i][0]) == 0:
                    objectness[i][1] = 1.0

            if intersect_over_union > neg_threshold and intersect_over_union < pos_threshold:
                if int(objectness[i][0]) == 0 and int(objectness[i][1]) == 0:
                    anchor_boolean[i][0] = 0.0 # ignore this anchor

    return anchor_booleans, objectness, box_regression, class_array

anchor_booleans2, objectness, box_regression, class_array = generate_label(class_label, bbox_label, anchors, anchor_booleans)
#%%
def anchor_sampling(anchor_booleans2, objectness, anchor_sampling_amount=anchor_sampling_amount):
    '''
    Input : anchor booleans and objectness label
    fixed amount of negative anchors and positive anchors for training. 
    If we use all the neg and pos anchors, model will overfit on the negative samples.
    Output: Updated anchor booleans. 
    '''
    positive_count = 0
    negative_count = 0
    
    for i in range(objectness.shape[0]):
        if int(objectness[i][0]) == 1: # If the anchor is positive

            if positive_count > anchor_sampling_amount: # If the positive anchors are more than the threshold amount, set the anchor boolean to 0.

                anchor_booleans2[i][0] = 0.0

            positive_count += 1

        if int(objectness[i][1]) == 1: # If the anchor is negatively labelled.
            if negative_count > anchor_sampling_amount: #If the negative anchors are more than the threshold amount, set the boolean to 0.

                anchor_booleans2[i][0] = 0.0

            negative_count += 1
    
    return anchor_booleans2
#%%
def generate_dataset(first_index, last_index, anchors, anchor_booleans):
        '''
        Input : starting index and final index of the dataset to be generated.
        Output: Anchor booleans, Objectness Label and Regression Label in batches.
        '''
        numanchors = len(anchors)
        
        batch_anchor_booleans   = []
        batch_objectness_array  = []
        batch_regression_array  = []
        batch_class_label_array = []

        for i in range(first_index, last_index):

            # Get the true labels and the ground truth boxes [x,y,w,h] for every file.
            true_labels, ground_truth_boxes = get_labels_from_xml(ann_files[i])

            # generate_labels for specified batches
            anchor_bools, objectness_label_array, box_regression_array, class_array = generate_label(true_labels, ground_truth_boxes, anchors, anchor_booleans)

            # get the updated anchor bools based on the fixed number of sample
            anchor_bools = anchor_sampling(anchor_bools, objectness_label_array)
            
            batch_anchor_booleans.append(anchor_bools)
            batch_objectness_array.append(objectness_label_array)
            batch_regression_array.append(box_regression_array)
            batch_class_label_array.append(class_array)

        batch_anchor_booleans   = np.reshape(np.asarray(batch_anchor_booleans), (-1, numanchors)) # (1, 6084, 1) -> (1, 6084)
        batch_objectness_array  = np.asarray(batch_objectness_array)
        batch_regression_array  = np.asarray(batch_regression_array)
        batch_class_label_array = np.asarray(batch_class_label_array)

        return (batch_anchor_booleans, batch_objectness_array, batch_regression_array, batch_class_label_array)
#%%
a,b,c,d = generate_dataset(0, 1, anchors, anchor_booleans2)
a.shape
b.shape
c.shape
d.shape
#%%
def read_images(first_index, last_index):
    '''
    Read the image files, then resize.
    Input : first and last index.
    Output: numpy array of images.
    '''
    images_list = []
    
    for i in range(first_index, last_index):
        
        im = cv2.imread(os.path.join(img_root, img_files[i]))
        im = cv2.resize(im, (image_height, image_width)) / 255
        
        images_list.append(im)
    
    return np.asarray(images_list)
#%%
anchors, an_bools = generate_anchors() # We only need to generate the anchors and the anchor booleans once.
numanchors = len(anchors)
#%%
'''
modelling
'''
img_input = K.Input((image_height, image_width, image_depth))

conv1 = K.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='SAME', activation='relu')
conv1_pool = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')
h = conv1_pool(conv1(img_input))

conv2 = K.layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='SAME', activation='relu')
conv2_pool = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')
h = conv2_pool(conv2(h)) 

conv3 = K.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='SAME', activation='relu')
conv3_pool = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')
h = conv3_pool(conv3(h)) # feature map, 28x28x256

conv_rpn = K.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='VALID', activation='relu')
sliding_window = conv_rpn(h) # 26x26x512

conv_cls = K.layers.Conv2D(filters=18, kernel_size=1, strides=(1, 1), padding='VALID', activation='linear')
conv_reg = K.layers.Conv2D(filters=36, kernel_size=1, strides=(1, 1), padding='VALID', activation='linear')

cls_output = conv_cls(sliding_window) # 26x26x18
reg_output = conv_reg(sliding_window) # 26x26x36

cls_output = tf.reshape(cls_output, (-1, numanchors, 2)) # 6084x2
reg_output = tf.reshape(reg_output, (-1, numanchors, 4)) # 6084x4

# cls_logit = tf.nn.softmax(cls_output, axis=-1)

model = K.models.Model(img_input, [cls_output, reg_output])
model.summary()
#%%
'''
loss function
'''
def smooth_L1(reg_pred, reg_true):
    diff = reg_pred - reg_true
    return tf.where(tf.math.abs(diff) < 1, 0.5 * tf.math.pow(diff, 2), diff-0.5)

def loss_function(cls_pred, cls_true, reg_pred, reg_true):
    # 평균 계산 어떻게?
    cls_loss = tf.reduce_mean(tf.multiply(batch_anchor_booleans, 
                                        tf.nn.softmax_cross_entropy_with_logits(cls_pred, cls_true)))

    reg_loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(batch_objectness_array, axis=-1), 
                                        tf.reduce_sum(smooth_L1(reg_pred, reg_true), axis=-1)))

    loss = cls_loss + lambda_ * reg_loss
    return loss
#%%
'''
training
'''
learning_rate = 1e-5
epochs = 100
batch_size = 1
decay_steps = 10000
decay_rate = 0.99
lambda_ = 10

optimizer = tf.keras.optimizers.RMSprop(learning_rate)

# TRAINING 
for epoch in range(epochs): # Each epoch.
    
    # Loop through the whole dataset in batches.
    for start_idx in tqdm(range(0, total_images, batch_size)):
        
        end_idx = start_idx + batch_size
        
        if end_idx >= total_images : end_idx = total_images - 1 # In case the end index exceeded the dataset.
            
        images = read_images(start_idx, end_idx) # Read images.
        
        with tf.GradientTape() as tape:
            # Get the labels needed.
            batch_anchor_booleans, batch_objectness_array, batch_regression_array, _ = generate_dataset(start_idx,end_idx, anchors, an_bools)
            
            cls_result, reg_result = model(images)
            
            loss = loss_function(cls_result, batch_objectness_array, reg_result, batch_regression_array)
        
        grad = tape.gradient(loss, model.weights)
        optimizer.apply_gradients(zip(grad, model.weights))
        
    print("Epoch:", epoch, ", TRAIN loss:", loss.numpy())
    print('\n')
#%%
img_array = read_images(1, 2)
anchor_prob, anchor_box = model(img_array)
boxes = anchor_box[0]
top_anchor = np.argsort(anchor_prob.numpy()[0][:, 0])[-5:]

fig, ax = plt.subplots()
ax.imshow(img_array[0])
for j in top_anchor:
    box = boxes.numpy()[j, :]
    x = box[0] * anchors[j][2] + anchors[j][0]
    y = box[1] * anchors[j][3] + anchors[j][1]
    w = math.exp(box[2]) * anchors[j][2]
    h = math.exp(box[3]) * anchors[j][3]
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
plt.show()
#%%