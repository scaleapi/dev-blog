#This file uploads images and annotations to Nucleus

import nucleus
from nucleus import DatasetItem, BoxAnnotation, BoxPrediction
import cleanlab
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import copy
from imagenet_helper import imagenet


API_KEY = 'live_3e2bb78567b3412594f68b0656d99aa1' # Put your own API Key here
client = nucleus.NucleusClient(API_KEY, use_notebook=False)
dataset_name = "Test"
#dataset_name = "Imagenet Validation with Cleanlab (predicted labels, mislablel score, coarse label)"
nucleus_dataset = client.create_dataset(dataset_name)

ALL_CLASSES = {
    'imagenet_val_set': imagenet
}

#Load in save model predictions for Imagenet Val. Look here to see in detail how this is done: https://github.com/cgnorthcutt/cleanlab/tree/master/examples
dataset = "imagenet_val_set"
labels = np.load('{}_original_labels.npy'.format(dataset), allow_pickle=True)
n_parts = 4
pyx_fn = '{}_pyx.part{}_of_{}.npy'
parts = [np.load(pyx_fn.format(dataset, i + 1, n_parts)) for i in range(n_parts)]
pyx = np.vstack(parts)
pred = np.load('{}_pyx_argmax_predicted_labels.npy'.format(dataset), allow_pickle=True)

label_error_indices = cleanlab.pruning.get_noise_indices(
        s=labels,
        psx=pyx,
        prune_method='prune_by_noise_rate',
        multi_label=False,
        sorted_index_method='self_confidence',
    )
num_errors = len(label_error_indices)
print('Estimated number of errors in {}:'.format(dataset), num_errors)
    
self_confidence = np.array([np.mean(pyx[i][labels[i]]) for i in label_error_indices])
margin = self_confidence - pyx[label_error_indices].max(axis=1)
inorder_indices = label_error_indices[np.argsort(margin)]
inorder_margin = margin[np.argsort(margin)]
label_error_dict = {inorder_indices[i]:inorder_margin[i] for i in range(len(inorder_indices))}

#Load in coarse label mappings
coarse_label_mappings = np.load('coarse_label_mappings.npy', allow_pickle='TRUE').item()

#Upload Images to Nucleus
root_dir = "/home/samyakparajuli/"
imagenet_path = root_dir + "data/imagenet/" #parent path of "val" directory
with open("imagenet_val_set_index_to_filepath.json", 'r') as rf:
    IMAGENET_INDEX_TO_FILEPATH = json.load(rf)

imagenet_val_list = []
for i, image_dir in enumerate(IMAGENET_INDEX_TO_FILEPATH):
    image_name = os.path.basename(image_dir)
    ref = image_name[15:]
    image_dir = imagenet_path + image_dir
    print(image_dir)
    mislabel_score = 0
    if i in label_error_indices:
          mislabel_score = label_error_dict[i]
    item = DatasetItem(
            image_location=image_dir, 
            reference_id=ref,
            metadata={"mislabel_score": mislabel_score}
        )
    imagenet_val_list.append(item)

nucleus_dataset.append(imagenet_val_list)

#Upload Annotations to Nucleus
imagenet_val_annotations = []
for i, image_dir in enumerate(IMAGENET_INDEX_TO_FILEPATH):
     image_name = os.path.basename(image_dir)
     ref = image_name[15:]
     image_dir = imagenet_path + image_dir
     image = plt.imread(image_dir)
     try:
          width, height, channel = image.shape
     except:
          width, height = image.shape
     mislabel_flag = 0
     given_label = ALL_CLASSES["imagenet_val_set"][labels[i]]
     pred_label = given_label
     if i in label_error_indices:
          mislabel_flag = 1
          pred_label = ALL_CLASSES["imagenet_val_set"][pred[i]]
     coarse_label = coarse_label_mappings[given_label]
     predicted_coarse_label = coarse_label_mappings[pred_label]
     annotation = BoxAnnotation(
                label=given_label, 
                x=0, 
                y=0, 
                width=width, 
                height=height, 
                reference_id=ref,
                metadata={"potential_mislabel": mislabel_flag, "predicted_label": pred_label, "coarse_label": coarse_label, "predicted_coarse_label":  predicted_coarse_label}
            )
     imagenet_val_annotations.append(annotation)
response = nucleus_dataset.annotate(imagenet_val_annotations, update=True, asynchronous=True)
print(response)