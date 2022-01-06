from matplotlib.pyplot import box
import scaleapi
from scaleapi.tasks import TaskType
from scaleapi.api import Api
import json
import os
import nucleus
from collections import defaultdict

api_key = 'live_3e2bb78567b3412594f68b0656d99aa1' # Put your own API Key here
project_name = 'Finding-Potential-Mislabels-ImagenetVal'
instruction_batch_name = 'test-instruction-batch'
regular_batch_name = 'test-regular-batch'

scaleapi_client = scaleapi.ScaleClient(api_key)
nucleus_client = nucleus.NucleusClient(api_key)

#Create Rapid project
project = scaleapi_client.create_project(
    project_name = project_name,
    task_type = TaskType.ImageAnnotation)
project = scaleapi_client.get_project(project_name = "test-blog")


dataset = nucleus_client.get_dataset("ds_c44fazkp5x6007gssc10") # Your dataset id
dataset_items = dataset.items
slice = nucleus_client.get_slice("slc_c4dqe1r4j30g099bc5n0") # Your dataset slice
slice_items = slice.info()["dataset_items"]

#Right now there's no way to persist metadata when using Nucleus to send to annotations, this code block imports the file from Nucleus and uploads to Rapid with the relevant metadata.
for s_item in slice_items:
    item = dataset.refloc(s_item["ref_id"])
    box_annotation = item["annotations"]["box"][0]
    given_label = box_annotation.label
    predicted_label = box_annotation.metadata["predicted_label"]
    coarse_label = box_annotation.metadata["coarse_label"]
    mislabel_score = s_item["metadata"]["mislabelScore"]
    predicted_coarse_label = box_annotation.metadata["predicted_coarse_label"]
    data = scaleapi_client.import_file(s_item["original_image_url"], project_name=project_name, metadata=json.dumps({"given_label":given_label, "predicted_label":predicted_label, "coarse_label":coarse_label, "predicted_coarse_label":predicted_coarse_label, "mislabel_score": mislabel_score}))

example_label_set = defaultdict(int)
number_of_examples_per_class_to_have = 5
for d_item in dataset_items:
    item = dataset.refloc(d_item.reference_id)
    box_annotation = item["annotations"]["box"][0]
    is_mislabel = box_annotation.metadata["potential_mislabel"]
    if not is_mislabel:
        given_label = box_annotation.label
        if example_label_set[given_label] < number_of_examples_per_class_to_have:
            data = scaleapi_client.import_file(item["item"].image_location, project_name=project_name, metadata=json.dumps({"given_label":given_label}))
            example_label_set[given_label] += 1
