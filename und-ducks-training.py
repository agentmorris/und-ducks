########
#
# und-ducks-training.py
#
# This file documents the model training process, starting from where und-ducks-training-data-prep.py
# leaves off.  Training happens at the yolov5 CLI, and the exact command line arguments are documented
# in the "Train" cell.
#
# Later cells in this file also:
#
# * Run the YOLOv5 validation scripts
# * Convert YOLOv5 val results to MD .json format
# * Use the MD visualization pipeline to visualize results
# * Use the MD inference pipeline to run the trained model
#
########

#%% Train

## Environment prep

"""
mamba create --name yolov5
mamba activate yolov5
mamba install pip
git clone https://github.com/ultralytics/yolov5 yolov5-current
cd yolov5-current
pip install -r requirements.txt
"""

#
# I got this error:
#    
# OSError: /home/user/anaconda3/envs/yolov5/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas.so.11: undefined symbol: cublasLtGetStatusString, version libcublasLt.so.11
#
# There are two ways I've found to fix this:
#
# CUDA was on my LD_LIBRARY_PATH, so this fixes it:
#
# LD_LIBRARY_PATH=
#
# Or if I do this:
# 
# pip uninstall nvidia_cublas_cu11
#
# ...when I run train.py again, it reinstalls the missing CUDA components,
# and everything is fine, but then the error comes back the *next* time I run it.
#
# So I pip uninstall again, and the circle of life continues.
#


## Training

# mamba activate yolov5

"""
cd ~/git/yolov5-current

# I usually have an older commit of yolov5 on my PYTHONPATH, remove it.
export PYTHONPATH=
LD_LIBRARY_PATH=

# On my 2x24GB GPU setup, a batch size of 16 failed, but 8 was safe.  Autobatch did not
# work; I got an incomprehensible error that I decided not to fix, but I'm pretty sure
# it would have come out with a batch size of 8 anyway.
BATCH_SIZE=8
IMAGE_SIZE=1280
EPOCHS=100
CLASS_MAPPING="multiclass" # binary,multiclass
DATA_YAML_FILE="/home/user/data/und-ducks/und-ducks-${CLASS_MAPPING}-yolov5/dataset.yaml"

TRAINING_RUN_NAME=und-ducks-yolov5x6-${CLASS_MAPPING}-b${BATCH_SIZE}-img${IMAGE_SIZE}-e${EPOCHS}
echo ${TRAINING_RUN_NAME}
echo ${DATA_YAML_FILE}
"""

"""
python train.py --img ${IMAGE_SIZE} --batch ${BATCH_SIZE} --epochs ${EPOCHS} --weights yolov5x6.pt --device 0,1 --project und-ducks --name ${TRAINING_RUN_NAME} --data ${DATA_YAML_FILE}
"""


## Resuming training

"""
cd ~/git/yolov5-current
mamba activate yolov5
LD_LIBRARY_PATH=
export PYTHONPATH=
python train.py --resume
"""

pass


#%% Make plots during training

import os
import pandas as pd
import matplotlib.pyplot as plt

results_file = os.path.expanduser('~/git/yolov5-current/und-ducks/und-ducks-yolov5x6-binary-b8-img1280-e100/results.csv')

df = pd.read_csv(results_file)
df = df.rename(columns=lambda x: x.strip())
    
fig,ax = plt.subplots()

df.plot(x = 'epoch', y = 'val/box_loss', ax = ax) 
df.plot(x = 'epoch', y = 'val/obj_loss', ax = ax, secondary_y = True) 

df.plot(x = 'epoch', y = 'train/box_loss', ax = ax) 
df.plot(x = 'epoch', y = 'train/obj_loss', ax = ax, secondary_y = True) 

plt.show()

fig,ax = plt.subplots()

df.plot(x = 'epoch', y = 'val/cls_loss', ax = ax) 
df.plot(x = 'epoch', y = 'train/cls_loss', ax = ax) 

plt.show()

fig,ax = plt.subplots()

df.plot(x = 'epoch', y = 'metrics/precision', ax = ax) 
df.plot(x = 'epoch', y = 'metrics/recall', ax = ax) 
df.plot(x = 'epoch', y = 'metrics/mAP_0.5', ax = ax) 
df.plot(x = 'epoch', y = 'metrics/mAP_0.5:0.95', ax = ax) 

plt.show()

# plt.close('all')


#%% Back up trained weights

"""
CLASS_MAPPING="multiclass" # binary,multiclass
TRAINING_RUN_NAME="und-ducks-yolov5x6-${CLASS_MAPPING}-b8-img1280-e100"
TRAINING_OUTPUT_FOLDER="/home/user/git/yolov5-current/und-ducks/${TRAINING_RUN_NAME}/weights"

echo $TRAINING_OUTPUT_FOLDER
echo $TRAINING_RUN_NAME

cp ${TRAINING_OUTPUT_FOLDER}/best.pt ~/models/und-ducks/${TRAINING_RUN_NAME}-best.pt
cp ${TRAINING_OUTPUT_FOLDER}/last.pt ~/models/und-ducks/${TRAINING_RUN_NAME}-last.pt
"""

pass


#%% Run models on validation data

import importlib
usgs_geese_inference = importlib.import_module('usgs-geese-inference')

model_folder = os.path.expanduser('~/models/und-ducks')
input_path = os.path.expanduser('~/data/und-ducks-lila/images')
yolo_working_dir = os.path.expanduser('~/git/yolov5-current')
class_mappings = ['binary','multiclass']

assert os.path.isdir(input_path)
assert os.path.isdir(yolo_working_dir)
assert os.path.isdir(model_folder)

commands = []

# i_class_mapping = 0; class_mapping = class_mappings[i_class_mapping]
for i_class_mapping,class_mapping in enumerate(class_mappings):
    
    training_run_name = 'und-ducks-yolov5x6-{}-b8-img1280-e100'.format(class_mapping)
    model_file = os.path.join(model_folder,training_run_name + '-best.pt')
    dataset_file_yaml = os.path.expanduser(
        '~/data/und-ducks/und-ducks-{}-yolov5/dataset.yaml'.format(class_mapping))
    dataset_file_json = os.path.expanduser(
        '~/data/und-ducks/und-ducks-{}-yolov5/und-ducks-md-class-mapping-{}.json'.format(
            class_mapping,class_mapping))
    output_dir = os.path.expanduser('~/tmp/und-ducks/inference-{}'.format(class_mapping))
    scratch_dir = os.path.join(output_dir,'scratch')
    
    assert os.path.isfile(model_file)
    assert os.path.isfile(dataset_file_yaml)
    assert os.path.isfile(dataset_file_json)

    cmd = "export LD_LIBRARY_PATH='' && export PYTHONPATH='/home/user/git/MegaDetector' && "
    # cmd += 'CUDA_VISIBLE_DEVICES={} '.format(i_class_mapping)    
    cmd += 'python run-izembek-model.py '
    cmd += '{} {} {} {} '.format(
        model_file,input_path,yolo_working_dir,scratch_dir)

    device_index = i_class_mapping
    
    # These should produce the same result
    category_mapping_file = dataset_file_yaml     
    # category_mapping_file = dataset_file_json
        
    output_file = os.path.join(output_dir,'und-ducks-{}-inference-results.json'.format(class_mapping))
    cmd += '--output_file {} --recursive --category_mapping_file {} --device {}'.format(
        output_file, category_mapping_file, device_index)
    
    commands.append(cmd)

for cmd in commands:
    print('{}\n'.format(cmd))

# import clipboard; clipboard.copy(commands[0])    
# import clipboard; clipboard.copy(commands[1])


#%% Generate a copy of the results that excludes training images

import json

# i_class_mapping = 0; class_mapping = class_mappings[i_class_mapping]
for i_class_mapping,class_mapping in enumerate(class_mappings):

    train_image_list_file = os.path.expanduser(
        '~/data/und-ducks/und-ducks-{}-yolov5/train_images.json'.format(class_mapping))
    val_image_list_file = os.path.expanduser(
        '~/data/und-ducks/und-ducks-{}-yolov5/val_images.json'.format(class_mapping))
    
    with open(train_image_list_file,'r') as f:
        train_images = set(json.load(f))
    with open(val_image_list_file,'r') as f:
        val_images = set(json.load(f))
    
    output_dir = os.path.expanduser('~/tmp/und-ducks/inference-{}'.format(class_mapping))
    output_file = os.path.join(output_dir,'und-ducks-{}-inference-results.json'.format(class_mapping))
    assert os.path.isfile(output_file)
    
    with open(output_file,'r') as f:
        inference_results = json.load(f)
    
    train_images_in_results = []
    val_images_in_results = []
    eval_images = []
    
    # im = inference_results['images'][0]
    for im in inference_results['images']:
        if im['file'] in train_images:
            train_images_in_results.append(im['file'])
        else:
            # Include images in the eval set whether they're in val or not, as long as they are not in
            # train
            eval_images.append(im)
            if im['file'] in val_images:
                val_images_in_results.append(im['file'])                
    
    print(
        'For mapping {}, of {} images in results, {} are in train (of {} train images), {} are in val (of {} val images)'.\
            format(class_mapping,len(inference_results['images']),
                len(train_images_in_results),len(train_images),
                len(val_images_in_results),len(val_images)))
    
    print('Writing out results for {} eval images'.format(len(eval_images)))

    eval_images_results_file = output_file.replace('.json','_eval_only.json')
    assert eval_images_results_file != output_file
    
    inference_results['images'] = eval_images
    
    with open(eval_images_results_file,'w') as f:
        json.dump(inference_results,f,indent=1)
    
    
#%% Generate previews and counts

commands = []

# i_class_mapping = 0; class_mapping = class_mappings[i_class_mapping]
for i_class_mapping,class_mapping in enumerate(class_mappings):

    output_dir = os.path.expanduser('~/tmp/und-ducks/inference-{}'.format(class_mapping))
    output_file = os.path.join(output_dir,'und-ducks-{}-inference-results.json'.format(class_mapping))
    eval_images_results_file = output_file.replace('.json','_eval_only.json')
    assert os.path.isfile(eval_images_results_file)

    count_csv_output_file = eval_images_results_file.replace('.json','_counts.csv')
    n_patches = 2000
    preview_folder = os.path.join(output_dir,'previews')
    
    cmd = 'python izembek-model-postprocessing.py '
    cmd += '{} '.format(eval_images_results_file)
    cmd += '--image_folder {} '.format(input_path)
    cmd += '--count_file {} '.format(count_csv_output_file)
    cmd += '--preview_folder {} '.format(preview_folder)
    cmd += '--n_patches {} '.format(n_patches)
    cmd += '--confidence_thresholds 0.2 0.3 0.4 0.5 0.6 0.7 '

    commands.append(cmd)

for cmd in commands:
    print('{}\n'.format(cmd))

# import clipboard; clipboard.copy(commands[1])


#%% Debug scrap

if False:

    #%%
    
    mapping = usgs_geese_inference.read_classes_from_yolo_dataset_file(dataset_file_yaml)
    inference_options = usgs_geese_inference.USGSGeeseInferenceOptions(
        project_dir=scratch_dir,
        yolo_working_dir=yolo_working_dir,
        model_file=model_file,
        use_symlinks=True,
        no_augment=False,
        no_cleanup=False,
        devices=[device_index],
        category_mapping_file=category_mapping_file)

    print(inference_options.yolo_category_id_to_name)
