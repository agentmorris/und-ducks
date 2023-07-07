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

"""
cd ~/git/yolov5-current

# I usually have an older commit of yolov5 on my PYTHONPATH, remove it.
export PYTHONPATH=
LD_LIBRARY_PATH=
mamba activate yolov5

# On my 2x24GB GPU setup, a batch size of 16 failed, but 8 was safe.  Autobatch did not
# work; I got an incomprehensible error that I decided not to fix, but I'm pretty sure
# it would have come out with a batch size of 8 anyway.
BATCH_SIZE=8
IMAGE_SIZE=1280
EPOCHS=200
DATA_YAML_FILE=/home/user/data/und-ducks/dataset.yaml

# TRAINING_RUN_NAME=und-ducks-yolov5x6-b${BATCH_SIZE}-img${IMAGE_SIZE}-e${EPOCHS}
TRAINING_RUN_NAME=und-ducks-yolov5x6-nolinks-b${BATCH_SIZE}-img${IMAGE_SIZE}-e${EPOCHS}

python train.py --img ${IMAGE_SIZE} --batch ${BATCH_SIZE} --epochs ${EPOCHS} --weights yolov5x6.pt --device 0,1 --project und-ducks --name ${TRAINING_RUN_NAME} --data ${DATA_YAML_FILE}
"""


## Monitoring training

"""
cd ~/git/yolov5-current
mamba activate yolov5
tensorboard --logdir usgs-ducks
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

results_file = os.path.expanduser('~/git/yolov5-current/und-ducks/und-ducks-yolov5x6-nolinks-b8-img1280-e200/results.csv')

df = pd.read_csv(results_file)
df = df.rename(columns=lambda x: x.strip())
    
fig,ax = plt.subplots()

df.plot(x = 'epoch', y = 'val/box_loss', ax = ax) 
df.plot(x = 'epoch', y = 'val/obj_loss', ax = ax, secondary_y = True) 

df.plot(x = 'epoch', y = 'train/box_loss', ax = ax) 
df.plot(x = 'epoch', y = 'train/obj_loss', ax = ax, secondary_y = True) 

plt.show()


#%% Back up trained weights

"""
TRAINING_RUN_NAME="und-ducks-yolov5x6-nolinks-b8-img1280-e200"
TRAINING_OUTPUT_FOLDER="/home/user/git/yolov5-current/und-ducks/${TRAINING_RUN_NAME}/weights"

cp ${TRAINING_OUTPUT_FOLDER}/best.pt ~/models/und-ducks/${TRAINING_RUN_NAME}-best.pt
cp ${TRAINING_OUTPUT_FOLDER}/last.pt ~/models/und-ducks/${TRAINING_RUN_NAME}-last.pt
"""

pass


#%% Validation with YOLOv5

import os

model_base = os.path.expanduser('~/models/und-ducks')
training_run_names = [
    'und-ducks-yolov5x6-nolinks-b8-img1280-e200'    
]

data_folder = os.path.expanduser('~/data/und-ducks')
image_size = 1280

# Note to self: validation batch size appears to have no impact on mAP
# (it shouldn't, but I verified that explicitly)
batch_size_val = 8

project_name = os.path.expanduser('~/tmp/und-ducks-val')
data_file = os.path.join(data_folder,'dataset.yaml')
augment = True

assert os.path.isfile(data_file)

model_file_to_command = {}

# training_run_name = training_run_names[0]
for training_run_name in training_run_names:
    model_file_base = os.path.join(model_base,training_run_name)
    model_files = [model_file_base + s for s in ('-last.pt','-best.pt')]
    
    # model_file = model_files[0]
    for model_file in model_files:
        assert os.path.isfile(model_file)
        
        model_short_name = os.path.basename(model_file).replace('.pt','')
        cmd = 'python val.py --img {} --batch-size {} --weights {} --project {} --name {} --data {} --save-txt --save-json --save-conf --exist-ok'.format(
            image_size,batch_size_val,model_file,project_name,model_short_name,data_file)        
        if augment:
            cmd += ' --augment'
        model_file_to_command[model_file] = cmd
        
    # ...for each model
    
# ...for each training run    

for k in model_file_to_command.keys():
    # print(os.path.basename(k))
    print('')
    cmd = model_file_to_command[k]
    print(cmd + '\n')


"""
cd ~/git/yolov5-current
mamba activate yolov5
LD_LIBRARY_PATH=
export PYTHONPATH=
"""

pass


#%% Convert YOLO val .json results to MD .json format

# pip install jsonpickle humanfriendly tqdm skicit-learn

import os
from data_management import yolo_output_to_md_output

import json
import glob

class_mapping_file = os.path.expanduser('~/data/und-ducks/und-ducks-md-class-mapping.json')
with open(class_mapping_file,'r') as f:
    category_id_to_name = json.load(f)
                        
base_folder = os.path.expanduser('~/tmp/und-ducks-val')
run_folders = os.listdir(base_folder)
run_folders = [os.path.join(base_folder,s) for s in run_folders]
run_folders = [s for s in run_folders if os.path.isdir(s)]

image_base = os.path.expanduser('~/data/und-ducks/yolo_val')
image_files = glob.glob(image_base + '/*.jpg')

prediction_files = []

# run_folder = run_folders[0]
for run_folder in run_folders:
    prediction_files_this_folder = glob.glob(run_folder+'/*_predictions.json')
    assert len(prediction_files_this_folder) <= 1
    if len(prediction_files_this_folder) == 1:
        prediction_files.append(prediction_files_this_folder[0])        

md_format_prediction_files = []

# prediction_file = prediction_files[0]
for prediction_file in prediction_files:

    detector_name = os.path.splitext(os.path.basename(prediction_file))[0].replace('_predictions','')
    
    # print('Converting {} to MD format'.format(prediction_file))
    output_file = prediction_file.replace('.json','_md-format.json')
    assert output_file != prediction_file
    
    yolo_output_to_md_output.yolo_json_output_to_md_output(
        yolo_json_file=prediction_file,
        image_folder=image_base,
        output_file=output_file,
        yolo_category_id_to_name=category_id_to_name,                              
        detector_name=detector_name,
        image_id_to_relative_path=None,
        offset_yolo_class_ids=False)    
    
    md_format_prediction_files.append(output_file)

# ...for each prediction file


#%% Visualize results with the MD visualization pipeline

postprocessing_output_folder = os.path.expanduser('~/tmp/und-ducks-val-previews')

from md_utils import path_utils

from api.batch_processing.postprocessing.postprocess_batch_results import (
    PostProcessingOptions, process_batch_results)

# prediction_file = md_format_prediction_files[0]
for prediction_file in md_format_prediction_files:
    
    assert '_md-format.json' in prediction_file
    base_task_name = os.path.basename(prediction_file).replace('_md-format.json','')

    options = PostProcessingOptions()
    options.image_base_dir = image_base
    options.include_almost_detections = True
    options.num_images_to_sample = None
    options.confidence_threshold = 0.25
    options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
    options.ground_truth_json_file = None
    options.separate_detections_by_category = False
    # options.sample_seed = 0
    
    options.parallelize_rendering = True
    options.parallelize_rendering_n_cores = 16
    options.parallelize_rendering_with_threads = False
    
    output_base = os.path.join(postprocessing_output_folder,
        base_task_name + '_{:.3f}'.format(options.confidence_threshold))
    
    os.makedirs(output_base, exist_ok=True)
    print('Processing to {}'.format(output_base))
    
    options.api_output_file = prediction_file
    options.output_dir = output_base
    ppresults = process_batch_results(options)
    html_output_file = ppresults.output_html_file
    path_utils.open_file(html_output_file)

# ...for each prediction file


#%%

#
# Run the MD pred pipeline 
#

"""
export PYTHONPATH=/home/user/git/MegaDetector:/home/user/git/yolov5-current
cd ~/git/MegaDetector/detection/
mamba activate yolov5

TRAINING_RUN_NAME="und-ducks-yolov5x6-nolinks-b8-img1280-e200-best"
MODEL_FILE="/home/user/models/und-ducks/${TRAINING_RUN_NAME}.pt"
DATA_FOLDER="/home/user/data/und-ducks"
RESULTS_FOLDER="/home/user/tmp/und-ducks/md-pipeline-results"
CLASS_MAPPING_FILE="/home/user/data/und-ducks/und-ducks-md-class-mapping.json"

python run_detector_batch.py ${MODEL_FILE} ${DATA_FOLDER}/yolo_val ${RESULTS_FOLDER}/${TRAINING_RUN_NAME}-val.json --recursive --quiet --output_relative_filenames --class_mapping_filename ${CLASS_MAPPING_FILE}

python run_detector_batch.py ${MODEL_FILE} ${DATA_FOLDER}/yolo_train ${RESULTS_FOLDER}/${TRAINING_RUN_NAME}-train.json --recursive --quiet --output_relative_filenames --class_mapping_filename ${CLASS_MAPPING_FILE}
"""

#
# Visualize results using the MD pipeline
#

"""
mamba activate cameratraps-detector
cd ~/git/MegaDetector/api/batch_processing/postprocessing/

TRAINING_RUN_NAME="und-ducks-yolov5x6-nolinks-b8-img1280-e200-best"
DATA_FOLDER="/home/user/data/und-ducks"
RESULTS_FOLDER="/home/user/tmp/und-ducks/md-pipeline-results"
PREVIEW_FOLDER="${DATA_FOLDER}/preview"
CONF_THRESH=0.3

python postprocess_batch_results.py ${RESULTS_FOLDER}/${TRAINING_RUN_NAME}-val.json ${PREVIEW_FOLDER}/${TRAINING_RUN_NAME}-val --image_base_dir ${DATA_FOLDER}/yolo_val --n_cores 12 --confidence_threshold ${CONF_THRESH} --parallelize_rendering_with_processes --no_separate_detections_by_category

python postprocess_batch_results.py ${RESULTS_FOLDER}/${TRAINING_RUN_NAME}-train.json ${PREVIEW_FOLDER}/${TRAINING_RUN_NAME}-train --image_base_dir ${DATA_FOLDER}/yolo_train --n_cores 12 --confidence_threshold ${CONF_THRESH} --parallelize_rendering_with_processes --no_separate_detections_by_category

xdg-open ${PREVIEW_FOLDER}/${TRAINING_RUN_NAME}-val/index.html 
xdg-open ${PREVIEW_FOLDER}/${TRAINING_RUN_NAME}-train/index.html
"""

pass
