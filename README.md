## UND duck detector training

### Overview

The code in this repo trains and evaluates models to detect ducks and other waterfowl in drone images, based on a dataset provided by the [Felege Wildlife Lab](https://arts-sciences.und.edu/academics/biology/susan-felege/).

There are around 35,000 images total, about 95% of which contain no birds.  Images are around 5000x3000.  A typical ground truth image looks like this:

<img src="sample_image.jpg" width="800px;"><br/>

The annotations you can vaguely see as different colors correspond to different species of bird.  Most of this repo operates on 1280x1280 patches that look like this (this is also a ground truth patch, not a model prediction):

<img src="annotated_patch.jpg" width="800px;"><br/>

### Files

These are listed in roughly the order in which you would use them.

#### und-ducks-data-import.py

* Match images to annotation files
* Read the original annotations in Pascal VOC format
* Convert to COCO format
* Do a bunch of miscellaneous consistency checking

#### und-ducks-training-data-prep.py

* For all the images with at least one annotation, slice into mostly-non-overlapping patches
* Optionally sample hard negatives (I did not end up actually using any hard negatives)
* Split into train/val
* Export to YOLO annotation format

#### und-ducks-training.py

* Train the model (training happens at the YOLOv5 CLI, but this script documents the commands)
* Run the YOLOv5 validation scripts
* Convert YOLOv5 val results to MD .json format
* Example code to use the MD visualization pipeline to visualize results
* Example code to use the MD inference pipeline to run the trained model

### Running the models, and doing stuff with the results

Much of the code in this repo is shared with the [Izembek Brant Goose Detector](https://github.com/agentmorris/usgs-geese) repo; the training code is bespoke for each model, but the inference code is identical.  So, for instructions on running the model and postprocessing the results, see the "[Running the model](https://github.com/agentmorris/usgs-geese#running-the-model)" section on the Izembek Detector repo.

