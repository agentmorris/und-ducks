########
#
# usgs-geese-training-data-prep.py
#
# Given the COCO-formatted data created by usgs-data-review.py, cut the source
# data into patches, write those patches out as YOLO-formatted annotations, and
# copy files into train/test folders.
#
# Assumes a fixed patch size, and just slides this along each image looking for
# overlapping annotations.  In general there is no overlap between patches, but
# the rightmost column and bottommost row are shifted left and up to stay at the
# desired patch size, so geese near the right and bottom of each image will be 
# sampled twice.
#
########

#%% Constants and imports

import os
import json
import shutil
import random

from collections import defaultdict
from tqdm import tqdm

from md_visualization import visualization_utils as visutils
from md_utils.path_utils import safe_create_link
from detection.run_tiled_inference import get_patch_boundaries, patch_info_to_patch_name

input_annotations_file = os.path.expanduser('~/data/und-ducks.json')
input_image_folder = '/media/user/My Passport1/2020_pair_surveys_UND_SFelege'
assert os.path.isdir(input_image_folder)

debug_max_image = -1

class_mapping_type = 'multiclass' # 'binary' or 'multiclass'
model_tag = 'yolov5'

if debug_max_image < 0:
    output_dir = os.path.expanduser('~/data/und-ducks/und-ducks-{}-{}'.format(class_mapping_type,model_tag))
else:
    output_dir = os.path.expanduser('~/data/und-ducks-mini-{}/und-ducks-{}-{}'.format(
        debug_max_image,class_mapping_type,model_tag))

class_mapped_json = os.path.join(output_dir,'und-ducks-{}.json'.format(class_mapping_type))

# This will contain *all* yolo-formatted patches
yolo_all_dir = os.path.join(output_dir,'yolo_all')
yolo_dataset_file = os.path.join(output_dir,'dataset.yaml')

# This file will be compatible with the MegaDetector repo's inference scripts
md_class_mapping_file = os.path.join(output_dir,'und-ducks-md-class-mapping-{}.json'.format(class_mapping_type))

# Just the train/val subsets
yolo_train_dir = os.path.join(output_dir,'yolo_train')
yolo_val_dir = os.path.join(output_dir,'yolo_val')

train_images_list_file = os.path.join(output_dir,'train_images.json')
val_images_list_file = os.path.join(output_dir,'val_images.json')

patch_size = [1280,1280]
patch_stride = 0.9

# Should we clip boxes to image boundaries, even if we know the object extends
# into an adjacent image?
clip_boxes = True

category_names_to_exclude = []

# We are going to write multiple copies of the class list file, because
# YOLOv5 expects "classes.txt", but BoundingBoxEditor expects "object.data"
class_file_names = ['object.data','classes.txt']

# This will store a mapping from patches back to the original images
patch_metadata_file = 'patch_metadata.json'
patch_metadata_file_positives_only = 'patch_metadata_positives_only.json'

val_image_fraction = 0.15

# Randomly sample a subset of negative patches (separate from sampling hard negatives)
p_use_empty_patch = 0

# If we have N images with annotations, we will choose hard_negative_fraction * N
# hard negatives, and from each of those we'll choose a number of patches equal to the
# average number of patches per image.
#
# YOLOv5 recommends 0%-10% hard negatives.
hard_negative_fraction = 0.2

# The YOLO spec leaves it slightly ambiguous wrt whether empty annotation files are 
# required/banned for hard negatives
write_empty_annotation_files_for_hard_negatives = True

random.seed(0)

patch_jpeg_quality = 95

# When we clip bounding boxes that are right on the edge of an image, clip them back
# by just a little more than we need to, to make BoundingBoxEditor happy
clip_epsilon_pixels = 0.01

do_tile_writes = True
do_image_copies = True

if not do_tile_writes:
    print('*** Warning: tile output disabled ***')

if not do_image_copies:
    print('*** Warning: image copying disabled ***')


#%% Folder prep

assert os.path.isdir(input_image_folder)
assert os.path.isfile(input_annotations_file)

os.makedirs(output_dir,exist_ok=True)
os.makedirs(yolo_all_dir,exist_ok=True)
os.makedirs(yolo_train_dir,exist_ok=True)
os.makedirs(yolo_val_dir,exist_ok=True)

# For a while I was writing images and annotations to different folders, so 
# the code still allows them to be different.
dest_image_folder = yolo_all_dir
dest_txt_folder = yolo_all_dir

# Just in case...
os.makedirs(dest_image_folder,exist_ok=True)
os.makedirs(dest_txt_folder,exist_ok=True)
            

#%% Read source annotations

with open(input_annotations_file,'r') as f:
    d = json.load(f)
    
source_image_id_to_annotations = defaultdict(list)
for ann in d['annotations']:
    source_image_id_to_annotations[ann['image_id']].append(ann)

print('Read {} annotations for {} images'.format(
    len(d['annotations']),len(d['images'])))

# This is a list of relative paths to images with no annotations available; we'll
# sample from those to include some hard negatives
assert 'images_without_annotations' in d
print('Read a list of {} images without annotations'.format(len(d['images_without_annotations'])))

source_category_id_to_name = {c['id']:c['name'] for c in d['categories']}
source_category_name_to_id = {c['name']:c['id'] for c in d['categories']}
source_category_ids_to_exclude = []
for s in category_names_to_exclude:
    if s in source_category_name_to_id:
        source_category_ids_to_exclude.append(source_category_name_to_id[s])
    else:
        print("Warning: I'm supposed to ignore category {}, but this .json file doesn't"
              "have that category".format(s))
    

#%% Re-map categories based on the model type

def invert_dict(d,verify_uniqueness=True):
    
    if verify_uniqueness:
        values = set()
        for k in d.keys():
            v = d[k]
            assert v not in values, 'Attempting to invert a non-invertible dict'
            values.add(v)
            
    return {v: k for k, v in d.items()}
    
assert class_mapping_type in ('binary','multiclass')

multiclass_mapping = {
    'other_songbird':'other_songbird',
    'northern_shoveler':'northern_shoveler',
    'american_coot':'american_coot',
    'mallard':'mallard',
    'blue_winged_teal':'blue_winged_teal',
    'other_waterfowl':'other_waterfowl',
    'pied_billed_grebe':'other_waterfowl',
    'lesser_scaup':'lesser_scaup',
    'redhead':'other_waterfowl',
    'gadwall':'gadwall',
    'northern_pintail':'other_waterfowl',
    'red_winged_blackbird':'other_songbird',
    'canada_goose':'canada_goose',
    'yellow_headed_blackbird':'other_songbird',
    'common_grackle':'other_songbird',
    'canvasback':'other_waterfowl',
    'ruddy_duck':'other_waterfowl',
    'american_wigeon':'other_waterfowl',
    'horned_grebe':'other_waterfowl',
    'green_winged_teal':'other_waterfowl',
    'other_waterbird':'other_waterfowl',
    'unknown':'unknown'
    }

if class_mapping_type == 'binary':
    
    category_id_to_name = {0:'bird'}
    category_name_to_id = invert_dict(category_id_to_name)
    source_category_to_output_category = {}
    for source_id in source_category_id_to_name:
        source_category_to_output_category[source_id] = 0
    
else:
    
    assert class_mapping_type == 'multiclass'    
    category_name_to_id = {}
    source_category_to_output_category = {}
    
    next_category_id = 0
    
    for source_id in source_category_id_to_name:
        source_name = source_category_id_to_name[source_id]
        output_name = multiclass_mapping[source_name]
        if output_name not in category_name_to_id:
            category_name_to_id[output_name] = next_category_id
            next_category_id += 1
        source_category_to_output_category[source_id] = category_name_to_id[output_name]
                    
    category_id_to_name = invert_dict(category_name_to_id)
        
# Re-map annotations

# Re-read so this cell is idempotent
with open(input_annotations_file,'r') as f:
    d = json.load(f)

# ann = d['annotations'][0]
for ann in d['annotations']:
    ann['category_id'] = source_category_to_output_category[ann['category_id']]
    
categories = []
for category_name in category_name_to_id:
    categories.append({'id':category_name_to_id[category_name],'name':category_name})
d['categories'] = categories

# Write the updated .json out
with open(class_mapped_json,'w') as f:
    json.dump(d,f,indent=1)
    
image_id_to_annotations = defaultdict(list)    
for ann in d['annotations']:
    image_id_to_annotations[ann['image_id']].append(ann)

category_ids_to_exclude = []
for source_id in source_category_ids_to_exclude:
    category_ids_to_exclude.append(source_category_to_output_category[source_id])


#%% Verify image ID uniqueness

image_ids = set()
for im in d['images']:
    assert im['id'] not in image_ids
    image_ids.add(im['id'])
    

#%% Support functions

def relative_path_to_image_name(rp):    
    image_name = rp.lower().replace('/','_')
    assert image_name.endswith('.jpg')
    image_name = image_name.replace('.jpg','')
    return image_name
    

#%% Create YOLO-formatted patches

# Takes about 10 minutes
#
# TODO: This is trivially parallelizable

## Create YOLO-formatted patches (prep)

# This will be a dict mapping patch names (YOLO files without the extension)
# to metadata about their sources
patch_metadata = {}

image_ids_with_annotations = []
n_patches = 0
n_boxes = 0
n_clipped_boxes = 0
n_excluded_boxes = 0
    

## Create YOLO-formatted patches (main loop)

random.seed(0)

# i_image = 1; im = d['images'][i_image]
for i_image,im in tqdm(enumerate(d['images']),total=len(d['images'])):

    if debug_max_image >= 0 and i_image > debug_max_image:
        break

    annotations = image_id_to_annotations[im['id']]
    
    # Skip images that have no annotations at all
    if len(annotations) == 0:
        continue
    
    image_fn = os.path.join(input_image_folder,im['file_name'])
    pil_im = visutils.open_image(image_fn)
    assert pil_im.size[0] == im['width']
    assert pil_im.size[1] == im['height']
    
    image_name = relative_path_to_image_name(im['file_name'])
    
    # The loop I'm about to do would be catastrophically inefficient if the numbers of annotations
    # per image were very large, but in practice, this whole script is going to be limited by
    # image I/O, and it's going to be negligible compared to training time anyway, so err'ing on
    # the side of readability, rather than using a more efficient box lookup system.
    
    n_patches_this_image = 0
    
    patch_start_positions = get_patch_boundaries(pil_im.size,patch_size,patch_stride)
    
    # i_patch = 0; patch_xy = patch_start_positions[i_patch]
    for i_patch,patch_xy in enumerate(patch_start_positions):
        
        patch_x_min = patch_xy[0]
        patch_y_min = patch_xy[1]
        patch_x_max = patch_x_min + patch_size[0] - 1
        patch_y_max = patch_y_min + patch_size[1] - 1
    
        # PIL represents coordinates in a way that is very hard for me to get my head
        # around, such that even though the "right" and "bottom" arguments to the crop()
        # function are inclusive... well, they're not really.
        #
        # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#coordinate-system
        #
        # So we add 1 to the max values.
        patch_im = pil_im.crop((patch_x_min,patch_y_min,patch_x_max+1,patch_y_max+1))
        assert patch_im.size[0] == patch_size[0]
        assert patch_im.size[1] == patch_size[1]

        # List of x/y/w/h/category boxes in absolute coordinates in the original image
        patch_boxes = []
        
        # Find all the boxes that have at least N pixel overlap with this patch
        # i_ann = 0; ann = annotations[0]
        for i_ann,ann in enumerate(annotations):
            
            # Does this annotation overlap with this patch?
            
            box = ann['bbox']
            
            # In the input annotations, boxes are x/y/w/h
            box_x_min = box[0]
            box_x_max = box[0] + box[2]
            box_y_min = box[1]
            box_y_max = box[1] + box[3]
            
            patch_contains_box = (patch_x_min < box_x_max and box_x_min < patch_x_max \
                                 and patch_y_min < box_y_max and box_y_min < patch_y_max)
            
            if patch_contains_box:
                
                # Is this a category we're ignoring?
                if ann['category_id'] in category_ids_to_exclude:
                    n_excluded_boxes += 1
                    continue                
                
                n_boxes += 1
                patch_boxes.append([box[0],box[1],box[2],box[3],ann['category_id']])
        
        # ...for each annotation
        
        del ann
        
        # If we have no annotations that overlap this patch...
        if len(patch_boxes) == 0:
            
            # We used to skip all patches with no boxes...
            # continue
        
            # ...but now we can randomly sample a subset of empty patches
            r = random.random()
            assert r >= 0.0 and r <= 1.0
            if r >= p_use_empty_patch:
                continue
        
        n_patches_this_image += 1
        
        patch_name = patch_info_to_patch_name(image_name,patch_x_min,patch_y_min)
        patch_image_fn = os.path.join(dest_image_folder,patch_name + '.jpg')
        patch_ann_fn = os.path.join(dest_txt_folder,patch_name + '.txt')
        
        # Write out patch image
        if (do_tile_writes):
            patch_im.save(patch_image_fn,quality=patch_jpeg_quality)
        
        # Create YOLO annotations
        
        yolo_boxes_this_patch = []
        
        # xywhc = patch_boxes[0]
        for xywhc in patch_boxes:
            
            x_center_absolute_original = xywhc[0] + (xywhc[2] / 2.0)
            y_center_absolute_original = xywhc[1] + (xywhc[3] / 2.0)
            box_w_absolute = xywhc[2]
            box_h_absolute = xywhc[3]
            
            x_center_absolute_patch = x_center_absolute_original - patch_x_min
            y_center_absolute_patch = y_center_absolute_original - patch_y_min
            
            assert (1 + patch_x_max - patch_x_min) == patch_size[0]
            assert (1 + patch_y_max - patch_y_min) == patch_size[1]
            
            x_center_relative = x_center_absolute_patch / patch_size[0]
            y_center_relative = y_center_absolute_patch / patch_size[1]
            
            box_w_relative = box_w_absolute / patch_size[0]
            box_h_relative = box_h_absolute / patch_size[1]
            
            if clip_boxes:
                
                clipped_box = False
                clip_epsilon_relative = clip_epsilon_pixels / patch_size[0]
                
                box_right = x_center_relative + (box_w_relative / 2.0)                    
                if box_right > 1.0:
                    clipped_box = True
                    overhang = box_right - 1.0
                    box_w_relative -= overhang
                    x_center_relative -= ((overhang / 2.0) + clip_epsilon_relative)

                box_bottom = y_center_relative + (box_h_relative / 2.0)                                        
                if box_bottom > 1.0:
                    clipped_box = True
                    overhang = box_bottom - 1.0
                    box_h_relative -= overhang
                    y_center_relative -= ((overhang / 2.0) + clip_epsilon_relative)
                
                box_left = x_center_relative - (box_w_relative / 2.0)
                if box_left < 0.0:
                    clipped_box = True
                    overhang = abs(box_left)
                    box_w_relative -= overhang
                    x_center_relative += ((overhang / 2.0) + clip_epsilon_relative)
                    
                box_top = y_center_relative - (box_h_relative / 2.0)
                if box_top < 0.0:
                    clipped_box = True
                    overhang = abs(box_top)
                    box_h_relative -= overhang
                    y_center_relative += ((overhang / 2.0) + clip_epsilon_relative)
                    
                if clipped_box:
                    n_clipped_boxes += 1
            
            # ...if we're clipping boxes
            
            # YOLO annotations are category x_center y_center w h
            yolo_box = [xywhc[4],
                        x_center_relative, y_center_relative, 
                        box_w_relative, box_h_relative]
            yolo_boxes_this_patch.append(yolo_box)
            
        # ...for each box
        
        this_patch_metadata = {
            'patch_name':patch_name,
            'original_image_id':im['id'],
            'patch_x_min':patch_x_min,
            'patch_y_min':patch_y_min,
            'patch_x_max':patch_x_max,
            'patch_y_max':patch_y_max,
            'hard_negative':False,
            'patch_boxes':patch_boxes
        }
                    
        patch_metadata[patch_name] = this_patch_metadata
        
        with open(patch_ann_fn,'w') as f:
            for yolo_box in yolo_boxes_this_patch:
                f.write('{} {} {} {} {}\n'.format(
                    yolo_box[0],yolo_box[1],yolo_box[2],yolo_box[3],yolo_box[4]))            
        
    # ...for each patch
    
    n_patches += n_patches_this_image
    
    if n_patches_this_image > 0:
        image_ids_with_annotations.append(im['id'])
    
# ...for each image

# We should only have processed each image once
assert len(image_ids_with_annotations) == len(set(image_ids_with_annotations))

n_images_with_annotations = len(image_ids_with_annotations)

print('\nProcessed {} boxes ({} clipped) ({} excluded) from {} patches on {} images'.format(
    n_boxes,n_clipped_boxes,n_excluded_boxes,n_patches,n_images_with_annotations))

annotated_image_ids = set([p['original_image_id'] for p in patch_metadata.values()])
assert len(annotated_image_ids) == n_images_with_annotations

print('Wrote YOLO-formatted positive images to {}'.format(dest_image_folder))
print('Wrote YOLO-formatted positive annotations to {}'.format(dest_txt_folder))

del pil_im


#%% Write out patch metadata file before sampling hard negatives

# This is not used later, it's only for debugging

patch_metadata_file_full_path_positives_only = \
    os.path.join(yolo_all_dir,patch_metadata_file_positives_only)
with open(patch_metadata_file_full_path_positives_only,'w') as f:
    json.dump(patch_metadata,f,indent=1)

print('Wrote patch metadata (positives only) to {}'.format(
    patch_metadata_file_full_path_positives_only))

# I really really really want to make sure I do a clean load in the next cell
del patch_metadata


#%% Sample hard negatives

random.seed(0)

# This cell appends to patch_metadata_mapping, so it's not idempotent; force a clean
# load before mucking with patch_metadata_mapping.
with open(patch_metadata_file_full_path_positives_only,'r') as f:
    patch_metadata = json.load(f)

n_hard_negative_source_images = round(hard_negative_fraction*n_images_with_annotations)
average_patches_per_image = n_patches / n_images_with_annotations

candidate_hard_negatives = d['images_without_annotations']
n_bypassed_candidates = 0
        
hard_negative_source_images = random.sample(candidate_hard_negatives,
                                             k=n_hard_negative_source_images)

n_patches_before_hard_negatives = len(patch_metadata)

assert len(hard_negative_source_images) == len(set(hard_negative_source_images))

# For each hard negative source image
#
# image_fn_relative = hard_negative_source_images[0]
for image_fn_relative in tqdm(hard_negative_source_images):
    
    image_fn_abs = os.path.join(input_image_folder,image_fn_relative)
    pil_im = visutils.open_image(image_fn_abs)
        
    image_id = image_fn_relative.replace('/','_')
    assert image_id not in annotated_image_ids
    
    # Sample random patches from this image
    sampled_patch_start_positions = random.sample(patch_start_positions,
                                                   k=round(average_patches_per_image))
    
    # For each sampled patch
    # i_patch = 0; patch_xy = sampled_patch_start_positions[i_patch]
    for i_patch,patch_xy in enumerate(sampled_patch_start_positions):
        
        patch_x_min = patch_xy[0]
        patch_y_min = patch_xy[1]
        patch_x_max = patch_x_min + patch_size[0] - 1
        patch_y_max = patch_y_min + patch_size[1] - 1
    
        patch_im = pil_im.crop((patch_x_min,patch_y_min,patch_x_max+1,patch_y_max+1))
        assert patch_im.size[0] == patch_size[0]
        assert patch_im.size[1] == patch_size[1]
        
        image_name = relative_path_to_image_name(image_fn_relative)
        
        patch_name = image_name + '_' + str(patch_x_min).zfill(4) + '_' + str(patch_y_min).zfill(4)
        patch_image_fn = os.path.join(dest_image_folder,patch_name + '.jpg')
        patch_ann_fn = os.path.join(dest_txt_folder,patch_name + '.txt')
        
        assert not os.path.isfile(patch_image_fn)
        assert not os.path.isfile(patch_ann_fn)
        
        # Write out patch image
        patch_im.save(patch_image_fn,quality=patch_jpeg_quality)
        
        # Write empty annotation file
        if write_empty_annotation_files_for_hard_negatives:
            with open(patch_ann_fn,'w') as f:
                pass
            assert os.path.isfile(patch_ann_fn)
        
        # Add to patch metadata list
        patch_name = patch_info_to_patch_name(image_name,patch_x_min,patch_y_min)
        assert patch_name not in patch_metadata
        this_patch_metadata = {
            'patch_name':patch_name,
            'original_image_id':image_id,
            'patch_x_min':patch_x_min,
            'patch_y_min':patch_y_min,
            'patch_x_max':patch_x_max,
            'patch_y_max':patch_y_max,
            'hard_negative':True,
            'patch_boxes':[]
            }
        patch_metadata[patch_name] = this_patch_metadata
    
    # ...for each patch

# ...for each hard negative image    

n_patches_after_hard_negatives = len(patch_metadata)

print('\nTraining with {} patches ({} before hard negatives)'.format(
    n_patches_after_hard_negatives,
    n_patches_before_hard_negatives))


#%% Do some consistency checking on our patch metadata

n_patches_from_negative_images = 0
n_patches_from_positive_images = 0

n_positive_patches = 0
n_negative_patches = 0

n_negative_patches_from_positive_images = 0

positive_image_ids = set()

for patch_name in patch_metadata.keys():
    
    v = patch_metadata[patch_name]
    
    patch_image_fn = os.path.join(dest_image_folder,patch_name + '.jpg')
    patch_ann_fn = os.path.join(dest_txt_folder,patch_name + '.txt')
    
    assert os.path.isfile(patch_image_fn)
    assert os.path.isfile(patch_ann_fn)
    
    with open(patch_ann_fn,'r') as f:
        patch_ann_lines = f.readlines()
    
    for s in patch_ann_lines:
        assert len(s.strip().split(' ')) == 5
    n_annotations_this_patch = len(patch_ann_lines)        
    assert n_annotations_this_patch == len(v['patch_boxes'])
        
    if n_annotations_this_patch > 0:
        n_positive_patches += 1
        positive_image_ids.add(v['original_image_id'])
    else:
        n_negative_patches += 1
        
    assert v['patch_name'] == patch_name
    if v['hard_negative']:
        n_patches_from_negative_images += 1
        assert n_annotations_this_patch == 0        
    else:
        n_patches_from_positive_images += 1
        if n_annotations_this_patch == 0:
            n_negative_patches_from_positive_images += 1

# ...for each patch

print('Found {} patches from {} positive images'.format(
    n_patches_from_positive_images,len(positive_image_ids)))
print('Found {} positive patches'.format(n_positive_patches))

print('Found {} negative patches total'.format(n_negative_patches))
print('Found {} patches from hard negative images'.format(n_patches_from_negative_images))
print('Found {} negative patches from positive images'.format(n_negative_patches_from_positive_images))

if p_use_empty_patch <= 0:
    assert n_patches_from_positive_images == n_positive_patches
    assert n_patches_from_negative_images == n_negative_patches


#%% Write out patch metadata file (for real this time)

patch_metadata_file_full_path = os.path.join(yolo_all_dir,patch_metadata_file)
with open(patch_metadata_file_full_path,'w') as f:
    json.dump(patch_metadata,f,indent=1)
    
print('Wrote patch metadata to {}'.format(patch_metadata_file_full_path))    


#%% Write out class list and category mapping

# Write the YOLO-formatted class file
#
# ...possibly more than once, to satisfy a couple of conventions on 
# what it's supposed to be called.
for class_file_name in class_file_names:
    class_file_full_path = os.path.join(yolo_all_dir,class_file_name)
    with open(class_file_full_path, 'w') as f:
        print('Writing class list to {}'.format(class_file_full_path))
        for i_class in range(0,len(category_id_to_name)):
            # Category IDs should range from 0..N-1
            assert i_class in category_id_to_name
            f.write(category_id_to_name[i_class] + '\n')

md_category_mapping = {}
for category_id in category_id_to_name:
    md_category_mapping[str(category_id)] = category_id_to_name[category_id]
with open(md_class_mapping_file,'w') as f:
    json.dump(md_category_mapping,f,indent=1)

print('Wrote MD class mapping to {}'.format(md_class_mapping_file))


#%% Force a clean load

del patch_metadata
del annotated_image_ids
del image_id_to_annotations


#%% Load patch metadata

patch_metadata_file_full_path = os.path.join(yolo_all_dir,patch_metadata_file)
with open(patch_metadata_file_full_path,'r') as f:
    patch_metadata = json.load(f)


#%% Look at the number of images each category appears in

category_id_to_images = defaultdict(set)

for patch_name in patch_metadata.keys():
    
    patch_info = patch_metadata[patch_name]
    if patch_info['hard_negative']:
        continue    
    for patch_box in patch_info['patch_boxes']:
        category_id = patch_box[4]
        category_id_to_images[category_id].add(patch_info['original_image_id'])
        
    # ...for each box
    
# ...for each patch    

for category_id in category_id_to_name:
    print('{} ({}): {}'.format(
        category_id, category_id_to_name[category_id],len(category_id_to_images[category_id])))


#%% Split image IDs into train/val (iterate over random seeds)

import numpy as np

all_image_ids = set()
image_id_to_categories = defaultdict(set)
for patch_name in patch_metadata.keys():
    patch_info = patch_metadata[patch_name]
    image_id = patch_info['original_image_id']
    all_image_ids.add(image_id)
    for box in patch_info['patch_boxes']:
        category_id = box[-1]
        image_id_to_categories[image_id].add(category_id)
    
print('Found {} unique image IDs for {} patches'.format(
    len(all_image_ids),len(patch_metadata)))

all_image_ids = list(all_image_ids)
n_val_image_ids = int(val_image_fraction*len(all_image_ids))

category_to_image_id = defaultdict(set)
for image_id in image_id_to_categories:
    for category_id in image_id_to_categories[image_id]:
        category_to_image_id[category_id].add(image_id)

max_random_seed = 1000

random_seed_to_worst_error = {}
random_seed_to_average_error = {}

# random_seed = 0
for random_seed in range(0,max_random_seed):
    
    # Randomly split into train/val
    random.seed(random_seed)
    val_image_ids = random.sample(all_image_ids,k=n_val_image_ids)
    val_image_ids_set = set(val_image_ids)
    
    # For each category, measure the % of images that went into the val set
    category_to_val_fraction = defaultdict(float)
    
    for category_id in category_to_image_id:
        category_images = category_to_image_id[category_id]
        n_category_images = len(category_images)
        n_category_val_images = 0
        for image_id in category_images:
            if image_id in val_image_ids_set:
                n_category_val_images += 1
        category_val_fraction = n_category_val_images / n_category_images
        category_to_val_fraction[category_id] = category_val_fraction
            
    # What's the furthest any category is from the target val fraction?
    category_errors = []
    for category_val_fraction in category_to_val_fraction.values():
        category_error = abs(category_val_fraction-val_image_fraction)
        category_errors.append(category_error)
    
    worst_error = max(category_errors)
    average_error = np.mean(category_errors)
        
    random_seed_to_worst_error[random_seed] = worst_error
    random_seed_to_average_error[random_seed] = average_error

random_seed_to_error_metric = {}
for r in random_seed_to_worst_error.keys():
    random_seed_to_error_metric[r] = \
        random_seed_to_worst_error[r] * random_seed_to_average_error[r]

min_error = None
min_error_seed = None

for r in random_seed_to_error_metric.keys():
    error_metric = random_seed_to_error_metric[r]
    if min_error is None or error_metric < min_error:
        min_error = error_metric
        min_error_seed = r
        

#%% Split image IDs into train/val (single random seed)

# We used 0 for the binary mapping in the general sense of defaultness, and 
# hard-coded the random seed for the multi-class mapping to make sure every class
# appears in both train and val.
class_mapping_type_to_random_seed = {'binary':0,'multiclass':min_error_seed}
random.seed(class_mapping_type_to_random_seed[class_mapping_type])

all_image_ids = set()
for patch_name in patch_metadata.keys():
    patch_info = patch_metadata[patch_name]
    all_image_ids.add(patch_info['original_image_id'])
print('Found {} unique image IDs for {} patches'.format(
    len(all_image_ids),len(patch_metadata)))

all_image_ids = list(all_image_ids)

n_val_image_ids = int(val_image_fraction*len(all_image_ids))

val_image_ids = random.sample(all_image_ids,k=n_val_image_ids)
val_image_ids_set = set(val_image_ids)

train_image_ids = []
for image_id in all_image_ids:
    if image_id not in val_image_ids_set:
        train_image_ids.append(image_id)


#%% Look at the class balance

category_id_to_n_boxes_train = defaultdict(int)
category_id_to_n_boxes_val = defaultdict(int)
category_id_to_n_boxes_all = defaultdict(int)

# patch_name = next(iter(patch_metadata.keys()))
for patch_name in patch_metadata.keys():
    patch_info = patch_metadata[patch_name]
    image_id = patch_info['original_image_id']
    b_val_image = (image_id in val_image_ids_set)
    for xywhc in patch_info['patch_boxes']:
        assert len(xywhc) == 5
        category_id = xywhc[4]
        category_id_to_n_boxes_all[category_id] = category_id_to_n_boxes_all[category_id] + 1
        assert category_id in category_id_to_name
        if b_val_image:
            category_id_to_n_boxes_val[category_id] = category_id_to_n_boxes_val[category_id] + 1
        else:
            category_id_to_n_boxes_train[category_id] = category_id_to_n_boxes_train[category_id] + 1

print('Training data:\n')
for category_id in category_id_to_n_boxes_all:
    category_name = category_id_to_name[category_id]
    n_boxes_train = category_id_to_n_boxes_train[category_id]
    n_boxes_val = category_id_to_n_boxes_val[category_id]
    print('Category {} ({}): {} train, {} val'.format(
        category_id,category_name,n_boxes_train,n_boxes_val))

for category_id in category_id_to_name:
    assert category_id_to_n_boxes_val[category_id] > 0
    assert category_id_to_n_boxes_train[category_id] > 0    


#%% Copy images to train/val folders

train_patch_names = []
val_patch_names = []

# For each patch
for patch_name in tqdm(patch_metadata.keys(),total=len(patch_metadata)):
    
    patch_info = patch_metadata[patch_name]
    
    # Make sure we have annotation/image files for this patch
    source_image_path = os.path.join(yolo_all_dir,patch_name + '.jpg')
    source_ann_path = os.path.join(yolo_all_dir,patch_name + '.txt')
    
    assert os.path.isfile(source_image_path)
    assert os.path.isfile(source_ann_path)
    
    # Copy to the place it belongs
    if patch_info['original_image_id'] in val_image_ids:
        val_patch_names.append(patch_name)
        target_folder = yolo_val_dir
    else:
        train_patch_names.append(patch_name)
        target_folder = yolo_train_dir
    
    target_image_path = os.path.join(target_folder,os.path.basename(source_image_path))
    target_ann_path = os.path.join(target_folder,os.path.basename(source_ann_path))
    
    if do_image_copies:
        shutil.copyfile(source_image_path,target_image_path)
        shutil.copyfile(source_ann_path,target_ann_path)
    
# ...for each patch        

print('\nCopied {} train patches, {} val patches'.format(
    len(train_patch_names),len(val_patch_names)))


#%% Save train/val splits

with open(train_images_list_file,'w') as f:
    json.dump(train_image_ids,f,indent=1)

with open(val_images_list_file,'w') as f:
    json.dump(val_image_ids,f,indent=1)          
          

#%% Generate the YOLO training dataset file

# Read class names
class_file_path = os.path.join(yolo_all_dir,'classes.txt')
with open(class_file_path,'r') as f:
    class_lines = f.readlines()
class_lines = [s.strip() for s in class_lines]    
class_lines = [s for s in class_lines if len(s) > 0]

# Write dataset.yaml
with open(yolo_dataset_file,'w') as f:
    
    yolo_train_folder_relative = os.path.relpath(yolo_train_dir,output_dir)
    yolo_val_folder_relative = os.path.relpath(yolo_val_dir,output_dir)
    
    f.write('# Train/val sets\n')
    f.write('path: {}\n'.format(output_dir))
    f.write('train: {}\n'.format(yolo_train_folder_relative))
    f.write('val: {}\n'.format(yolo_val_folder_relative))
    
    f.write('\n')
    
    f.write('# Classes\n')
    f.write('names:\n')
    for i_class,class_name in enumerate(class_lines):
        f.write('  {}: {}\n'.format(i_class,class_name))


#%% Prepare simlinks for BoundingBoxEditor

# ...so it can appear that images and labels are in separate folders

def create_virtual_yolo_dirs(yolo_base_dir):
    
    images_dir = os.path.join(yolo_base_dir,'images')
    labels_dir = os.path.join(yolo_base_dir,'labels')
    
    os.makedirs(images_dir,exist_ok=True)
    os.makedirs(labels_dir,exist_ok=True)
    
    files = os.listdir(os.path.join(yolo_all_dir))
    source_images = [fn for fn in files if fn.lower().endswith('.jpg')]
    source_labels = [fn for fn in files if fn.lower().endswith('.txt') and fn != 'classes.txt']
    
    # fn = source_images[0]
    for fn in source_images:
        link_exists = os.path.join(yolo_all_dir,fn)
        link_new = os.path.join(images_dir,fn)
        safe_create_link(link_exists,link_new)
    for fn in source_labels:
        link_exists = os.path.join(yolo_all_dir,fn)
        link_new = os.path.join(labels_dir,fn)
        safe_create_link(link_exists,link_new)
    
    link_exists = os.path.join(yolo_all_dir,'object.data')        
    link_new = os.path.join(labels_dir,'object.data')
    safe_create_link(link_exists,link_new)

# Only do this for the "all" dir!  Doing this in the training and val dirs 
# impacts training.

create_virtual_yolo_dirs(yolo_all_dir)
# create_virtual_yolo_dirs(yolo_train_dir)
# create_virtual_yolo_dirs(yolo_val_dir)


#%% Generate a patch-level COCO .json file

from data_management import yolo_to_coco

coco_patch_file = os.path.join(output_dir,'und-ducks-patches.json')

_ = yolo_to_coco.yolo_to_coco(input_folder=yolo_all_dir,
             class_name_file=class_file_path,
             output_file=coco_patch_file)
             

#%% Preview patches

from md_visualization import visualize_db

options = visualize_db.DbVizOptions()

options.num_to_visualize = None

# Target size for rendering; set either dimension to -1 to preserve aspect ratio
options.viz_size = (700, -1)
options.include_filename_links = True

preview_folder = os.path.join(output_dir,'patch_preview')

_ = visualize_db.process_images(db_path=coco_patch_file,
                            output_dir=preview_folder,
                            image_base_dir=yolo_all_dir,
                            options=options)

from md_utils.path_utils import open_file
open_file(preview_folder + '/index.html')
