########
#
# usgs-geese-prepare-lila-dataset.py
#
# Prepare a public (re-)release of the images and annotation used in training.
#
########

#%% Imports and constants

import os
import json
import copy
import shutil
import random

import humanfriendly

from tqdm import tqdm
from collections import defaultdict

from md_visualization import visualization_utils as vis_utils

# input_file = os.path.expanduser('~/data/und-ducks/und-ducks-multiclass-yolov5/und-ducks-multiclass.json')
input_file = os.path.expanduser('~/data/und-ducks.json')
output_dir = os.path.expanduser('~/data/und-ducks-lila')
output_file = os.path.join(output_dir,'und-ducks-lila.json')
output_dir_images = os.path.join(output_dir,'images')

drive_root = '/media/user/My Passport1'
image_base = os.path.join(drive_root,'2020_pair_surveys_UND_SFelege')

os.makedirs(output_dir_images,exist_ok=True)
assert os.path.isfile(input_file)
assert os.path.isdir(image_base)

add_image_sizes = True


#%% Read input data

with open(input_file,'r') as f:
    d = json.load(f)

image_id_to_annotations = defaultdict(list)
image_id_to_image = {}

for im in d['images']:
    image_id_to_image[im['id']] = im
    
for ann in d['annotations']:
    image_id_to_annotations[ann['image_id']].append(ann)


#%% Find empty images with annotation files

# In practice, this is nine images

images_with_annotation_files_but_no_annotations = []

for im in d['images']:
    if im['id'] not in image_id_to_annotations:
        images_with_annotation_files_but_no_annotations.append(im['id'])

print('Found {} images with annotation files but no annotations'.format(
    len(images_with_annotation_files_but_no_annotations)))


#%% Validate input data, compute total size, and add an image-level empty category

##%% First the positives

total_size_bytes_positives = 0

# im = d['images'][0]
for im in tqdm(d['images']):
    image_id = im['id']   
    if image_id not in image_id_to_annotations:
        continue
    input_fn_abs = os.path.join(image_base,im['file_name'])
    assert os.path.isfile(input_fn_abs)    
    total_size_bytes_positives += os.path.getsize(input_fn_abs)
# ...for each non-empty image


##%% Now the negatives

max_non_empty_id = max([cat['id'] for cat in d['categories']])
empty_id = max_non_empty_id + 1

empty_image_filenames = []
empty_annotations = []
empty_images_with_no_annotation_files = []
empty_images_with_annotation_files = []

total_size_bytes_negatives = 0

# Grab the images that had empty annotation files
for im in tqdm(d['images']):
    image_id = im['id']   
    if image_id not in image_id_to_annotations:
        empty_image_filenames.append(image_id)
del im

# At this point, we should only have the empty images that had annotation files, but no annotations
assert all([fn in image_id_to_image for fn in empty_image_filenames])
assert len(empty_image_filenames) == len(images_with_annotation_files_but_no_annotations)

empty_image_filenames.extend(d['images_without_annotations'])

# i_empty_image = 0; image_fn_relative = empty_images[0]
for i_empty_image,image_fn_relative in tqdm(enumerate(
        empty_image_filenames),total=len(empty_image_filenames)):
    
    # E.g.:
    # '20200425_cot_18w051_pa_02_45m_x5s/DJI_0085.JPG'

    image_id = image_fn_relative
    image_fn_abs = os.path.join(image_base,image_fn_relative)    
    
    # Empty images with empty annotation files already have IDs; create IDs and structs
    # for new images.    
    if image_id not in image_id_to_image:
                
        im = {}
        im['file_name'] = image_fn_relative
        im['id'] = im['file_name']
        im['annotation_file'] = None    
        
        # Don't add to image_id_to_image; I want that to reflect just images
        # with annotation files.
        # image_id_to_image[im['id']] = im
        
        if add_image_sizes:
            pil_im = vis_utils.open_image(image_fn_abs)
            image_w = pil_im.size[0]
            image_h = pil_im.size[1]
            im['width'] = image_w
            im['height'] = image_h
            
        empty_images_with_no_annotation_files.append(im)
    
    else:
        
        im = image_id_to_image[image_id]
        empty_images_with_annotation_files.append(im)
        
    ann = {}
    ann['image_id'] = im['id']
    ann['id'] = im['id']
    ann['category_id'] = empty_id
    
    empty_annotations.append(ann)
    image_id_to_annotations[im['id']] = [ann]
    
    assert os.path.isfile(image_fn_abs)    
    total_size_bytes_negatives += os.path.getsize(image_fn_abs)
    
# ...for each empty image (with or without annotation files)  
    
total_size_bytes = total_size_bytes_positives + total_size_bytes_negatives

print('\nValidated {} files totaling {} ({} positive, {} negative)'.format(
    len(d['images']),
    humanfriendly.format_size(total_size_bytes),
    humanfriendly.format_size(total_size_bytes_positives),
    humanfriendly.format_size(total_size_bytes_negatives)))

print('Added annotations for {} images with no annotation files'.format(
    len(empty_images_with_no_annotation_files)))

assert len(empty_images_with_no_annotation_files) == len(d['images_without_annotations'])

assert len(empty_annotations) == \
    len(empty_images_with_no_annotation_files) + \
    len(empty_images_with_annotation_files)


#%% Sample empty images

# The whole dataset is 250GB, which is unnecessary.

random.seed(0)

# Keep a number of empty images equal to the number of positive images
n_empty_images_to_keep = len(d['images'])
sampled_empty_images_with_no_annotation_files = random.sample(empty_images_with_no_annotation_files,
                                                              k=n_empty_images_to_keep)
sampled_empty_image_ids = set([im['id'] for im in sampled_empty_images_with_no_annotation_files])

# I'm not super-confident in control/wetland images that *don't* have annotation files,
# don't include these.  Some of these were actually annotated, but the non-annotated images
# in these folders are less reliable.
sampled_empty_image_ids = [image_id for image_id in sampled_empty_image_ids if \
                           ('control' not in image_id.lower() or 'wetland' not in image_id.lower())]
sampled_empty_image_ids_set = set(sampled_empty_image_ids)
sampled_empty_images_with_no_annotation_files = [im for im in sampled_empty_images_with_no_annotation_files \
                                                 if im['id'] in sampled_empty_image_ids_set]

empty_images_to_keep = sampled_empty_images_with_no_annotation_files
assert len(empty_images_to_keep) == len(sampled_empty_image_ids)
        
print('Kept {} of {} sampled empty images after filtering'.format(
    len(sampled_empty_image_ids),n_empty_images_to_keep))
    
empty_annotations_to_keep = []
for ann in empty_annotations:
    if ann['image_id'] in sampled_empty_image_ids:
        empty_annotations_to_keep.append(ann)

print('Keeping {} of {} empty images'.format(
    len(sampled_empty_images_with_no_annotation_files),
    len(empty_images_with_no_annotation_files) + \
    len(empty_images_with_annotation_files)))

print('Keeping {} of {} empty annotations'.format(
    len(empty_annotations_to_keep),
    len(empty_annotations)))


#%% Merge the empty/non-empty structs

empty_category = {'id':empty_id,'name':'Empty'}

output_d = copy.deepcopy(d)
output_d['categories'].append(empty_category)
output_d['images'].extend(empty_images_to_keep)
output_d['annotations'].extend(empty_annotations_to_keep)

# Make sure image IDs are unique
image_id_to_image = {}
for im in output_d['images']:
    assert im['id'] not in image_id_to_image
    image_id_to_image[im['id']] = im
    
assert len(image_id_to_image) == len(output_d['images'])

for ann in output_d['annotations']:
    if ann['category_id'] == empty_id:
        assert 'bbox' not in ann
    else:
        assert 'bbox' in ann


#%% Copy images

# Takes ~5 minutes

os.makedirs(output_dir_images,exist_ok=True)

ids_copied = set()

# im = output_d['images'][0]
for im in tqdm(output_d['images']):
    
    input_fn_abs = os.path.join(image_base,im['file_name'])
    image_id = im['id']
    assert im['id'] == im['file_name'].replace('images/','')
    assert image_id.endswith('.JPG')
    assert image_id not in ids_copied
    ids_copied.add(image_id)    
    output_fn_abs = os.path.join(output_dir_images,image_id)
    if not os.path.isfile(output_fn_abs):
        os.makedirs(os.path.dirname(output_fn_abs),exist_ok=True)
        shutil.copyfile(input_fn_abs,output_fn_abs)


#%% Convert the .json representation to point to the output data

output_image_base = output_dir

# im = output_d['images'][0]
for im in output_d['images']:
    output_fn_relative = 'images/' + im['id']
    output_fn_abs = os.path.join(output_image_base,output_fn_relative)
    assert os.path.isfile(output_fn_abs)
    
    # This check is only here so we can run this cell twice without
    # prefixing the path name twice
    if im['file_name'] != output_fn_relative:
        im['file_name'] = output_fn_relative
    

#%% Write the output file

with open(output_file,'w') as f:
    json.dump(output_d,f,indent=1)
    

#%% Check DB integrity

from data_management.databases import integrity_check_json_db

options = integrity_check_json_db.IntegrityCheckOptions()
options.baseDir = output_dir
options.bCheckImageSizes = True
options.bCheckImageExistence = True
options.bFindUnusedImages = True
options.bRequireLocation = False

sorted_categories, _, _= integrity_check_json_db.integrity_check_json_db(output_file, options)


#%%

# DB contains 3824 images, 5279 annotations, 3367 bboxes, 22 categories, no sequence info

s = """
  1912 Empty
   876 other_waterfowl
   622 blue_winged_teal
   561 unknown
   502 northern_shoveler
   411 american_coot
   101 mallard
    73 gadwall
    33 lesser_scaup
    31 red_winged_blackbird
    31 yellow_headed_blackbird
    25 other_songbird
    21 northern_pintail
    14 common_grackle
    13 redhead
    12 ruddy_duck
    12 american_wigeon
    11 canada_goose
    10 pied_billed_grebe
     3 canvasback
     3 green_winged_teal
     2 horned_grebe
"""

lines = s.split('\n')
lines = [s.strip() for s in lines]
lines = [s for s in lines if len(s) > 0]

for s in lines:
    tokens = s.split()
    assert len(tokens) == 2
    box_count = tokens[0]
    class_name = tokens[1].replace('_',' ')
    print('<li>{} ({} boxes)</li>'.format(class_name,box_count))

#%% Preview some images

from md_visualization import visualize_db
from md_utils.path_utils import open_file

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 20
viz_options.trim_to_images_with_bboxes = True
viz_options.add_search_links = False
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
viz_options.include_filename_links = True

html_output_file, _ = visualize_db.process_images(db_path=output_file,
                                                    output_dir=os.path.join(output_dir,'preview'),
                                                    image_base_dir=output_dir,
                                                    options=viz_options)
open_file(html_output_file)



#%% Debug scrap

if False:
    
    pass

    #%%
    
    images_without_annotations_set = set(d['images_without_annotations'])
    assert len(images_without_annotations_set) == len(d['images_without_annotations'])
    for empty_image in tqdm(empty_images_with_no_annotation_files):
        empty_image_id = empty_image['id']
        if empty_image_id not in images_without_annotations_set:
            print(empty_image_id)
            

    #%%
    
    """
    Size mismatch for image 20200429_cot_15w199_pa_02_45m_x5s/DJI_0083.JPG: (reported 3264,2448, actual 5280,2970)    
    Size mismatch for image 20200427_cot_16w293_pa_02_45m_x5s/treatment_wetland/dslr_w293/DSC_3751.JPG (reported 5568,3128, actual 5568,3712)
    """
    target_fn = '20200429_cot_15w199_pa_02_45m_x5s/DJI_0083.JPG'