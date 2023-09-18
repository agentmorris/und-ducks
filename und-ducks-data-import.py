########
#
# usgs-geese-data-import.py
#
# * Checks internal consistency of the UND duck data data
#
# * Convert original annotations from Pascal VOC format to COCO Camera Traps
#
# The COCO Camera Traps file produced at the end of this script will include all images
# that had an annotation file.  Some of the images with annotation files were empty, and
# therefore have no annotations in the .json file.  As far as we know, *all* of the images
# without annotation files are empty, so they don't have proper representation in the .json
# file, but it will include a field called "images_without_annotations", which is just 
# what it sounds like.
#
########

#%% Imports and constants

import os
import json

from tqdm import tqdm
from collections import defaultdict

import xmltodict

from md_utils import path_utils
from md_visualization import visualization_utils as visutils

base_folder = '/media/user/My Passport1/2020_pair_surveys_UND_SFelege'
assert os.path.isdir(base_folder)

annotation_path_root = '2020_pair_surveys'

output_json_file = os.path.expanduser('~/data/und-ducks.json')

image_preview_folder = os.path.expanduser('~/tmp/und-ducks/preview')
os.makedirs(image_preview_folder,exist_ok=True)

from ct_utils import get_iou

# IoU threshold for merging similar boxes when there are multiple annotation files
# for an image
iou_merge_threshold = 0.5


#%% Enumerate files

all_files = path_utils.recursive_file_list(base_folder)
all_files = [os.path.relpath(fn,base_folder) for fn in all_files]

# Ignoring the 'control_wetland' and 'treatment_wetland' folders
#
# Actually, some of the annotations *not* in these folders refer to files that *are* 
# in these folders, so keeping them around for now
# all_files = [fn for fn in all_files if ('control' not in fn and 'treatment' not in fn)]

image_files = [fn for fn in all_files if path_utils.is_image_file(fn)]
for fn in image_files:
    assert fn.endswith('.JPG')
    
json_files = [fn for fn in all_files if fn.endswith('.json')]
xml_files = [fn for fn in all_files if fn.endswith('.xml')]

print('Enumerated:')
print('{} total files'.format(len(all_files)))
print('{} image files'.format(len(image_files)))
print('{} json files'.format(len(json_files)))
print('{} xml files'.format(len(xml_files)))

image_files_set = set(image_files)
xml_files_set = set(xml_files)
json_files_set = set(json_files)


#%% Examine annotation files

json_files_without_xml_files = []

for i_file,json_file in enumerate(json_files):
    xml_copy = json_file.replace('.json','.xml')
    if xml_copy not in xml_files_set:
        json_files_without_xml_files.append(json_file)
        
print('Found {} .json files without xml files:'.format(len(json_files_without_xml_files)))

# This turns out to be just one file, not worth worrying about
for s in json_files_without_xml_files:
    print(s)
    
if False:
    missing_json_dir = os.path.dirname(os.path.join(base_folder,json_files_without_xml_files[-1]))    
    path_utils.open_file(missing_json_dir)
    

#%% Read annotation files

relative_filename_to_annotation_dict = {}

# xml_file = xml_files[0]
for xml_file in tqdm(xml_files):
    xml_file_abs = os.path.join(base_folder,xml_file)
    with open(xml_file_abs,'r') as f:
        file_data = f.read()
    annotation_dict = xmltodict.parse(file_data)
        
    assert (len(annotation_dict) == 1) and ('annotation' in annotation_dict.keys())
    
    relative_filename_to_annotation_dict[xml_file] = annotation_dict

class_name_to_count = defaultdict(int)

image_to_annotations = {}

# This turns out to be only nine files
xml_files_with_empty_annotation_lists = []

images_with_multiple_xml_files = {}

all_annotations = []

# xml_file = xml_files[-1]
for i_xml_file,xml_file in tqdm(enumerate(xml_files),total=len(xml_files)):
    
    annotation_dict = relative_filename_to_annotation_dict[xml_file]    
    annotation = annotation_dict['annotation']
    annotation['xml_file'] = xml_file
    assert annotation['size']['depth'] == '3'
    
    # E.g.:
    #
    # 'E:/2020_pair_surveys/20200523_cot_17w256_pa_01_45m_x5s_JI/DJI_0115.JPG'
    # 'E:/2020_pair_surveys/20200523_cot_17w256_pa_01_45m_x5s_JI/DJI_0115.JPG'
    image_path = annotation['path'].replace('\\','/')
    assert annotation_path_root in image_path
    
    # E.g.:
    #
    # '20200523_cot_17w256_pa_01_45m_x5s_JI/DJI_0115.JPG'
    # '20200424_cot_05w197_pa_02_45m_x5s/100MEDIA/DJI_0045.JPG'
    image_path_relative = image_path.split(annotation_path_root)[-1][1:]
    
    assert image_path_relative.split('/')[-2] == annotation['folder']    
    assert image_path_relative in image_files_set
        
    # Do some normalization
    if 'object' not in annotation:
        
        xml_files_with_empty_annotation_lists.append(xml_file)
        annotation['object'] = []
        
    elif isinstance(annotation['object'],list):
        
        assert len(annotation['object']) > 0
        
    else:
        
        assert isinstance(annotation['object'],dict)
        annotation['object'] = [annotation['object']]
    
    for ann in annotation['object']:
        
        assert ann['truncated'] in ('0','1')
        assert ann['difficult'] == '0'
        assert ann['pose'].lower() == 'unspecified'
        class_name_to_count[ann['name']] = class_name_to_count[ann['name']] + 1
        
    assert isinstance(annotation['object'],list)

    all_annotations.extend(annotation['object'])
    
    # If this image is annotated twice...
    if image_path_relative in image_to_annotations:
        
        # print('Warning: multiple annotations for {}:'.format(image_path_relative))
        old_annotation = image_to_annotations[image_path_relative]
        print(old_annotation['xml_file'])
        print(old_annotation['object'])        
        print(annotation['xml_file'])
        print(annotation['object'])
        print('')
        
        if image_path_relative not in images_with_multiple_xml_files:
            images_with_multiple_xml_files[image_path_relative] = [old_annotation['xml_file']]
        images_with_multiple_xml_files[image_path_relative].append(xml_file)
                
        multiple_xml_handling = 'smart-merge' # 'merge','smart-merge','use-larger'
        
        # Keep whichever is larger...
        if multiple_xml_handling == 'use-larger':
            if len(annotation['object']) > len(old_annotation['object']):
                image_to_annotations[image_path_relative] = annotation
        
        # Keep them all...
        elif multiple_xml_handling == 'merge':
            image_to_annotations[image_path_relative]['object'].extend(annotation['object'])
        
        # Try to keep only unique annotations        
        elif multiple_xml_handling == 'smart-merge':
                    
            old_boxes = image_to_annotations[image_path_relative]['object']
            boxes_to_append = []
            # box = annotation['object'][0]
            for box in annotation['object']:                
                
                b0 = box['bndbox']
                b0 = [int(b0['xmin']),int(b0['ymin']),
                       int(b0['xmax'])-int(b0['xmin']),int(b0['ymax'])-int(b0['ymin'])]                
                
                matches_existing_box = False
                
                for old_box in old_boxes:                                        
                    b1 = old_box['bndbox']
                    b1 = [int(b1['xmin']),int(b1['ymin']),
                           int(b1['xmax'])-int(b1['xmin']),int(b1['ymax'])-int(b1['ymin'])]
                    iou = get_iou(b0,b1)
                    if iou >= iou_merge_threshold:
                        matches_existing_box = True
                        break
                
                if not matches_existing_box:                    
                    boxes_to_append.append(box)
                    
            for box_to_append in boxes_to_append:
                image_to_annotations[image_path_relative]['object'].append(box_to_append)                
                
            
    else:
        
        image_to_annotations[image_path_relative] = annotation

# ...for each annotation file

# path_utils.open_file(os.path.join(base_folder,image_path_relative))
    
images_without_annotations = []

for fn in image_files:
    if fn not in image_to_annotations.keys():
        images_without_annotations.append(fn)

print('\n{} of {} images have no annotations'.format(
    len(images_without_annotations),
    len(image_files)))

print('\n{} of {} annotated images have multiple annotation files'.format(
    len(images_with_multiple_xml_files),
    len(image_to_annotations)))

print('Found a total of {} annotations in {} files'.format(len(all_annotations),
                                                           len(image_to_annotations)))


#%% Convert to COCO camera traps

image_id_to_image = {}
empty_files = []
annotations = []
next_category_id = 0
category_name_to_category_id = {}

# i_image = 1; image_fn_relative = list(image_to_annotations.keys())[i_image]
for i_image,image_fn_relative in tqdm(enumerate(image_to_annotations.keys()),
                                      total=len(image_to_annotations)):
    
    im = {}

    annotation = image_to_annotations[image_fn_relative]
    
    image_fn_abs = os.path.join(base_folder,image_fn_relative)
    
    pil_im = visutils.open_image(image_fn_abs)
    image_w = pil_im.size[0]
    image_h = pil_im.size[1]
    
    assert int(annotation['size']['width']) == image_w
    assert int(annotation['size']['height']) == image_h
        
    im['width'] = image_w
    im['height'] = image_h
    im['annotation_file'] = annotation['xml_file']
    im['file_name'] = image_fn_relative    
    im['id'] = image_fn_relative
    
    # i_obj = 0; obj = annotation['object'][i_obj]
    for i_obj,obj in enumerate(annotation['object']):
    
        ann = {}
        ann['id'] = im['id'] + '_' + str(i_obj).zfill(4)
        ann['image_id'] = im['id']
        
        # COCO camera traps boxes are in absolute, floating-point coordinates
        assert isinstance(obj['bndbox'],dict)
        x = int(obj['bndbox']['xmin'])
        y = int(obj['bndbox']['ymin'])
        w = (int(obj['bndbox']['xmax']) - x)
        h = (int(obj['bndbox']['ymax']) - y)
        
        ann['bbox'] = [x,y,w,h]
                        
        category_name = obj['name']
        
        if category_name not in category_name_to_category_id:
            category_name_to_category_id[category_name] = next_category_id
            next_category_id += 1
        ann['category_id'] = category_name_to_category_id[category_name]
        
        ann['pose'] = obj['pose'].lower()
        ann['truncated'] = int(obj['truncated'])
        ann['difficult'] = int(obj['difficult'])
        
        annotations.append(ann)
                        
    # ...for each object in this image's annotations
    
    assert image_fn_relative not in image_id_to_image
    image_id_to_image[image_fn_relative] = im
        
# ...for each image


#%% Write COCO .json file

images = list(image_id_to_image.values())

print('\nParsed {} annotations for {} images'.format(len(annotations),len(images)))

info = {}
info['version'] = '2023.09.14'
info['description'] = 'UND duck survey data'

categories = []
for category_name in category_name_to_category_id:
    category_id = category_name_to_category_id[category_name]
    category = {
        'id':category_id,
        'name':category_name
        }
    categories.append(category)

d = {}
d['images'] = images
d['annotations'] = annotations
d['categories'] = categories
d['info'] = info 
d['images_without_annotations'] = images_without_annotations

with open(output_json_file,'w') as f:
    json.dump(d,f,indent=2)

print('Finished writing output to {}'.format(output_json_file))


#%% Scrap

if False:

    #%% Read .json file and render one image
    
    from collections import defaultdict
    
    with open(output_json_file,'r') as f:
        d = json.load(f)
    
    image_id_to_annotations = defaultdict(list)
    for ann in d['annotations']:
        image_id_to_annotations[ann['image_id']].append(ann)
    
    i_image = 1000
    im = d['images'][i_image]
    annotations = image_id_to_annotations[im['id']]
    print('Found {} annotations for this image'.format(len(annotations)))
    
    category_id_to_name = {c['id']:c['name'] for c in d['categories']}
    
    boxes = []
    categories = []
    for ann in annotations:
        boxes.append(ann['bbox'])
        categories.append(ann['category_id'])
        
    input_file = os.path.join(base_folder,im['file_name'])
    output_file = os.path.join(image_preview_folder,im['file_name'].replace('/','_') + \
                               '_preview.jpg')
    
    visutils.draw_db_boxes_on_file(input_file, output_file, 
                                   boxes=boxes, classes=categories,
                                   label_map=category_id_to_name, thickness=4, expansion=0)
    
    path_utils.open_file(output_file)
    
    
    #%% Check DB integrity
    
    json_source = output_json_file
    # json_source = os.path.expanduser('~/data/und-ducks/und-ducks-binary/und-ducks-binary.json')
    
    from data_management.databases import integrity_check_json_db
    
    options = integrity_check_json_db.IntegrityCheckOptions()
    options.baseDir = base_folder
    options.bCheckImageSizes = False
    options.bCheckImageExistence = True
    options.bFindUnusedImages = False
    options.bRequireLocation = False
    
    sorted_categories, _, _= integrity_check_json_db.integrity_check_json_db(json_source, options)
    
    """
    [{'id': 6, 'name': 'other_waterfowl', '_count': 876},
     {'id': 5, 'name': 'blue_winged_teal', '_count': 622},
     {'id': 0, 'name': 'unknown', '_count': 561},
     {'id': 2, 'name': 'northern_shoveler', '_count': 502},
     {'id': 3, 'name': 'american_coot', '_count': 411},
     {'id': 4, 'name': 'mallard', '_count': 101},
     {'id': 10, 'name': 'gadwall', '_count': 73},
     {'id': 8, 'name': 'lesser_scaup', '_count': 33},
     {'id': 12, 'name': 'red_winged_blackbird', '_count': 31},
     {'id': 14, 'name': 'yellow_headed_blackbird', '_count': 31},
     {'id': 1, 'name': 'other_songbird', '_count': 25},
     {'id': 11, 'name': 'northern_pintail', '_count': 21},
     {'id': 15, 'name': 'common_grackle', '_count': 14},
     {'id': 9, 'name': 'redhead', '_count': 13},
     {'id': 17, 'name': 'ruddy_duck', '_count': 12},
     {'id': 18, 'name': 'american_wigeon', '_count': 12},
     {'id': 13, 'name': 'canada_goose', '_count': 11},
     {'id': 7, 'name': 'pied_billed_grebe', '_count': 10},
     {'id': 16, 'name': 'canvasback', '_count': 3},
     {'id': 20, 'name': 'green_winged_teal', '_count': 3},
     {'id': 19, 'name': 'horned_grebe', '_count': 2}]
    """
    ''
    
    
    #%% Preview some images
    
    from md_visualization import visualize_db
    from md_utils.path_utils import open_file
    
    viz_options = visualize_db.DbVizOptions()
    viz_options.num_to_visualize = None
    viz_options.trim_to_images_with_bboxes = True
    viz_options.add_search_links = False
    viz_options.sort_by_filename = False
    viz_options.parallelize_rendering = True
    viz_options.include_filename_links = True
    viz_options.viz_size = (1280,-1)
    
    html_output_file, _ = visualize_db.process_images(db_path=json_source,
                                                        output_dir=image_preview_folder,
                                                        image_base_dir=base_folder,
                                                        options=viz_options)
    open_file(html_output_file)
