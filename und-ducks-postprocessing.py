########
#
# und-ducks-postprocessing.py
#
# Stuff we do with model results after inference:
#
# * Generate patch-level preview pages given a set of image-level results.
#
# * Generate estimated image-level counts from image-level results (and write to .csv).
#
########

#%% Constants and imports

import os
import json
import random
import pandas as pd

from tqdm import tqdm

import importlib
und_ducks_inference = importlib.import_module('und-ducks-inference')

from md_visualization import visualization_utils as vis_utils
from md_utils import path_utils

default_preview_confidence_thresholds = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
default_counting_confidence_thresholds = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]


#%% Support functions

def intersects(box1,box2):
    """
    Determine whether two rectangles in x,y,w,h format intersect
    """
    
    # Convert to x1,y1,x2,y2 format
    r1 = [box1[0],box1[1],box1[0]+box1[2],box1[1]+box1[3]]
    r2 = [box2[0],box2[1],box2[0]+box2[2],box2[1]+box2[3]]
    
    if (r1[0]>=r2[2]) or (r1[2]<=r2[0]) or (r1[3]<=r2[1]) or (r1[1]>=r2[3]):
        return False
    else:
        return True


#%% Postprocessing functions

def patch_level_preview(image_level_results_file,image_folder_base,preview_folder,
                        n_patches=10000,preview_confidence_thresholds=None):
    """
    Given an image-level results file:
    
    * Map each image name to a list of detections
    * Randomly sample patches from that results file
    * For each sampled patch
      * Find all results within that patch, convert to patch-relative coordinates
      * Extract the patch and write to disk
    * Generate the preview page, including counts for each patch
    """
    
    if preview_confidence_thresholds is None:
        preview_confidence_thresholds = default_preview_confidence_thresholds
        
    assert os.path.isfile(image_level_results_file)
    assert os.path.isdir(image_folder_base)
    os.makedirs(preview_folder,exist_ok=True)
    
    with open(image_level_results_file,'r') as f:
        image_level_results = json.load(f)
        
    print('Loaded results for {} images from {}'.format(
        len(image_level_results['images']),image_level_results_file))
            
    relative_image_fn_to_results = {}
    
    # Map image filenames to results for that image
    # im = image_level_results['images'][0]
    for im in image_level_results['images']:
        relative_image_fn_to_results[im['file']] = im
          
    # Convert relative coordinates to absolute coordinates, and compute anything
    # we need to compute once per image (like patch boundaries)
    
    # im = image_level_results['images'][0]
    for im in image_level_results['images']:
                
        image_file_relative = im['file']
        image_file_absolute = os.path.join(image_folder_base,image_file_relative)
        pil_im = vis_utils.open_image(image_file_absolute)
        
        # Coordinates are x/y/w/h, in normalized coordinates, we need to un-normalize
        image_size = pil_im.size
        
        detections_absolute = []
        # det = im['detections'][0]
        for det in im['detections']:
            assert 'bbox' in det and len(det['bbox']) == 4
            bbox_absolute = [
                det['bbox'][0] * image_size[0],
                det['bbox'][1] * image_size[1],
                det['bbox'][2] * image_size[0],
                det['bbox'][3] * image_size[1]
                ]
            det_absolute = {}
            det_absolute['bbox'] = bbox_absolute
            det_absolute['conf'] = det['conf']
            det_absolute['category'] = det['category']
            detections_absolute.append(det_absolute)
        
        im['detections_absolute'] = detections_absolute
        
        im['image_size'] = image_size
        
        patch_boundaries = und_ducks_inference.get_patch_boundaries(image_size,
                                                                    und_ducks_inference.patch_size,
                                                                    patch_stride=None)
        im['patch_boundaries'] = patch_boundaries
        
        assert len(im['detections_absolute']) == len(im['detections'])
        
    # ...for each image        
    
    del image_size
    
    patch_size = und_ducks_inference.patch_size
    
    # Generate a list of image/patch tuples to sample
    all_image_patch_tuples = []
    for i_image,fn in enumerate(relative_image_fn_to_results.keys()):
        patch_boundaries = relative_image_fn_to_results[fn]['patch_boundaries']
        for i_patch,patch_xy in enumerate(patch_boundaries):
            all_image_patch_tuples.append((fn,patch_xy))
    
    del patch_boundaries
            
    if len(all_image_patch_tuples) <= n_patches:
        sampled_patch_tuples = all_image_patch_tuples
    else:
        random.seed(0)
        sampled_patch_tuples = random.sample(all_image_patch_tuples,n_patches)
    
    patch_folder = os.path.join(preview_folder,'patches')
    os.makedirs(patch_folder,exist_ok=True)
    
    patch_level_results = {}
    patch_level_results['info'] = image_level_results['info']
    patch_level_results['detection_categories'] = image_level_results['detection_categories']
    patch_level_results['images'] = []
    
    # i_patch = 0; patch = sampled_patch_tuples[i_patch]
    #
    # TODO: parallelize this loop
    for i_patch,patch in tqdm(enumerate(sampled_patch_tuples),total=len(sampled_patch_tuples)):
    
        image_fn_relative = patch[0]
        image_fn_absolute = os.path.join(image_folder_base,image_fn_relative)
        assert os.path.isfile(image_fn_absolute)
        patch_xy = patch[1]
        
        # Generate a usable filename for this patch
        image_name = und_ducks_inference.relative_path_to_image_name(image_fn_relative)
        patch_fn_relative = und_ducks_inference.patch_info_to_patch_name(
            os.path.splitext(image_name)[0],patch_xy[0],patch_xy[1]) + '.jpg'
        patch_fn_absolute = os.path.join(patch_folder,patch_fn_relative)
        
        pil_im = vis_utils.open_image(image_fn_absolute)
        
        # Extract the patch
        _ = und_ducks_inference.extract_patch_from_image(pil_im,patch_xy,
                                     patch_image_fn=patch_fn_absolute,
                                     patch_folder=None,image_name=None,overwrite=True)
        
        # Find detections that overlap with this patch, and convert to patch-relative,
        # normalized coordinates
        im_results = relative_image_fn_to_results[image_fn_relative]
        detections_this_image = im_results['detections_absolute']
        
        detections_this_patch = []
        
        # x/y/w/h
        r1 = [patch_xy[0],patch_xy[1],patch_size[0],patch_size[1]]
        
        # image_level_detection = detections_this_image[0]
        for image_level_detection in detections_this_image:
            
            r2 = image_level_detection['bbox']
            
            if intersects(r1,r2):
                
                patch_level_detection = {}
                patch_level_detection['conf'] = image_level_detection['conf']
                patch_level_detection['category'] = image_level_detection['category']
                
                # Convert to patch-relative absolute coordinates
                patch_bbox = [
                    image_level_detection['bbox'][0] - patch_xy[0],
                    image_level_detection['bbox'][1] - patch_xy[1],
                    image_level_detection['bbox'][2],
                    image_level_detection['bbox'][3]
                    ]
                    
                # Now normalize to patch-relative normalized coordinates
                patch_bbox[0] = patch_bbox[0]/patch_size[0]
                patch_bbox[1] = patch_bbox[1]/patch_size[1]
                patch_bbox[2] = patch_bbox[2]/patch_size[0]
                patch_bbox[3] = patch_bbox[3]/patch_size[1]
                
                patch_level_detection['bbox'] = patch_bbox
                                      
                detections_this_patch.append(patch_level_detection)
            
            # ...if this detection is within this patch
            
        # ...for every detection on this image
        
        # Add patch-specific results to patch_level_results['images']
        patch_im = {}
        patch_im['detections'] = detections_this_patch
        patch_im['file'] = patch_fn_relative
        patch_level_results['images'].append(patch_im)
        
    # ...for every sampled patch
    
    # Write out a .json file with the patch-level results
    patch_level_results_file = os.path.join(preview_folder,'patch_level_results.json')
    with open(patch_level_results_file,'w') as f:
        json.dump(patch_level_results,f,indent=1)
    
    # Generate the preview page
    preview_page_folder = os.path.join(preview_folder,'preview_page')
    os.makedirs(preview_page_folder,exist_ok=True)
    
    from api.batch_processing.postprocessing.postprocess_batch_results import (
        PostProcessingOptions, process_batch_results)
    
    base_task_name = os.path.basename(image_level_results_file)
        
    html_files = []
    
    for confidence_threshold in preview_confidence_thresholds:
        
        options = PostProcessingOptions()
        options.image_base_dir = patch_folder
        options.include_almost_detections = True
        options.num_images_to_sample = None # n_patches
        options.confidence_threshold = confidence_threshold
        options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
        options.ground_truth_json_file = None
        options.separate_detections_by_category = False
        # options.sample_seed = 0
        
        options.parallelize_rendering = False
        options.parallelize_rendering_n_cores = 16
        options.parallelize_rendering_with_threads = False
        
        output_base = os.path.join(preview_page_folder,
            base_task_name + '_{:.3f}'.format(options.confidence_threshold))
        
        os.makedirs(output_base, exist_ok=True)
        print('Processing to {}'.format(output_base))
        
        options.api_output_file = patch_level_results_file
        options.output_dir = output_base
        ppresults = process_batch_results(options)
        html_output_file = ppresults.output_html_file
        
        # path_utils.open_file(html_output_file)
        html_files.append(html_output_file)
    
    # ...for each confidence threshold

    return html_files
    
# ...def patch_level_preview()


def image_level_counting(image_level_results_file,
                         image_name_prefix,
                         drive_root_path,
                         output_file=None,
                         overwrite=False,
                         counting_confidence_thresholds=None):
    """
    Given an image-level results file:
        
    * Count the number of occurrences in each class above a few different thresholds
    * Write the resulting counts to .csv
    
    'image_name_pefix' is everything between the drive root path, e.g.:
    
    /media/user/My Passport1'
     
    ...and a particular image in a results file.  E.g. for the results file:
    
    media_user_My_Passport_2022-10-16_md_results_image_level_nms.json
     
    ...in which filenames look like:
    
    CAM3/CAM30033.JPG
     
    The prefix will be:
    
    2022-10-12/
    
    **eval** is a special-case prefix for handling the eval set.
    """
    
    if counting_confidence_thresholds is None:
        counting_confidence_thresholds = default_counting_confidence_thresholds
    
    if output_file is None:
        output_file = image_level_results_file + '_counts.csv'

    if os.path.isfile(output_file) and not overwrite:
        print('Output file {} exists and overwrite=False, skiping'.format(output_file))
        return
        
    with open(image_level_results_file,'r') as f:
        image_level_results = json.load(f)
        
    print('Loaded image-level results for {} images'.format(
        len(image_level_results['images'])))
    
    # Make sure images are unique
    image_filenames = [im['file'] for im in image_level_results['images']]
    assert len(image_filenames) == len(set(image_filenames))
    
    category_names = image_level_results['detection_categories'].values()
    category_names = [s.lower() for s in category_names]
    category_id_to_name = {}
    for category_id in image_level_results['detection_categories']:
        category_id_to_name[category_id] = \
            image_level_results['detection_categories'][category_id].lower()
    category_name_to_id = {v: k for k, v in category_id_to_name.items()}
    
    # This will be a list of dicts with fields
    #
    # image_path_local (str)
    # image_path_original (str)
    # confidence_threshold
    # E.g.: 'count_brant', 'count_other', 'count_gull', 'count_canada', 'count_emperor'
    results = []
    
    # im = image_level_results['images'][0]
    for im in tqdm(image_level_results['images']):
        
        image_path_prefix_relative = im['file']
        
        if image_name_prefix == '**eval**':
            # For the eval set, filenames look like
            # 'val-images/2019_Replicate_2019-10-11_Cam3_CAM39080.JPG'            
            prefix = '2017-2019/01_JPGs/'
            fn = os.path.basename(image_path_prefix_relative)
            fn = fn.replace('Replicate_','Replicate*').replace('Out_lagoon','Out*lagoon')
            fn = fn.replace('_','/')
            fn = fn.replace('*','_')
            image_path_drive_relative = prefix + fn
            
        else:
            image_path_drive_relative = os.path.join(image_name_prefix,image_path_prefix_relative)
        
        image_path_absolute = os.path.join(drive_root_path,image_path_drive_relative)
        assert os.path.isfile(image_path_absolute)
        
        for confidence_threshold_set in counting_confidence_thresholds:
            
            category_id_to_count = {}
            for cat_id in image_level_results['detection_categories'].keys():
                category_id_to_count[cat_id] = 0
            
            category_id_to_threshold = {}
            
            # If we're using the same confidence threshold for all classes
            if isinstance(confidence_threshold_set,float):
                for cat_id in category_id_to_count:
                    category_id_to_threshold[cat_id] = confidence_threshold_set
            # Otherwise this should map category *names* (not IDs) to thresholds
            else:
                assert isinstance(confidence_threshold_set,dict)
                assert len(category_name_to_id) == len(confidence_threshold_set)
                for category_name in category_name_to_id:
                    assert category_name in confidence_threshold_set, \
                        'No threshold mapping for category {}'.format(category_name)
                for category_name in confidence_threshold_set:
                    category_id_to_threshold[category_name_to_id[category_name]] = \
                        confidence_threshold_set[category_name]                        
            
            # det = im['detections'][0]
            for det in im['detections']:
                
                confidence_threshold = category_id_to_threshold[det['category']]
                if det['conf'] >= confidence_threshold:                
                    category_id_to_count[det['category']] = \
                        category_id_to_count[det['category']] + 1
            
            # ...for each detection
            
            im_results = {}
            im_results['image_path_drive_relative'] = image_path_drive_relative
            im_results['confidence_threshold_string'] = str(confidence_threshold_set)
            
            for category_id in category_id_to_count:
                im_results['count_' + category_id_to_name[category_id]] = \
                    category_id_to_count[category_id]
                im_results['threshold_' + category_id_to_name[category_id]] = \
                    category_id_to_threshold[category_id]
        
            results.append(im_results)
            
        # ...for each confidence threhsold        
        
    # ...for each image
    
    # Convert to a dataframe
    df = pd.DataFrame.from_dict(results)
    df.to_csv(output_file,header=True,index=False)
    
# ...def image_level_counting(...)


#%% Interactive driver

if False:
    
    #%% Preview
    
    n_patches = 2000
    preview_confidence_thresholds = [0.5,0.6,0.7]
    image_level_results_base = os.path.expanduser('~/tmp/und-ducks/image_level_results/')
    default_image_folder_base = '/media/user/My Passport/2020_pair_surveys_UND_SFelege'
    
    # Use this to specify non-default image folder bases for individual files
    result_file_to_folder_base = {}
    
    image_level_results_filenames = os.listdir(image_level_results_base)
    image_level_results_filenames = [fn for fn in image_level_results_filenames if \
                                     fn.endswith('.json')]        
    image_level_results_filenames = [fn for fn in image_level_results_filenames if \
                                     'nms' in fn]
    
    image_level_results_filenames.sort()

    preview_folder_base = os.path.expanduser('~/tmp/und-ducks/preview')    
    
    html_files = []
    
    # TODO: parallelize this loop
    #
    # i_file = 0; image_level_results_file = image_level_results_filenames[i_file]
    for i_file,image_level_results_file in enumerate(image_level_results_filenames):
        
        print('\Generating preview for file {} of {}'.format(
            i_file,len(image_level_results_filenames)))
        
        if image_level_results_file in result_file_to_folder_base:
            image_folder_base = result_file_to_folder_base[image_level_results_file]        
        else:
            image_folder_base = default_image_folder_base
            
        assert os.path.isdir(image_folder_base)
    
        preview_folder = os.path.join(preview_folder_base,
                                      os.path.splitext(os.path.basename(image_level_results_file))[0])
                
        image_level_results_file_absolute = os.path.join(image_level_results_base,
                                                         image_level_results_file)
        image_html_files = patch_level_preview(image_level_results_file_absolute,
                                               image_folder_base,preview_folder,
                                               n_patches=n_patches,
                                               preview_confidence_thresholds=preview_confidence_thresholds)
        html_files.extend(image_html_files)
    
    # ...for each results file
    
    for fn in html_files:
        path_utils.open_file(fn)

    
    #%% Counting
    
    # TODO: this has not yet been ported from the USGS-goose script
    
    output_file = None
    overwrite = True
    counting_confidence_thresholds = None
    drive_root_path = '/media/user/My Passport'
    assert os.path.isdir(drive_root_path)
    
    image_level_results_base = os.path.expanduser('~/tmp/usgs-inference/image_level_results')
    image_level_results_filenames = os.listdir(image_level_results_base)
    image_level_results_filenames = [fn for fn in image_level_results_filenames if \
                                     fn.endswith('.json')]
    image_level_results_filenames = [fn for fn in image_level_results_filenames if 'nms' in fn]
    
    image_level_results_filenames.sort()
        
    image_results_file_relative_to_prefix = {}
    
    confidence_threshold_sets = [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675]
    
    # E.g.: 'count_brant', 'count_other', 'count_gull', 'count_canada', 'count_emperor'
    confidence_threshold_sets.append(
        {'brant':0.65, 'other':0.65, 'gull':0.65, 'canada':0.625, 'emperor':0.6})
    
    # image_level_results_file_relative = image_level_results_filenames[0]
    for image_level_results_file_relative in image_level_results_filenames:
        
        if image_level_results_file_relative in image_results_file_relative_to_prefix:
            image_name_prefix = image_results_file_relative_to_prefix[image_level_results_file_relative]
        elif 'eval' in image_level_results_file_relative:
            image_name_prefix = '**eval**'
        else:
            # The prefix is everything between the universal root path, e.g.:
            #
            # /media/user/My Passport1'
            # 
            # ...and a particular image in a results file.  E.g. for the results file:
            #
            # media_user_My_Passport_2022-10-16_md_results_image_level_nms.json
            # 
            # ...in which filenames look like:
            #
            # CAM3/CAM30033.JPG
            # 
            # The prefix will be:
            #
            # 2022-10-12/
            image_name_prefix = os.path.basename(image_level_results_file_relative).\
                replace('My_Passport','My Passport').\
                split('_')[3] + '/'
            # assert image_name_prefix.startswith('2022') and len(image_name_prefix) == 11
    
        image_level_results_file = os.path.join(image_level_results_base,
                                                image_level_results_file_relative)
        
        image_level_counting(image_level_results_file,
                             image_name_prefix,
                             drive_root_path,
                             output_file=None,
                             overwrite=True,
                             counting_confidence_thresholds=confidence_threshold_sets)    

    # ...for each results file

    
    #%% Zip results files                            
    
    image_level_results_base = os.path.expanduser('~/tmp/und-ducks/image_level_results/')
    image_level_results_filenames = os.listdir(image_level_results_base)
    image_level_results_filenames = [fn for fn in image_level_results_filenames if \
                                     fn.endswith('.csv')]
    image_level_results_filenames = [os.path.join(image_level_results_base,fn) for \
                                     fn in image_level_results_filenames]

    import zipfile
    from zipfile import ZipFile

    output_path = image_level_results_base

    def zip_file(fn, overwrite=True):
        
        basename = os.path.basename(fn)
        zip_file_name = os.path.join(output_path,basename + '.zip')
        
        if (not overwrite) and (os.path.isfile(zip_file_name)):
            print('Skipping existing file {}'.format(zip_file_name))
            return
        
        print('Zipping {} to {}'.format(fn,zip_file_name))
        
        with ZipFile(zip_file_name,'w',zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(fn,arcname=basename,compresslevel=9,compress_type=zipfile.ZIP_DEFLATED)

    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(len(image_level_results_filenames))
    with tqdm(total=len(image_level_results_filenames)) as pbar:
        for i,_ in enumerate(pool.imap_unordered(zip_file,image_level_results_filenames)):
            pbar.update()
