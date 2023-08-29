########
#
# und-ducks-inference.py
#
# This file is currently a stub; I've deleted all the content except the constants that
# are unique to this model, to force myself to merge this functionality with 
# usgs-geese-inference.py the next time I run inference with this model
#
########

#%% Constants and imports

expected_yolo_category_id_to_name = {0: 'unknown',
     1: 'other_songbird',
     2: 'northern_shoveler',
     3: 'american_coot',
     4: 'mallard',
     5: 'blue_winged_teal',
     6: 'other_waterfowl',
     7: 'pied_billed_grebe',
     8: 'lesser_scaup',
     9: 'redhead',
     10: 'gadwall',
     11: 'northern_pintail',
     12: 'red_winged_blackbird',
     13: 'canada_goose',
     14: 'yellow_headed_blackbird',
     15: 'common_grackle',
     16: 'canvasback',
     17: 'ruddy_duck',
     18: 'american_wigeon',
     19: 'horned_grebe',
     20: 'green_winged_teal',
     21: 'other_waterbird'
}


#%% Interactive driver

if False:
    
    pass

    #%%
    
    input_folder_base = '/media/user/My Passport/2020_pair_surveys_UND_SFelege'
        
    results = run_model_on_folder(input_folder_base,recursive=True)
    

#%% Scrap

if False:

    pass
    
    #%% Unused variable suppression
    
    patch_results_after_nms_file = None
    patch_folder_for_folder = None
    
    
    #%% Preview results for patches at a variety of confidence thresholds
    
    patch_results_file = patch_results_after_nms_file
            
    from api.batch_processing.postprocessing.postprocess_batch_results import (
        PostProcessingOptions, process_batch_results)
    
    postprocessing_output_folder = os.path.join(project_dir,'preview')

    base_task_name = os.path.basename(patch_results_file)
        
    for confidence_threshold in [0.4,0.5,0.6,0.7,0.8]:
        
        options = PostProcessingOptions()
        options.image_base_dir = patch_folder_for_folder
        options.include_almost_detections = True
        options.num_images_to_sample = 7500
        options.confidence_threshold = confidence_threshold
        options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
        options.ground_truth_json_file = None
        options.separate_detections_by_category = True
        # options.sample_seed = 0
        
        options.parallelize_rendering = True
        options.parallelize_rendering_n_cores = 16
        options.parallelize_rendering_with_threads = False
        
        output_base = os.path.join(postprocessing_output_folder,
            base_task_name + '_{:.3f}'.format(options.confidence_threshold))
        
        os.makedirs(output_base, exist_ok=True)
        print('Processing to {}'.format(output_base))
        
        options.api_output_file = patch_results_file
        options.output_dir = output_base
        ppresults = process_batch_results(options)
        html_output_file = ppresults.output_html_file
        
        path_utils.open_file(html_output_file)
    

    #%% Render boxes on one of the original images
    
    input_folder_base = '/media/user/My Passport/2020_pair_surveys_UND_SFelege'
    md_results_image_level_nms_fn = os.path.expanduser(
        '~/tmp/und-ducks/image_level_results/'+ \
        'media_user_My_Passport_2020_pair_surveys_UND_SFelege_md_results_image_level_nms.json')
    
    with open(md_results_image_level_nms_fn,'r') as f:
        md_results_image_level = json.load(f)

    i_image = 0
    output_image_file = os.path.join(project_dir,'test.jpg')
    detections = md_results_image_level['images'][i_image]['detections']    
    image_fn_relative = md_results_image_level['images'][i_image]['file']
    image_fn = os.path.join(input_folder_base,image_fn_relative)
    assert os.path.isfile(image_fn)
    
    detector_label_map = {}
    for category_id in yolo_category_id_to_name:
        detector_label_map[str(category_id)] = yolo_category_id_to_name[category_id]
        
    vis_utils.draw_bounding_boxes_on_file(input_file=image_fn,
                          output_file=output_image_file,
                          detections=detections,
                          confidence_threshold=0.4,
                          detector_label_map=detector_label_map, 
                          thickness=1, 
                          expansion=0)
    
    path_utils.open_file(output_image_file)
    
