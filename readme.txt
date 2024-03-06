this folder consists of:
1. Python file:
    1. <Image_classification_localization.py>
    2. <Image_classification_localization_for_cronjob.py>



* functions in "Image_classification_localization.py" "script:
    a func. to convert .hrpt images into .png                                                    <hrpt_decode>
    a func. to go save all errors from csv file into the database                                <write_to_db_from_csv>
    a func. to write correlated RF events of a specific image to the database                    <write_aquired_RF_event_records_v2>
    a func. to excute any given query                                                            <excute_query>
    a func. to correlate Image degredation errors and RF events                                  <correlate_ml_rf>
    a func. to fetch all correlated RF events for a specific image                               <get_correlated_rf_events>
    a func. to localize image degredation errors on original image                               <run_localize_on_original>
    a func. to convert a given pixel into date and time                                          <from_pixl_to_time_frame>
    a func. to localize image degredation errors on partitioned images                           <run_localization>
    a func. to get the bounding box for white and black type image degredation                   <get_white_black_bounding_boxs>
    a func. to get localtion of point type image degredation error                               <get_scatter_location>
    a func. to call localize on original and localize partitions                                 <localize>
    a func. to invoke sagemaker deployed endpoint to classify a given partitioned image          <classify_fn>
    a func. to convert images from other formats (ex: gif) to png                                <covert_image_format_png>
    a func. to partition the whole image into sub images                                         <partition_image_to_sub_images>
    a func. to call other partitioning function when needed                                      <main_partition_fn>
    a func. to copy images between s3 prifexs                                                    <copy_from_s3_to_s3>
    a func. to list content of a given s3 prifexs                                                <list_s3_files>
    a func. to upload all content of a given local path to a given s3 prefix                     <uploading_to_s3>
    a func. to download images from s3 to a given local path                                     <load_image_and_save_local>
    a func. to upload a given file to a given s3 prefix                                          <save_to_s3>
    a func. to do processing on the original image like adding frame and text                    <main_processing_main_img>
    a func. to do processing on the partitioned image like adding frame and text                 <main_processing_sub_img>






output will be saved to:

                "bucket_name/stand_alone/"station name"/"Satellite_name"/operation/processed_images/"
    example:

                "s3://rfims-prototype/stand_alone/AOML/NOAA18/operation/processed_images/"


Usage:
1. conda activate pytorch_p39
2. python /location of the file/Image_classification_localization.py



Another way is to call attached bash file
1. ./location of the file/Image_classification_localization_for_cronjob.sh




