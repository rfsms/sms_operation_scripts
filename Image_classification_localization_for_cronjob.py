
import shutil
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from Image_classification_localization import *



if __name__ == "__main__":
    event = {
        "img_path": "",
        "bucket_name": "rfims-prototype",
        "ENDPOINT_NAME": "rfims-prototype-ml-t2-medium-v2",
        "font": "operation/font/Arial.ttf",
        "db_host": "localhost",
        "db_user_name": "sms",
        "db_password": "ABCD1234",
        "db_name": "stand_alone"
    }

    s3 = boto3.resource('s3')
    s3_prefixes = ["stand_alone/AOML/NOAA18/To_Be_Processed/level0/","stand_alone/AOML/NOAA19/To_Be_Processed/level0/","stand_alone/AOML/METOPB/To_Be_Processed/level0/",
                   "stand_alone/table_mountain/NOAA18/To_Be_Processed/level0/"]
    for prefix in s3_prefixes:
        src_files = list_s3_files(event["bucket_name"], prefix)
        for input_Img_path in src_files:
            excute_flag = False
            input_Img_path = input_Img_path.split(event["bucket_name"])[-1][1:]
            if ".hrpt" in input_Img_path:
                excute_flag = True
                executable_path = "/home/ubuntu/project/HRPT-Decoder/build/hrpt-decoder"
                bucket_name = event["bucket_name"]
                hrpt_img_path_on_S3 = input_Img_path
                sub_folder = hrpt_img_path_on_S3.split("/")[4]
                file_name = hrpt_img_path_on_S3.split("/")[-1]
                jpg_img_path_on_S3 = "/".join(hrpt_img_path_on_S3.split("/")[0:3]) + "/operation/imported_images/" + sub_folder + "/" + file_name
                local_save_path = "/".join(hrpt_img_path_on_S3.split("/")[0:-1]) + "/"
                os.makedirs(local_save_path, exist_ok=True)
                save_path = load_image_and_save_local(bucket_name, hrpt_img_path_on_S3, local_save_path, file_name)

                out_path = local_save_path + file_name.split(".hrpt")[0] + ".png"
                hrpt_decode(executable_path, save_path, out_path)

                save_to_s3(out_path, bucket_name, jpg_img_path_on_S3.split(".hrpt")[0] + ".png")

                shutil.rmtree(local_save_path)


                event["img_path"] = jpg_img_path_on_S3.split(".hrpt")[0] + ".png"

            elif any(word in input_Img_path for word in [".png",".jpg","jpeg",".gif"]):
                excute_flag = True
                event["img_path"] = input_Img_path


            if excute_flag:
                res = main(event)
                s3.Object(event["bucket_name"], input_Img_path).delete()
                print(input_Img_path + " Processed and deleted")
                print("*"*50)
            elif ".csv" not in input_Img_path:
                print(input_Img_path + " Not accepted format")


