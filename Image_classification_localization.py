
import shutil
from datetime import timedelta
import json
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import math
import boto3
import numpy as np
import logging
import datetime
import pandas as pd
import pymysql.cursors
import cv2
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import subprocess
import uuid


######################################################################################################################################################
# Image processing
######################################################################################################################################################


def add_txt_to_image(img, txt, font_path, loc_point=(5, 5), txt_color=(255, 165, 0), txt_font_size=15):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, txt_font_size)
    draw.text(loc_point, txt, txt_color, font=font)
    return img


def add_frame_to_image(img, frame_size_h, frame_size_w):
    img_size = img.size
    new_size = (img_size[1] + frame_size_h, img_size[0] + frame_size_w, 3)
    new_img = np.ones(new_size, dtype=np.uint8) * 255

    new_img = Image.fromarray(new_img, "RGB")
    ing_new_size = (new_size[1], new_size[0])

    box = tuple((n - o) // 2 for n, o in zip(ing_new_size, img_size))

    new_img.paste(img, box)
    return new_img


def draw_h_v_lines_to_img(img, shape, color=(255, 165, 0)):
    img1 = ImageDraw.Draw(img)
    img1.line(shape, fill=color, width=4)
    return img

#add text to sub image
def main_processing_sub_img(img_local_path, save_local_path, font_path, frame_size=100, sub_img_w=224, sub_img_h=224):
    print('Starting processing for sub_image ' + img_local_path)
    txt_color = (255, 165, 0)
    img = Image.open(img_local_path)
    img = img.convert("RGB")
    img_name = img_local_path.split("/")[-1].split(".")[0]
    col = img_name.split("_")[-1]
    row = img_name.split("_")[-2]
    txt = col + "_" + row
    new_img = add_frame_to_image(img, frame_size, frame_size)
    curent_pix = sub_img_h * int(row) + (frame_size / 2)
    point = (0, curent_pix)
    time_frame = from_pixl_to_time_frame(img_name, (0, curent_pix - (frame_size / 2)))
    time_frame = str(time_frame)
    new_img = add_txt_to_image(new_img, time_frame, font_path, (5, (frame_size / 2) - 20), txt_color=txt_color)
    curent_pix = (sub_img_h * (int(row) + 1)) + (frame_size / 2)
    point = (0, curent_pix)
    time_frame = from_pixl_to_time_frame(img_name, (0, curent_pix - (frame_size / 2)))
    time_frame = str(time_frame)
    new_img = add_txt_to_image(new_img, time_frame, font_path, (5, sub_img_h + frame_size / 2), txt_color=txt_color)
    new_img = add_txt_to_image(new_img, txt, font_path, txt_color=txt_color)
    new_img.save(save_local_path)
    print('Processing endded successfully for image ' + img_local_path)

#add text to main image
def main_processing_main_img(main_img_local_path, save_path, font_path, count_h, count_w, frame_size=2000,
                             sub_img_w=224, sub_img_h=224, time_frame_font_size=50, line_number_font_size=100):
    print('Starting processing for main_image ' + main_img_local_path)
    frame_size_h = int(frame_size / 2)
    frame_size_w = frame_size
    txt_color = (255, 165, 0)
    img_name = main_img_local_path.split("/")[-1].split(".")[0]
    img = Image.open(main_img_local_path)
    img = img.convert("RGB")
    new_main_img_with_frame = add_frame_to_image(img, frame_size_h, frame_size_w)
    main_img_w, main_img_h = img.size
    for i in range(count_h + 1):
        shape = [(0 + frame_size_w / 2, (sub_img_h * (i)) + frame_size_h / 2),
                 (main_img_w + frame_size_w / 2, (sub_img_h * (i)) + frame_size_h / 2)]

        img_with_lines = draw_h_v_lines_to_img(new_main_img_with_frame, shape)

        tmp_row = sub_img_h * i + (frame_size_h / 2)
        point = (100 + main_img_w + frame_size_w / 2, tmp_row + 50)
        img_with_lines = add_txt_to_image(img_with_lines, str(i), font_path, point, txt_color=txt_color,
                                          txt_font_size=line_number_font_size)


    for i in range(count_w + 1):
        shape = [((sub_img_w * (i)) + frame_size_w / 2, 0 + frame_size_h / 2),
                 ((sub_img_w * (i)) + frame_size_w / 2, main_img_h + frame_size_h / 2)]
        img_with_lines = draw_h_v_lines_to_img(new_main_img_with_frame, shape)

        tmp_row = sub_img_w * i + (frame_size_w / 2)
        point = (tmp_row + 50, frame_size_h / 2 - 200)
        img_with_lines = add_txt_to_image(img_with_lines, str(i), font_path, point, txt_color=txt_color,
                                          txt_font_size=line_number_font_size)

    for i in range(count_h + 1):
        tmp_row = sub_img_h * i + (frame_size_h / 2)
        point = (10, tmp_row)
        time_frame = from_pixl_to_time_frame(img_name, (10, tmp_row - (frame_size_h / 2)))
        time_frame = str(time_frame)
        new_img_with_txt = add_txt_to_image(img_with_lines, time_frame, font_path, point, txt_color=txt_color,
                                            txt_font_size=time_frame_font_size)

    new_img_with_txt.save(save_path)
    print('Processing endded successfully for image ' + main_img_local_path)


######################################################################################################################################################
# S3 related
######################################################################################################################################################


def save_to_s3(file_path, bucket_name, s3_save_path):
    print('Starting Saving to S3 bucket ' + file_path)
    client = boto3.client('s3')
    client.upload_file(file_path, bucket_name, s3_save_path)
    print('Saving to S3 Bucket Ended successfully ' + file_path)




def load_image_and_save_local(bucket_name, path_on_S3, save_path, file_name):
    print('Starting loading the image and saving to local path ' + file_name)
    save_path = save_path + file_name
    client = boto3.client('s3')
    client.download_file(bucket_name, path_on_S3, save_path)
    print('Loading image and saving to local path endded successfully')
    print("save path",save_path)
    return save_path



def uploading_to_s3(bucket_name, local_path, partitioned_prefix, img_name):
    print('Starting uploading to s3 ' + img_name)
    images = os.listdir(local_path)
    uploaded_images = 0
    if len(images) > 0:
        for img in images:
            count = 0
            img_path = local_path + img
            s3_save_path = partitioned_prefix + img
            while count < 3:
                try:
                    save_to_s3(img_path, bucket_name, s3_save_path)
                    count = 10
                    uploaded_images += 1
                except:
                    count += 1
    else:
        print('No images to be uploaded')
        return 0
    if uploaded_images > 0:
        print('Uploading to s3 Ended successfully')
    else:
        print('No image has been successfully uploaded')
    return len(images)


def list_s3_files(bucket_name, Prefix):
    print('Starting listing S3 path files')
    s3_client = boto3.client("s3")
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=Prefix)
    files = response.get("Contents")
    files_list = []
    if files:
        for file in files:
            files_list.append('s3://' + bucket_name + "/" + file['Key'])
        print('listing S3 path files Ended')
    return files_list


def copy_from_s3_to_s3(copy_source, bucket_name, save_path):
    print('Starting copying images to classify directory')
    s3 = boto3.resource('s3')
    count = 0
    while count < 3:
        try:
            s3.meta.client.copy(copy_source, bucket_name, save_path)
            count = 10
            print('copy Ended successfully')
            return "done"
        except:
            print('fail to copy image ', copy_source)
            return "fail"


######################################################################################################################################################
# Partitioning
######################################################################################################################################################


def covert_image_format_png(img_path, output_path, bucket_name, s3_prefix, img_name):
    print('Starting Converting to PNG ' + img_path)
    png_img_path = output_path + img_name.split(".")[0] + '.png'
    try:
        if ".gif" in img_path:
            im = Image.open(img_path)
        else:
            opencv_image = cv2.imread(img_path)
            color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(color_coverted)
        mypalette = im.getpalette()
    except IOError:
        print("Cant load", img_path)
        png_img_path = "error loading the image"

    if mypalette:
        im.putpalette(mypalette)
    new_im = Image.new("RGBA", im.size)
    new_im.paste(im)
    new_im.save(png_img_path)
    s3_path = s3_prefix + "/" + img_name.split(".")[0] + ".png"



    save_to_s3(png_img_path, bucket_name, s3_path)
    print('Converting to png Ended successfully')
    return png_img_path


def partition_image_to_sub_images(img_path, out_path, img_name, w=224, h=224):
    print('Starting Partitioning ' + img_name)
    img = np.asanyarray(Image.open(img_path))[:, :, 0:3]
    # img = np.asanyarray(cv2.imread(img_path))[:, :, 0:3]
    img_h = img.shape[0]
    img_w = img.shape[1]
    out_path = out_path + "/partitioned/"
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    h_count, w_count = math.floor(img_h / h), math.floor(img_w / w)

    for i_h in range(h_count):
        for i_w in range(w_count):
            window = img[i_h * h:(i_h + 1) * h, i_w * w:(i_w + 1) * w]
            im = Image.fromarray(window)
            im.save(out_path + img_name.split(".")[0] + "_" + str(i_h) + "_" + str(i_w) + '.png')

        window = img[i_h * h:(i_h + 1) * h, img.shape[1] - w:img.shape[1]]
        im = Image.fromarray(window)

        im.save(out_path + img_name.split(".")[0] + "_" + str(i_h) + "_" + str(i_w + 1) + '.png')

    for i_w in range(w_count):
        window = img[img.shape[0] - h:img.shape[0], i_w * w:(i_w + 1) * w]
        im = Image.fromarray(window)
        im.save(out_path + img_name.split(".")[0] + "_" + str(i_h + 1) + "_" + str(i_w) + '.png')

    window = img[img.shape[0] - h:img.shape[0], img.shape[1] - w:img.shape[1]]
    im = Image.fromarray(window)
    im.save(out_path + img_name.split(".")[0] + "_" + str(i_h + 1) + "_" + str(i_w + 1) + '.png')
    print('Partitioning Ended successfully')
    return out_path, img_w, img_h, w_count, h_count


def main_partition_fn(bucket_name, image_path_on_S3, output_local_path, s3_prefix, font_local_path,img_name):
    partitioned_local_path = None
    uploaded_images = 0
    count = 0
    info = {}
    while count < 3:
        try:
            gif_local_path = load_image_and_save_local(bucket_name, image_path_on_S3, output_local_path, img_name)
            count = 10
        except:
            count += 1
    if count != 10:
        partitioned_local_path = "error in loading image and save local"
    else:
        count = 0
        while count < 3:
            try:
                png_local_path = covert_image_format_png(gif_local_path, output_local_path, bucket_name, s3_prefix,
                                                         img_name)
                if "error" not in png_local_path:
                    count = 10
            except:
                count += 1
        if count != 10:
            partitioned_local_path = "error in converting image to png"
    if partitioned_local_path == None:
        count = 0
        while count < 3:
            try:
                partitioned_local_path, img_w, img_h, w_count, h_count = partition_image_to_sub_images(png_local_path,
                                                                                                       output_local_path,
                                                                                                       img_name)
                info[img_name.split(".")[0]] = {"img_h": img_h, "img_w": img_w, "h_count": h_count, "w_count": w_count}
                png_processed_local_path = png_local_path.split(".")[0] + "_processed.png"
                main_processing_main_img(png_local_path, png_processed_local_path, font_local_path, h_count, w_count)
                s3_path = s3_prefix + "/localized_processed/" + img_name.split(".")[0] + ".png"
                save_to_s3(png_processed_local_path, bucket_name, s3_path)
                count = 10
            except:
                count += 1
        if count != 10:
            partitioned_local_path = "error in partition_image_to_sub_images"
        else:
            uploaded_images = uploading_to_s3(bucket_name, partitioned_local_path, s3_prefix + "/partitioned/",
                                              img_name)

    return partitioned_local_path, uploaded_images, info


######################################################################################################################################################
# Classify
######################################################################################################################################################
def classify_fn(ENDPOINT_NAME, img_url):
    print('Starting classification')
    runtime = boto3.client('runtime.sagemaker')
    count = 0
    while count < 3:
        try:
            response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='application/json',
                                               Body=json.dumps(img_url))
            response_body = response['Body']
            img_class = response_body.read().decode("utf-8")
            img_class = json.loads(img_class)[0]
            count = 10
        except:
            count += 1
            img_class = ["fail"]
    if count == 10:
        print('classification Ended successfully')
    else:
        print('error in classifying ', json.loads(img_class)[0])

    return img_class


######################################################################################################################################################
# main Localize
######################################################################################################################################################
#localization for folder 
def localize(classfied_prefix, out_folder, info, bucket_name, localize_prefix, localized_processed_prefix,
             font_local_path, s3_prefix):
    print('Starting localization')
    error_classified_images = list_s3_files(bucket_name, classfied_prefix + "error/")
    localized_local_path = out_folder + "localized/"
    classified_local_path = out_folder + "classified/"
    localized_processed_local_path = out_folder + "localized_processed/"
    img_name = localize_prefix.split("/")[-3]
    json_filename = img_name + "_error_time_stamp.json"
    open(out_folder + json_filename, "w").close()
    json_file = open(out_folder + json_filename, "a")
    loc = {img_name: info[img_name]}
    No_Error_Flag = False
    if len(error_classified_images) > 0 and any(img.endswith(".png") for img in error_classified_images):
        for img in error_classified_images:
            if img.endswith(".png"):
                try:
                    image = img.split("/")[-1]
                    print("starting localization for image " + image)
                    classfied_image_path_on_S3 = classfied_prefix + "error/" + image
                    __ = load_image_and_save_local(bucket_name, classfied_image_path_on_S3, classified_local_path,
                                                   image)

                    loc[image] = run_localization(image, classified_local_path, localized_local_path)
                    save_to_s3(localized_local_path + image, bucket_name, localize_prefix + image)
                    main_processing_sub_img(localized_local_path + image, localized_processed_local_path +
                                            image.replace(".png", "_processed_with_localization.png"),font_local_path)
                    s3_path = localized_processed_prefix + image.replace(".png", "_processed_with_localization.png")

                    save_to_s3(localized_processed_local_path + image.replace(".png", "_processed_with_localization.png"), bucket_name, s3_path)


                except:
                    print("localization failed for image " + image)
    else:
        print("no classified images found")
        No_Error_Flag = True
        print("image " + img_name + " has no error")
    json.dump(loc, json_file)
    image = list(loc.keys())[0] + ".png"
    __ = load_image_and_save_local(bucket_name, localize_prefix.replace("localized/", "") + image, localized_local_path,
                                   image)
    try:
        original_localized_img_local_path, total_no_of_error, Pass_Date = run_localize_on_original(s3_prefix,loc, json_filename,No_Error_Flag,
                                                                                        localized_local_path, True)
        image = original_localized_img_local_path.split("/")[-1]
        save_to_s3(original_localized_img_local_path, bucket_name, localize_prefix + img_name + ".png")
        save_to_s3(original_localized_img_local_path.replace(".png", ".csv"), bucket_name,
                   localize_prefix + image.split(".")[0] + ".csv")
        save_to_s3(out_folder + json_filename, bucket_name, localize_prefix + json_filename)
        png_processed_local_path = original_localized_img_local_path.split(".")[0] + "_processed_with_localization.png"
        main_processing_main_img(original_localized_img_local_path, png_processed_local_path, font_local_path,
                                 info[img_name]["h_count"], info[img_name]["w_count"])
        s3_path = localized_processed_prefix + img_name + "_with_localization.png"
        save_to_s3(png_processed_local_path, bucket_name, s3_path)
        print(total_no_of_error)


    except:
        print("localization back to original failed for image " + image)
        total_no_of_error = 0
        Pass_Date = datetime.datetime(2000, 1, 1, 00, 00, 00)

    if total_no_of_error >= 0:
        try:
            save_to_s3(original_localized_img_local_path.replace(".png", "_black.csv"), bucket_name,
                       localize_prefix.replace("localized/", "db/") + image.split(".")[0] + "_black.csv")
            save_to_s3(original_localized_img_local_path.replace(".png", "_white.csv"), bucket_name,
                       localize_prefix.replace("localized/", "db/") + image.split(".")[0] + "_white.csv")
            save_to_s3(original_localized_img_local_path.replace(".png", "_point.csv"), bucket_name,
                       localize_prefix.replace("localized/", "db/") + image.split(".")[0] + "_point.csv")
        except:
            print("Failed to upload csv files")
    print('localization Ended successfully')

    return Pass_Date


######################################################################################################################################################
# Localize
######################################################################################################################################################

def get_scatter_location(img, thr=100):
    points = []
    locations = np.where((img[:, :, 0] < thr) & (img[:, :, 0] > 0))

    for i in range(len(locations[0])):
        row, col = locations[0][i], locations[1][i]
        point = (col, row)
        points.append(point)

    return points


def conv_multiply(img, kernal):
    array_img = np.array(img).astype("int32")
    for i in range(1, img.shape[1] - 1):
        for j in range(1, img.shape[0] - 1):
            img_part = img[i - 1:i + 2, j - 1:j + 2, :]
            array_img[i, j, :] = np.sum(np.dot(kernal, img_part)) if np.sum(np.dot(kernal, img_part)) > 0 else 0
    array_img[0, :, :] = array_img[-1, :, :] = array_img[:, 0, :] = array_img[:, -1, :] = 0
    return array_img


def get_index_of_last_value_h(img, str_h, str_w, w=224, value=0):
    col = img[:, str_w, 0].tolist()
    if sum(col[str_h:]) / len(col) != value:
        i = str_h
        idx = None
        if i < w - 1:
            while i < len(col):
                if col[i] != value:
                    idx = i
                    i = len(col) + 1
                i += 1
        else:
            idx = w
    else:
        idx = w
    return idx - 1


def get_white_black_bounding_boxs(img, w=224, value=0):
    boxs = []
    if value == 255:
        idxs_h = np.where(np.amax(img[:, :, 0], axis=1) == value)[0].tolist()
    elif value == 0:
        idxs_h = np.where(np.amin(img[:, :, 0], axis=1) == value)[0].tolist()
    else:
        print("not valid value")
    if idxs_h:
        i = 0
        while i < len(idxs_h):
            idx_h = idxs_h[i]
            str_idx_h_zero_1 = idx_h
            str_idx_w_zero_1 = np.where(img[idx_h, :, 0] == value)[0][0]
            end_idx_w_zero_1 = np.where(img[idx_h, :, 0] == value)[0][-1]
            end_idx_h_zero_1 = get_index_of_last_value_h(img, str_idx_h_zero_1, str_idx_w_zero_1, w, value)

            # represents the top left corner of rectangle
            start_point = (0, str_idx_h_zero_1)

            # represents the bottom right corner of rectangle
            end_point = (w, end_idx_h_zero_1)
            if abs(end_idx_w_zero_1 - str_idx_w_zero_1) + abs(end_point[1] - start_point[1]) > 3:
                boxs.append([start_point, end_point])
            if end_idx_h_zero_1 < w - 1:
                i = idxs_h.index(end_idx_h_zero_1) + 1
            else:
                i = w + 1

    return boxs


def run_localization(img_name, in_path, save_path):
    img_path = in_path + img_name
    white_box_color = (0, 0, 255)
    black_box_color = (0, 255, 0)
    scatter_color = (255, 0, 0)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    img = Image.open(img_path)
    h, w = img.size

    img_copy = img.copy()
    img_drow = ImageDraw.Draw(img_copy)

    kernal = np.array([[-1, -1, -1],
                       [-1, 6.5, -1],
                       [-1, -1, -1]])

    conv_img = conv_multiply(np.array(img), kernal)
    conv_img = conv_img.astype(np.uint8)
    points = get_scatter_location(conv_img, thr=255)
    white_boxs = get_white_black_bounding_boxs(np.array(img), w, value=255)
    black_boxs = get_white_black_bounding_boxs(np.array(img), w, value=0)
    pnts, bb, wb = {}, {}, {}

    if white_boxs:
        print("number of white error are " + str(len(white_boxs)))
        st_txt = "start point of white box "
        en_txt = "end point of white box "
        for i, box in enumerate(white_boxs):
            wb[str(i)] = {"str": (int(box[0][0]), int(box[0][1])), "end": (int(box[1][0]), int(box[1][1]))}
            print(st_txt + str(i) + " is " + str(box[0]))
            print(en_txt + str(i) + " is " + str(box[1]))
            img_drow.rectangle(box, outline=white_box_color, width=3)
    else:
        print("No white Boxs")

    if black_boxs:
        print("number of black error are " + str(len(black_boxs)))
        st_txt = "start point of black box "
        en_txt = "end point of black box "
        for i, box in enumerate(black_boxs):
            bb[str(i)] = {"str": (int(box[0][0]), int(box[0][1])), "end": (int(box[1][0]), int(box[1][1]))}
            print(st_txt + str(i) + " is " + str(box[0]))
            print(en_txt + str(i) + " is " + str(box[1]))
            img_drow.rectangle(box, outline=black_box_color, width=3)
    else:
        print("No Black Boxs")

    if len(points) > 0:
        print("number of scattered errors are " + str(len(points)))
        for i, point in enumerate(points):
            pnts[str(i)] = {"point_loc": (int(point[0]), int(point[1]))}
            print("scatter error nubmer " + str(i) + " location is " + str(point))
            print("scatter error nubmer " + str(i) + " value is " + str(np.array(img)[point[1], point[0], 0]))
            img_drow.ellipse([point[0], point[1], point[0] + 3, point[1] + 3], outline=scatter_color)
    else:
        print("no scattered error")

    img_copy.save(save_path + img_name)
    return {"points": pnts, "black": bb, "white": wb}


######################################################################################################################################################
# back to original
######################################################################################################################################################


def localize_on_original(box, h, w, h_count, w_count, img_h, img_w, new_h, new_w, point_flage=False):
    if h == h_count and w < w_count:
        if not point_flage:
            str_box = (box['str'][0] + w * new_w, img_h - new_h + 1 + box['str'][1])
            end_box = (box['end'][0] + w * new_w, img_h - new_h + 1 + box['end'][1])
        else:
            str_box = (box['point_loc'][0] + w * new_w, img_h - new_h + 1 + box['point_loc'][1])
            end_box = str_box
    elif h == h_count and w == w_count:
        if not point_flage:
            str_box = (img_w - new_w + 1 + box['str'][0], img_h - new_h + 1 + box['str'][1])
            end_box = (img_w - new_w + 1 + box['end'][0], img_h - new_h + 1 + box['end'][1])
        else:
            str_box = (img_w - new_w + 1 + box['point_loc'][0], img_h - new_h + 1 + box['point_loc'][1])
            end_box = str_box
    elif h != h_count and w == w_count:
        if not point_flage:
            str_box = (img_w - new_w + 1 + box['str'][0], box['str'][1] + h * new_h)
            end_box = (img_w - new_w + 1 + box['end'][0], box['end'][1] + h * new_h)
        else:
            str_box = (img_w - new_w + 1 + box['point_loc'][0], box['point_loc'][1] + h * new_h)
            end_box = str_box
    else:
        if not point_flage:
            str_box = (box['str'][0] + w * new_w, box['str'][1] + h * new_h)
            end_box = (box['end'][0] + w * new_w, box['end'][1] + h * new_h)
        else:
            str_box = (box['point_loc'][0] + w * new_w, box['point_loc'][1] + h * new_h)
            end_box = str_box

    return str_box, end_box


def from_pixl_to_time_frame(img_name, point):
    if "noaa" in img_name.lower():
        img_name = img_name[7:].replace("_","")
    elif "HFS" in img_name:
        img_name = img_name[4:]
    img_year = int(img_name[0:4])
    img_month = int(img_name[4:6])
    img_day = int(img_name[6:8])
    img_h = int(img_name[8:10])
    img_mint = int(img_name[10:12])
    img_sec = int(img_name[12:14])

    str_h = point[1]
    elapsed_sec = str_h / 6.0
    str_date = datetime.datetime(img_year, img_month, img_day, img_h, img_mint, img_sec)

    time_frame = str_date + datetime.timedelta(seconds=elapsed_sec)
    return time_frame


def check_cont(str_pnt1, end_p1, str_pnt2, end_p2):
    if str_pnt1[1] == str_pnt2[1] and end_p1[1] == end_p2[1]:
        return True
    else:
        return False


def run_localize_on_original(s3_prefix,json_file, json_file_name, No_Error_Flag, path, original_with_localize_flag=False):
    keys = list(json_file.keys())
    img_name = keys[0]
    file_name = json_file_name.split(".")[0]
    header = ["image name", "error type", "shape h", "shape w", "count h", "count w", "h", "w", "number", "start pixel",
              "end pixel"]
    img_h = json_file[img_name]["img_h"]
    img_w = json_file[img_name]["img_w"]
    h_count = json_file[img_name]["h_count"]
    w_count = json_file[img_name]["w_count"]
    if No_Error_Flag:
        keys = list(json_file.keys())
        img_name = keys[0]
        all_info = [[img_name, "point", img_h, img_w, h_count, w_count, 0, 0, 0, (0, 0), (0,1)],
                    [img_name, "black", img_h, img_w, h_count, w_count, 0, 0, 0, (0, 0), (0, 1)],
                    [img_name, "white", img_h, img_w, h_count, w_count, 0, 0, 0, (0, 0), (0, 1)]]
        num = 1
        shutil.copy(path + img_name + ".png", path + "/" + file_name + ".png")
    else:
        img = Image.open(path + img_name + ".png")
        img_drow = ImageDraw.Draw(img)
        new_w = 224
        new_h = 224
        keys = keys[1:]
        all_info = []
        for key in keys:
            data = json_file[key]
            h = int(key.split(".")[0].split("_")[-2])
            w = int(key.split(".")[0].split("_")[-1])
            points = data["points"]
            blacks = data["black"]
            whites = data["white"]
            white_box_color = (0, 0, 255)
            black_box_color = (0, 255, 0)
            scatter_color = (255, 0, 0)
            thickness = 2
            for black in blacks:

                box = blacks[black]

                str_box, end_box = localize_on_original(box, h, w, h_count, w_count, img_h, img_w, new_h, new_w)

                info = [img_name, "black", img_h, img_w, h_count, w_count, h, w, black, str_box, end_box]
                all_info.append(info)

                if original_with_localize_flag:
                    img_drow.rectangle([str_box, end_box], outline=black_box_color, width=thickness)
            for white in whites:
                box = whites[white]

                str_box, end_box = localize_on_original(box, h, w, h_count, w_count, img_h, img_w, new_h, new_w)
                info = [img_name, "white", img_h, img_w, h_count, w_count, h, w, white, str_box, end_box]
                all_info.append(info)
                if original_with_localize_flag:
                    img_drow.rectangle([str_box, end_box], outline=white_box_color, width=thickness)
            for point in points:
                box = points[point]

                str_box, end_box = localize_on_original(box, h, w, h_count, w_count, img_h, img_w, new_h, new_w, True)
                info = [img_name, "point", img_h, img_w, h_count, w_count, h, w, point, str_box, end_box]
                all_info.append(info)
                if original_with_localize_flag:
                    img_drow.ellipse([str_box[0], str_box[1], str_box[0] + 3, str_box[1] + 3], outline=scatter_color)
        num = len(all_info)
        if original_with_localize_flag:
            img.save(path + "/" + file_name + ".png")
    if len(all_info) > 0:
        df = pd.DataFrame(all_info)
        csv_file_path = path + file_name + ".csv"
        df.to_csv(csv_file_path, index=False, header=header)
        Pass_Date = merge_cont_boxes(img_name, s3_prefix, csv_file_path, file_name, "black", path,No_Error_Flag)
        Pass_Date = merge_cont_boxes(img_name, s3_prefix, csv_file_path, file_name, "white", path, No_Error_Flag)
        Pass_Date = merge_cont_boxes(img_name, s3_prefix, csv_file_path, file_name, "point", path, No_Error_Flag)

    return path + "/" + file_name + ".png", num, Pass_Date



def merge_cont_boxes(img_name,s3_prefix,csv_file, file_name, error_type,output_local_path_origin,No_Error_Flag):
    df = pd.read_csv(csv_file)
    header = list(df)
    header.append("start time frame")
    header.append("end time frame")
    header.append("s3_path")
    header.append("original_img")
    header.append("sat_name")
    header.append("local_folder_name")
    header.append("Pass_Date")
    header.append("has_error")
    if "NOAA18" in s3_prefix:
        sat_name = "NOAA18"
    elif "NOAA19" in s3_prefix:
        sat_name = "NOAA19"
    elif "METOPB" in s3_prefix:
        sat_name = "METOPB"
    elif "METOPC" in s3_prefix:
        sat_name = "METOPC"
    elif "test" in s3_prefix:
        sat_name = "TEST"
    all_info = []
    if error_type in ["point", "white"]:
        df = df[df["error type"] == error_type]
        data = df.values.tolist()
        for row in data:
            str_box = tuple(map(int, row[-2][1:-1].split(', ')))
            end_box = tuple(map(int, row[-1][1:-1].split(', ')))
            str_time_frame = from_pixl_to_time_frame(file_name, str_box)
            str_time_frame = str(str_time_frame)
            end_time_frame = from_pixl_to_time_frame(file_name, end_box)
            end_time_frame = str(end_time_frame)
            row.append(str_time_frame)
            row.append(end_time_frame)
            row.append(s3_prefix)
            row.append(img_name)
            row.append(sat_name)
            row.append(output_local_path_origin.replace("localized/",""))
            tmp = from_pixl_to_time_frame(file_name, (0, 0))
            row.append(str(tmp))
            row.append(not No_Error_Flag)
            all_info.append(row)


    else:
        all_h = list(set(df["h"]))
        all_n = list(set(df["number"]))
        for h in all_h:
            for n in all_n:
                new_df = df[(df["h"] == h) & (df["number"] == n) & (df["error type"] == error_type)]
                row = new_df[new_df.w == new_df.w.min()].values.tolist()
                if row:
                    str_box = row[0][-2][1:-1]
                    str_box = tuple(map(int, str_box.split(', ')))
                    row = new_df[new_df.w == new_df.w.max()].values.tolist()
                    end_box = row[0][-1][1:-1]
                    end_box = tuple(map(int, end_box.split(', ')))
                    l = row[0][0:-2]
                    l.append(str_box)
                    l.append(end_box)
                    str_time_frame = from_pixl_to_time_frame(file_name, str_box)
                    str_time_frame = str(str_time_frame)
                    end_time_frame = from_pixl_to_time_frame(file_name, end_box)
                    end_time_frame = str(end_time_frame)
                    l.append(str_time_frame)
                    l.append(end_time_frame)
                    l.append(s3_prefix)
                    l.append(img_name)
                    l.append(sat_name)
                    l.append(output_local_path_origin.replace("localized/",""))
                    tmp2 = from_pixl_to_time_frame(file_name, (0, 0))
                    l.append(str(tmp2))
                    l.append(not No_Error_Flag)

                    all_info.append(l)
    if len(all_info) == 0:
        l = ["NA" for i in range(len(header))]
        all_info.append(l)
    df = pd.DataFrame(all_info)
    df.to_csv(csv_file.split(".csv")[0] + "_" + error_type + ".csv", index=False, header=header,
              date_format='%Y-%m-%d %H:%M:%S')

    return from_pixl_to_time_frame(file_name, (0, 0))











######################################################################################################################################################
# correlation
######################################################################################################################################################




def get_correlated_rf_events(db_host, db_user_name, db_password, db_name, Img_Date_st, Img_Date_end, delta ):

    # Add one day and 12 hours
    new_date = Img_Date_end + timedelta(days=delta["Day"], hours=delta["Hour"], minutes=delta["Mint"])

    # Convert the result back to ISO format
    new_iso_end_date = new_date.isoformat()

    new_date = Img_Date_st - timedelta(days=delta["Day"], hours=delta["Hour"], minutes=delta["Mint"])

    new_iso_start_date = new_date.isoformat()

    cnx = pymysql.connect(host=db_host,
                          user=db_user_name,
                          passwd=db_password,
                          db=db_name,
                          charset='utf8mb4',
                          connect_timeout=20,
                          cursorclass=pymysql.cursors.DictCursor)

    # Execute a query
    cursor = cnx.cursor()

    query = "SELECT * FROM rf_events WHERE timestamp > %s and timestamp < %s"
    cursor.execute(query, (new_iso_start_date, new_iso_end_date))
    result = cursor.fetchall()

    return result








def correlate(img_time_st, img_time_end, RF_event_time, delta):
    new_date = img_time_end + timedelta(days=delta["Day"], hours=delta["Hour"], minutes=delta["Mint"])

    # Convert the result back to ISO format
    new_iso_end_date = new_date.isoformat()

    new_date = img_time_st - timedelta(days=delta["Day"], hours=delta["Hour"], minutes=delta["Mint"])

    new_iso_start_date = new_date.isoformat()


    return new_iso_start_date < RF_event_time and new_iso_end_date >= RF_event_time






def merge_ML_RF_events(ML_record,RF_record,Pass_ID,station, Error_Source, RF_event_type="RFMS"):
    record = [Pass_ID]
    [record.append(r) for r in ML_record]
    [record.append(r) for r in RF_record]

    record.append(Error_Source)
    record.append(station)
    record.append(RF_event_type)

    return record


def correlate_ml_rf(ML_records, RF_records, delta):
    Pass_ID = str(uuid.uuid4())
    total_records = []
    sat_name = ML_records[0]["sat_name"]
    station = ML_records[0]["s3_path"].split("stand_alone/")[1].split("/")[0]
    for ML_record in ML_records:
        err_st = ML_record["error_start_time"]
        err_end = ML_record["error_end_time"]
        ML_record = list(ML_record.values())
        ML_record.pop(0)
        if len(RF_records) > 0:
            for i, RF_record in enumerate(RF_records):
                rf_time = RF_record["timestamp"]
                if correlate(err_st, err_end, rf_time, delta):
                    RF_record = list(RF_record.values())
                    RF_record.pop(0)
                    record = merge_ML_RF_events(ML_record, RF_record, Pass_ID, station, "Both")
                    total_records.append(tuple(record))
                    RF_records.pop(i)
                else:
                    RF_record = [[].append(None) for i in range(31)]
                    record = merge_ML_RF_events(ML_record, RF_record, Pass_ID, station, "Img_Dgrd", None)
                    total_records.append(tuple(record))
        else:
            RF_record = [[].append(None) for i in range(31)]
            record = merge_ML_RF_events(ML_record, RF_record, Pass_ID, station, "Img_Dgrd", None)
            total_records.append(tuple(record))

    if len(RF_records) > 0:

        for RF_record1 in RF_records:
            RF_record1 = list(RF_record1.values())
            RF_record1.pop(0)
            ML_record = [[].append(None) for i in range(19)]
            ML_record[-4] = sat_name
            record = merge_ML_RF_events(ML_record, RF_record1, Pass_ID, station, "RF_evt")
            total_records.append(tuple(record))


    return total_records



######################################################################################################################################################
# Database
######################################################################################################################################################



def excute_query(db_host, db_user_name, db_password, db_name, query, cond = None):
    cnx = pymysql.connect(host=db_host,
                          user=db_user_name,
                          passwd=db_password,
                          db=db_name,
                          charset='utf8mb4',
                          connect_timeout=20,
                          cursorclass=pymysql.cursors.DictCursor)

    # Execute a query
    cursor = cnx.cursor()
    if cond:
        cursor.execute(query, cond)
    else:
        cursor.execute(query)

    cnx.commit()
    cnx.close()
    result = cursor.fetchall()

    return result



def write_aquired_RF_event_records_v2(db_host, db_user_name, db_password, db_name, image_name):

    query = "SELECT * FROM ml_localization WHERE image_name = %s"
    Img_results = excute_query(db_host, db_user_name, db_password, db_name, query, cond=(image_name,))
    if len(Img_results) > 0:
        Img_Date_end = from_pixl_to_time_frame(image_name, (0, Img_results[0]["pic_size_h_pix"]))
        Img_Date_st = from_pixl_to_time_frame(image_name, (0, 0))
        delta = {"Day" : 0, "Hour" : 0, "Mint" : 0.01}

        RF_results = get_correlated_rf_events(db_host, db_user_name, db_password, db_name, Img_Date_st, Img_Date_end, delta )

        print("length of image results ", len(Img_results))
        print("length of RF results ", len(RF_results))
        total_records = correlate_ml_rf(Img_results, RF_results, delta)

        query = """INSERT INTO ml_localization_rf_events (Pass_ID,image_name,error_type,pic_size_h_pix,pic_size_w_pix,sub_img_count_h,sub_img_count_w,
                                                                        sub_img_loc_h,sub_img_loc_w,num_errors_raw,sub_img_error_start_pix,sub_img_error_end_pix,error_start_time,error_end_time,
                                                                        s3_path,original_img,sat_name,local_folder_name,Pass_Date,has_error,PCI,_id,beam,carrierID,cellID,eNodeB,elevationAngle,
                                                                        elevationAngleUnits,eventID,headingAzimuth,headingAzimuthUnits,inverseAxialRatio,labels,locationLat,locationLatUnits,locationLon,
                                                                        locationLonUnits,maxBandwidth,maxBandwidthUnits,maxFrequency,maxFrequencyUnits,maxPower,maxPowerUnits,mode,notifyCarrier,
                                                                        remoteID,severityLevel,signalType,tiltAngle,tiltAngleUnits,time_stamp,Error_Source,station,RF_event_type)
                                                                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                                                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                                                                    """



        for rec in total_records:
            excute_query(db_host, db_user_name, db_password, db_name, query, cond = rec)


        print("records for image {} are added".format(image_name))





def write_to_db_from_csv(db_host, db_user_name, db_password, db_name, bucket_name, path_on_S3, save_path):
    file_name = path_on_S3.split("/")[-1]
    print('Starting writing to database for ' + file_name)
    try:
        connection = pymysql.connect(host=db_host,
                                     user=db_user_name,
                                     passwd=db_password,
                                     db=db_name,
                                     charset='utf8mb4',
                                     connect_timeout=20,
                                     cursorclass=pymysql.cursors.DictCursor)

        if connection.open:

            cursor_obj = connection.cursor()

            record = """INSERT INTO ml_localization (image_name,error_type,pic_size_h_pix,pic_size_w_pix,sub_img_count_h,sub_img_count_w,
                				sub_img_loc_h,sub_img_loc_w,num_errors_raw,sub_img_error_start_pix,sub_img_error_end_pix,error_start_time,error_end_time,
                				s3_path,original_img,sat_name,local_folder_name,Pass_Date,has_error)
                                          VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                              """

            csv_path = load_image_and_save_local(bucket_name, path_on_S3, save_path, file_name)

            df = pd.read_csv(csv_path).dropna()
            recs = df.values.tolist()
            for rec in recs:
                record_data = tuple(rec)
                cursor_obj.execute(record, record_data)

            connection.commit()
            connection.close()
            print('Writing to database ended successfully')
    except pymysql.MySQLError as e:
        print("ERROR: Unexpected error: osama.")
        print(e)
        sys.exit()



##########################################################################################################################################################
# HRPT Decoder
##########################################################################################################################################################


def hrpt_decode(executable_path,in_path,out_path):
    # Replace this with the path to your executable
    # in_path = "/home/ubuntu/project/HRPT-Decoder/" + in_path
    # out_path = "/home/ubuntu/project/HRPT-Decoder/" + out_path

    # Replace this with any command-line arguments you need to pass to your executable
    args = ["-o", out_path, "-i", in_path, "-t", "NOAA", "-c", "5", "-e", "150", "--ignore_rest"]

    # Run the executable and capture its output
    result = subprocess.run([executable_path] + args, capture_output=True, text=True)

    # Print the output of the executable
    print(result.stdout)

    # args = " -o " +  out_path + " -i " +  in_path + " -t  NOAA  -c  5 -e 150 --ignore_rest"
    #
    # os.system(executable_path + args)



def main(event):

    font_path_in_s3 = event["font"]

    image_path_on_S3 = event["img_path"]
    output_local_path_origin = "project/img_class_loc/" + ("/".join(image_path_on_S3.split("/")[0:-1]) + "/").replace("imported_images","processed_images")
    os.makedirs(output_local_path_origin, exist_ok=True)


    now = str(datetime.datetime.now()).split(".")[0].replace(" ", "-")
    img_name = "_".join(image_path_on_S3.split("/")[-1].split(".")[0:-1]) + "_" + now
    tmp = image_path_on_S3.split("/")[-1].split(".")
    img_name_with_extention = "_".join(tmp[0:-1]) + "_" + now + "." + tmp[-1]
    output_local_path = output_local_path_origin + img_name + "/"
    if not os.path.isdir(output_local_path):
        os.mkdir(output_local_path)

    output_local_path_classified = output_local_path + "classified/"
    if not os.path.isdir(output_local_path_classified):
        os.mkdir(output_local_path_classified)

    output_local_path_localized = output_local_path + "localized/"
    if not os.path.isdir(output_local_path_localized):
        os.mkdir(output_local_path_localized)

    output_local_path_localized_processed = output_local_path + "localized_processed/"
    if not os.path.isdir(output_local_path_localized_processed):
        os.mkdir(output_local_path_localized_processed)

    bucket_name = event["bucket_name"]
    font_local_path = load_image_and_save_local(bucket_name, font_path_in_s3, output_local_path_origin,
                                                font_path_in_s3.split("/")[-1])

    db_host = event["db_host"]
    db_user_name = event["db_user_name"]
    db_password = event["db_password"]
    db_name = event["db_name"]

    s3_prefix = "/".join(event["img_path"].split("/")[0:3]) + "/" + "operation/processed_images/" + img_name
    ENDPOINT_NAME = event["ENDPOINT_NAME"]
    partitioned_prefix = s3_prefix + "/partitioned/"
    classified_prefix = s3_prefix + "/classified/"

    localized_prefix = s3_prefix + "/localized/"
    localized_processed_prefix = s3_prefix + "/localized_processed/"
    db_prefix = s3_prefix + "/db/"

    partitioned_local_path, uploaded_images, info = main_partition_fn(bucket_name, image_path_on_S3, output_local_path,
                                                                      s3_prefix, font_local_path,img_name_with_extention)
    total_images = []
    classified_success = 0
    if "error" not in partitioned_local_path and uploaded_images > 0:
        partitioning_message = "partitioning images done successfully"
        files_list = list_s3_files(bucket_name, partitioned_prefix)

        for file in files_list:
            if ".png" in file:
                img_url = [file]
                img_class = classify_fn(ENDPOINT_NAME, img_url)
                if img_class[0] != "fail":
                    # img_url[0]["img_class"] = img_class[0]
                    s3_save_path = classified_prefix + str(img_class) + "/" + file.split("/")[-1]
                    copy_source = {'Bucket': bucket_name, 'Key': file.replace('s3://' + bucket_name + "/", "")}
                    copy_flag = copy_from_s3_to_s3(copy_source, bucket_name, s3_save_path)
                    total_images.append(img_url)
                    if copy_flag != "fail":
                        classified_success += 1

        if classified_success > 0:
            partitioning_message += " and Classification done successfully"
            Pass_Date = localize(classified_prefix, output_local_path, info, bucket_name, localized_prefix,
                     localized_processed_prefix, font_local_path, s3_prefix)

            files = list_s3_files(bucket_name, db_prefix)

            if len(files) > 0:
                partitioning_message += " and Localization done successfully"
                for file in files:
                    file = file.replace("s3://" + bucket_name + "/", "")
                    write_to_db_from_csv(db_host, db_user_name, db_password, db_name, bucket_name, file,output_local_path)


                write_aquired_RF_event_records_v2(db_host, db_user_name, db_password, db_name, img_name)


            else:
                partitioning_message += " and error in Localization"
        else:
            partitioning_message += " and error in Classification"


    elif "error" in partitioned_local_path:
        partitioning_message = partitioned_local_path
    else:
        partitioning_message = 'No image has been successfully uploaded'

    final_message = str(len(total_images)) + " partitioned successfully and " + str(
        classified_success) + " classified successfully"


    # shutil.rmtree(output_local_path_origin)

    print(partitioning_message)
    return {
        "out": final_message,
        "partitioning message": partitioning_message
    }





###############################################################################################
# for mass run
###############################################################################################

# event = {
#   "img_path": "stand_alone/AOML/sat_name/operation/imported_images/level0/",
#   "bucket_name": "rfims-ml-addson",
#   # "S3_Prefix" : "stand_alone/AOML/NOAA18/",
#   "ENDPOINT_NAME": "rfims-ml-addson-ml-t2-medium",
#   "font": "operation/font/Arial.ttf",
#   "db_host": "localhost",
#   "db_user_name": "sms",
#   "db_password": "ABCD1234",
#   "db_name": "stand_alone"
# }
#
#
#
#
#
#
# images = {"NOAA18" : ["20210519161112n01.gif","20210603144925n19.gif","20211003135638n15.gif","20211004133150n15.gif"]
#          ,"NOAA19" : ["20211010034729n15.gif","20211010050557n03.gif","20211010151958n18.gif","20211013144333n03.gif"]}
#
#
# for key in images:
#     for img in images[key]:
#         new_event = event.copy()
#         new_event["img_path"] = event["img_path"].replace("sat_name", key) + img
#         res = main(new_event)
#         print("*" * 100)
#         print("*" * 100)
#         print("*" * 100)
#         print("*" * 100)




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


    input_Img_path = sys.argv[1].split(event["bucket_name"])[-1][1:]

    if ".hrpt" in input_Img_path:

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
        event["img_path"] = input_Img_path


    res = main(event)


