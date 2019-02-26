import skimage.io as skio
import cv2 as cv
import matplotlib.pyplot as plt
import os
import pandas as pd

def yolo_to_annotation(size,boxyolo):
    yw = boxyolo[3]
    yh = boxyolo[4]
    w = yw*size[0]
    h = yh*size[1]
    centerx=boxyolo[1]*size[0]
    centery=boxyolo[2]*size[1]
    left_coord_x = int(centerx - (w/2))
    left_coord_y = int(centery - (h/2))
    right_coord_x = int(centerx + (w/2))
    right_coord_y = int(centery + (h/2))
    return (left_coord_x,left_coord_y),(right_coord_x,right_coord_y)

def annotation_to_yolo(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def get_filenames(image_filename, input_label_dir, input_image_dir):
    filename_master = image_filename[:-4]
    filename_txt = input_label_dir + filename_master +'.txt'
    filename_jpg =  input_image_dir + image_filename
    return filename_master, filename_txt, filename_jpg

def read_image_properties(image_path, text_path):
    img = skio.imread(image_path)
    bb = pd.read_csv(text_path, sep= ' ',index_col=None, header=None)
    y,x,z = img.shape
    size = [x,y]
    return img,bb,size

def draw_bbox_to_image(filename, input_label_dir):
    filename_master, filename_txt, filename_jpg = get_filenames(filename, input_label_dir, input_image_dir)
    image,bb,size = read_image_properties(filename_jpg, filename_txt)
    left_coord,right_coord = yolo_to_annotation(size, bb)
    image = cv.rectangle(image,left_coord, right_coord,(255,100,100),15)
    
    return image

def save_image(filename, output_image_dir, image):
    path = output_image_dir + filename
    skio.imsave(path, image)

def process_rule(input_image_dir, input_label_dir, output_image_dir):
    for filename in os.listdir(input_image_dir):
        try:
            image = draw_bbox_to_image(filename, input_label_dir)
            
            save_image(filename, output_image_dir,image)
        except:
            print('file open error, ', filename)

if __name__ == "__main__":

    input_image_dir ="/home/nodeflux/Nodeflux/Dataset/Gun/target_file/knife/"
    input_label_dir ="/home/nodeflux/Nodeflux/Dataset/Gun/label_file/knife/"
    output_image_dir ="/home/nodeflux/Nodeflux/Dataset/Gun/target_file/knife_bbox/"

    process_rule(input_image_dir,input_label_dir, output_image_dir)