import os
from random import sample

import numpy as np
from PIL import Image, ImageDraw

from utils.random_data import get_random_data, get_random_data_with_MixUp
from utils.utils import convert_annotation, get_classes

#-----------------------------------------------------------------------------------#
#   Origin_VOCdevkit_path   原始数据集所在的路径
#-----------------------------------------------------------------------------------#
Origin_VOCdevkit_path   = "VOCdevkit_Origin"
#-----------------------------------------------------------------------------------#
#   input_shape             生成的图片大小。
#-----------------------------------------------------------------------------------#
input_shape             = [640, 640]

if __name__ == "__main__":
    Origin_JPEGImages_path  = os.path.join(Origin_VOCdevkit_path, "VOC2007/JPEGImages")
    Origin_Annotations_path = os.path.join(Origin_VOCdevkit_path, "VOC2007/Annotations")
    
    #---------------------------#
    #   遍历标签并赋值
    #---------------------------#
    xml_names = os.listdir(Origin_Annotations_path)

    #------------------------------#
    #   获取两个图像与标签
    #------------------------------#
    sample_xmls     = sample(xml_names, 2)
    unique_labels   = get_classes(sample_xmls, Origin_Annotations_path)
    jpg_name_1  = os.path.join(Origin_JPEGImages_path, os.path.splitext(sample_xmls[0])[0] + '.jpg')
    jpg_name_2  = os.path.join(Origin_JPEGImages_path, os.path.splitext(sample_xmls[1])[0] + '.jpg')
    xml_name_1  = os.path.join(Origin_Annotations_path, sample_xmls[0])
    xml_name_2  = os.path.join(Origin_Annotations_path, sample_xmls[1])
    
    line_1 = convert_annotation(jpg_name_1, xml_name_1, unique_labels)
    line_2 = convert_annotation(jpg_name_2, xml_name_2, unique_labels)

    #------------------------------#
    #   各自数据增强
    #------------------------------#
    image_1, box_1  = get_random_data(line_1, input_shape) 
    image_2, box_2  = get_random_data(line_2, input_shape) 
    
    #------------------------------#
    #   合并mixup
    #------------------------------#
    image_data, box_data = get_random_data_with_MixUp(image_1, box_1, image_2, box_2)
    
    img = Image.fromarray(image_data.astype(np.uint8))
    for j in range(len(box_data)):
        thickness = 3
        left, top, right, bottom  = box_data[j][0:4]
        draw = ImageDraw.Draw(img)
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i],outline=(255, 255, 255))
    img.show()
