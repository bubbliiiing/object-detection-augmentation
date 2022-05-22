import os
from random import sample

import numpy as np
from PIL import Image, ImageDraw

from utils.random_data import get_random_data, get_random_data_with_Mosaic
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
    #   获取4个图像与标签
    #------------------------------#
    sample_xmls     = sample(xml_names, 4)
    unique_labels   = get_classes(sample_xmls, Origin_Annotations_path)
    
    annotation_line = []
    for xml in sample_xmls:
        line = convert_annotation(os.path.join(Origin_JPEGImages_path, os.path.splitext(xml)[0] + '.jpg'), os.path.join(Origin_Annotations_path, xml), unique_labels)
        annotation_line.append(line)

    #------------------------------#
    #   合并mosaic
    #------------------------------#
    image_data, box_data = get_random_data_with_Mosaic(annotation_line, input_shape)
    
    img = Image.fromarray(image_data.astype(np.uint8))
    for j in range(len(box_data)):
        thickness = 3
        left, top, right, bottom  = box_data[j][0:4]
        draw = ImageDraw.Draw(img)
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i],outline=(255, 255, 255))
    img.show()
