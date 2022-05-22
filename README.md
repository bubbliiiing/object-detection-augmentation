## object-detection-augmentation-这里面存放了一些目标检测算法的数据增强方法。如mosaic、mixup。
---

## 目录
1. [数据增强测试](#数据增强测试)
2. [生成图片与标签](#生成图片与标签)

## 数据增强测试
以test开头的几个py文件用于测试不同的数据增强方法。
### 测试步骤
1、Origin_VOCdevkit_path用于指定VOC数据集所在的文件夹；       
2、input_shape代表数据增强后的图片的大小；     
3、运行test_*.py即可查看对应的数据增强效果。     

## 标签处理
以generate开头的几个py文件用于生成并保存数据增强后的标签与图片。
### 生成步骤
1、Origin_VOCdevkit_path用于指定需要增强的数据集路径；     
2、Out_VOCdevkit_path用于指定输出的数据集路径；     
3、Out_Num用于增强生成多少张图片；    
4、input_shape代表数据增强后的图片的大小；       
5、运行generate_*.py即可生成并保存数据增强后的标签与图片。