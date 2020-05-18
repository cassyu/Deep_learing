import os
import pandas as pd
from sklearn.utils import shuffle

label_list = []
image_list = []

image_dir = ""#绝对路径
label_dir = ""
savepath = ""

for road in os.listdir(image_dir):
    for record in os.listdir(os.path.join(image_dir,road)):
        for camera in os.listdir(os.path.join(image_dir,road,record)):
            for image_name in os.listdir(os.path.join(image_dir,road,record,camera)):
                label_image_name = image_name.replace('.jpg,','_bin.png')
                image_path = os.path.join(image_dir,road,record,camera,image_name)
                label_path = os.path.join(image_dir,'Label_'+str.lower(road),'Label',record,camera,label_image_name)
                image_list.append(image_path)
                label_list.append(label_path)

csv_file = pd.DataFrame({'image':image_list,'label':label_list})
csv_file = shuffle(csv_file)
csv_file.to_csv(os.path.join(savepath,'train.csv'),index=False)
