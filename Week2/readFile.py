import os
import pandas as pd
from sklearn.utils import shuffle

label_list = []
image_list = []

image_dir = ""#绝对路径
label_dir = ""

for s1 in os.listdir(image_dir):
    image_sub_dir1 = os.path.join(image_dir, s1)
    label_sub_dir1 = os.path.join(label_dir,'Label_' + str.lower(s1), 'Label')
    for s2 in os.listdir(image_sub_dir1):
        image_sub_dir2 = os.path.join(image_sub_dir1, s2)
        label_sub_dir2 = os.path.join(label_sub_dir1, s2)
        for s3 in os.listdir(image_sub_dir2):
            image_sub_dir3 = os.path.join(image_sub_dir2, s3)
            label_sub_dir3 = os.path.join(label_sub_dir2, s3)
            for s4 in os.listdir(image_sub_dir3):
                sff = s4.replace('.jpg', '_bin.png')
                image_sub_dir4 = os.path.join(image_sub_dir3,s4)
                label_sub_dir4 = os.path.join(label_sub_dir3,s4)

                if not os.path.exits(image_sub_dir4):
                    continue
                if not os.path.exits(label_sub_dir4):
                    continue

                label_list.append(label_sub_dir4)
                image_list.append(image_sub_dir4)

assert len(image_list) == len(label_list)

total_length = len(image_list)
sixth_part = int(total_length*0.6)
eight_part = int(total_length*0.8)

all = pd.DataFrame({'image':image_list,'label':label_list})
all_shuffle = shuffle(all)

train_dataset = all_shuffle[:sixth_part]
val_dataset = allshuffle[sixth_part:eigth_part]
test_dataset= all_shuffle[eigth_part:]


train_dataset.to_csv()
val_dataset.to_csv()
test_dataset.to_csv()

 



