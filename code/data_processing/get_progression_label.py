import pandas as pd
import shutil
import numpy as np
import pdb
from tqdm import tqdm
import os
import json

#The splited dataset, train, test, and val are used to create folders A, B, label and the train, val, test text files.
df_train = pd.read_csv('~/data/split/train.csv', header=0)
df_val = pd.read_csv('~/data/split/val.csv', header=0)
df_test = pd.read_csv('~/data/split/test.csv', header=0)

# concat all the dataframes
df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

disease_labels = [
                'lung opacity',
                'pleural effusion',
                'atelectasis',
                'enlarged cardiac silhouette',
                'pulmonary edema/hazy opacity',
                'pneumothorax', 
                'consolidation',
                'pneumonia']


filtered_df = df_all[df_all['label_name'].isin(disease_labels)]
df = filtered_df[['current_image_id', 'previous_image_id', 'label_name', 'comparison']]


#Only keep one label for one pair of data, reduce the effect of bbox and disease
df = df.drop_duplicates(
  subset = ['current_image_id', 'previous_image_id'],
  keep = 'first').reset_index(drop = True)

df1 = pd.read_csv('~/cxr-record-all-jpg.csv', sep=",", header=0)
df1 = df1[['dicom_id', 'path']]

progression_label_dict = {}
df = df[:10]

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    current_img_id = row['current_image_id']
    previous_img_id = row['previous_image_id']
    comparsion = row['comparison']
    label_name = row['label_name']

    
    progression_label = np.full([len(disease_labels)], np.nan)
    index = disease_labels.index(label_name)
    label_to_int = {'worsened': 0, 'improved': 1, 'no change': 2}
    label = label_to_int[comparsion]
    progression_label[index] = int(label)

    current_img_path = df1.loc[df1['dicom_id']==current_img_id]['path'].values.tolist()[0].strip()
    previous_img_path = df1.loc[df1['dicom_id']==previous_img_id]['path'].values.tolist()[0].strip()

    #sub_id_label[current_img_sub_id].append(label_name)
    
    current_img_name = current_img_path.split("/")[-1]
    previous_img_name = previous_img_path.split("/")[-1]
    
    current_img_name = current_img_path.split("/")[-1]
    previous_img_name = previous_img_path.split("/")[-1]
    # if current_img_name in not_exist_img_list or previous_img_name in not_exist_img_list:
    #     continue
    new_img_name = current_img_name.split(".")[0] + "_" + previous_img_name.split(".")[0]

    # with open(label_path, "w") as f: #write the comparison result to Labels folder
    #     f.write(comparsion)
    progression_label_dict[new_img_name] = progression_label.tolist()

with open('progression_label.json', 'w') as f:
    json.dump(progression_label_dict, f)

# dfc.to_csv('current_sub_id_study_id.csv')
# dfp.to_csv('previous_sub_id_study_id.csv')