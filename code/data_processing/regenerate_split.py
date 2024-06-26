# ignore the case that with all 'nan' label value

import pandas as pd
import numpy as np
import pdb
from tqdm import tqdm

df = pd.read_csv('~/data/symptom_label_chexpert.csv', header=0)
train_list = []
val_list = []
test_list = []

all_nan_list = []
for i in tqdm(range(0, len(df), 2)):
    df_a = df.iloc[i]
    df_b = df.iloc[i+1]
    dicom_id_a = df_a['dicom_id']
    dicom_id_b = df_b['dicom_id']
    split = df_a['split']
    
    values_a = np.array(df_a[['Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia']].values.tolist())
    values_b = np.array(df_b[['Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia']].values.tolist())
    if np.isnan(values_a).all() or np.isnan(values_b).all():
        name = dicom_id_a + '_' + dicom_id_b + '.jpg'
        all_nan_list.append(name)
    else:
        name = dicom_id_a + '_' + dicom_id_b + '.jpg'
        if split == 'train':
            train_list.append(name)
        elif split == 'val':
            val_list.append(name)
        else:
            test_list.append(name)

with open('~/data/my_list/train.txt', 'w') as f:
    for case in train_list:
        f.write(case + '\n')
f.close()

with open('~/data/my_list/val.txt', 'w') as f:
    for case in val_list:
        f.write(case + '\n')
f.close()

with open('~/data/my_list/test.txt', 'w') as f:
    for case in test_list:
        f.write(case + '\n')
f.close()
        
