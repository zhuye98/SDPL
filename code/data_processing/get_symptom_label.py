import pandas as pd
import numpy as np
import pdb
from tqdm import tqdm
import os
import json

disease_labels = {
                        'lung opacity',                   # checked
                        'pleural effusion',               # checked
                        'atelectasis',                    # checked
                        'enlarged cardiac silhouette',    # checked: Cardiomegaly
                        'pulmonary edema/hazy opacity',   # checked: Edema
                        'pneumothorax',                   # checked
                        'consolidation',                  # checked
                        # 'fluid overload/heart failure',   # not in mimic-jpg-2.0.0-cheXpert, 
                        'pneumonia'                       # checked
                    }

df_train = pd.read_csv('~/data/split/train.csv', header=0)
df_val = pd.read_csv('~/data/split/val.csv', header=0)
df_test = pd.read_csv('~/data/split/test.csv', header=0)

# file can either be chexpert or negbio
df_cxr_symptoms = pd.read_csv('~/mimic-cxr-2.0.0-chexpert.csv', header=0)
df_cxr_symptoms = df_cxr_symptoms[['study_id', 'Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 'Edema',
                                   'Pneumothorax', 'Consolidation', 'Pneumonia', 'No Finding']]

df_cxr_record = pd.read_csv('~/cxr-record-all-jpg.csv', sep=",", header=0)
df_cxr_record = df_cxr_record[['subject_id', 'study_id', 'dicom_id', 'path']]

filtered_df_train = df_train[df_train['label_name'].isin(disease_labels)]
filtered_df_val = df_val[df_val['label_name'].isin(disease_labels)]
filtered_df_test = df_test[df_test['label_name'].isin(disease_labels)]

df_tosave = pd.DataFrame(columns=['subject_id', 'study_id', 'dicom_id', 'path', 
                                  'Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 
                                  'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia', 'split'])
subject_ids = []
not_ava_study_ids = []
count = 0
for idx, row in tqdm(filtered_df_train.iterrows(), total=filtered_df_train.shape[0]):
    # check all the subject_id
    current_img_id = row['current_image_id']
    previous_img_id = row['previous_image_id']
    
    # _cur as 'current', _pre as 'previous'
    data_cur = df_cxr_record.loc[df_cxr_record['dicom_id']==current_img_id][['subject_id', 'study_id', 'dicom_id', 'path']].values.tolist()[0]
    data_pre = df_cxr_record.loc[df_cxr_record['dicom_id']==previous_img_id][['subject_id', 'study_id', 'dicom_id', 'path']].values.tolist()[0]

    data_cur_path = data_cur[-1]
    data_pre_path = data_pre[-1]
    
    data_cur[-1] = '~/data/A/'+ data_cur_path.split('/')[-1]
    data_pre[-1] = '~/data/B/'+ data_pre_path.split('/')[-1]

    study_id_cur = data_cur[1]
    study_id_pre = data_pre[1]

    # check corresponding symptoms
    try:
        cxr_symptoms_cur = df_cxr_symptoms.loc[df_cxr_symptoms['study_id']==study_id_cur][['Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia', 'No Finding']].values.tolist()[0]
        if cxr_symptoms_cur[-1] == 1:
            cxr_symptoms_cur = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            cxr_symptoms_cur = cxr_symptoms_cur[:-1]
        cxr_symptoms_pre = df_cxr_symptoms.loc[df_cxr_symptoms['study_id']==study_id_pre][['Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia', 'No Finding']].values.tolist()[0]
        if cxr_symptoms_pre[-1] == 1:
            cxr_symptoms_pre = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            cxr_symptoms_pre = cxr_symptoms_pre[:-1]
    except IndexError:
        not_ava_study_ids.append(study_id_cur)
        continue

    # replace the '-1' with 1 in the lists (cxr_symptoms_cur and cxr_symptoms_pre)
    cxr_symptoms_cur = [1.0 if x == -1 else x for x in cxr_symptoms_cur]
    cxr_symptoms_pre = [1.0 if x == -1 else x for x in cxr_symptoms_pre]
    
    data_cur.extend(cxr_symptoms_cur)
    data_pre.extend(cxr_symptoms_pre)
    data_cur.append('train')
    data_pre.append('train')
    df_tosave = pd.concat([df_tosave, pd.DataFrame(np.array(data_cur).reshape(1, 13), columns=['subject_id', 'study_id', 'dicom_id', 'path', 
                                  'Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 
                                  'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia', 'split'])], ignore_index=True)
    df_tosave = pd.concat([df_tosave, pd.DataFrame(np.array(data_pre).reshape(1, 13), columns=['subject_id', 'study_id', 'dicom_id', 'path', 
                                  'Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 
                                  'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia', 'split'])], ignore_index=True)
    
df_tosave.to_csv('~/data/symptom_label_chexpert.csv', index=None)

df_tosave = pd.DataFrame(columns=['subject_id', 'study_id', 'dicom_id', 'path', 
                                  'Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 
                                  'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia', 'split'])


for idx, row in tqdm(filtered_df_val.iterrows(), total=filtered_df_val.shape[0]):
    # check all the subject_id
    current_img_id = row['current_image_id']
    previous_img_id = row['previous_image_id']

    data_cur = df_cxr_record.loc[df_cxr_record['dicom_id']==current_img_id][['subject_id', 'study_id', 'dicom_id', 'path']].values.tolist()[0]
    data_pre = df_cxr_record.loc[df_cxr_record['dicom_id']==previous_img_id][['subject_id', 'study_id', 'dicom_id', 'path']].values.tolist()[0]

    data_cur_path = '~/data/A/'+data_cur[-1]
    data_pre_path = '~/data/B/'+data_pre[-1]
    
    data_cur[-1] = data_cur_path.split('/')[-1]
    data_pre[-1] = data_pre_path.split('/')[-1]

    study_id_cur = data_cur[1]
    study_id_pre = data_pre[1]

    # check corresponding symptoms
    try:
        cxr_symptoms_cur = df_cxr_symptoms.loc[df_cxr_symptoms['study_id']==study_id_cur][['Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia', 'No Finding']].values.tolist()[0]
        if cxr_symptoms_cur[-1] == 1:
            cxr_symptoms_cur = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            cxr_symptoms_cur = cxr_symptoms_cur[:-1]
        cxr_symptoms_pre = df_cxr_symptoms.loc[df_cxr_symptoms['study_id']==study_id_pre][['Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia', 'No Finding']].values.tolist()[0]
        if cxr_symptoms_pre[-1] == 1:
            cxr_symptoms_pre = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            cxr_symptoms_pre = cxr_symptoms_pre[:-1]
    except IndexError:
        not_ava_study_ids.append(study_id_cur)
        continue
    
    data_cur.extend(cxr_symptoms_cur)
    data_pre.extend(cxr_symptoms_pre)
    data_cur.append('val')
    data_pre.append('val')

    df_tosave = pd.concat([df_tosave, pd.DataFrame(np.array(data_cur).reshape(1, 13), columns=['subject_id', 'study_id', 'dicom_id', 'path', 
                                  'Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 
                                  'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia', 'split'])], ignore_index=True)
    df_tosave = pd.concat([df_tosave, pd.DataFrame(np.array(data_pre).reshape(1, 13), columns=['subject_id', 'study_id', 'dicom_id', 'path', 
                                  'Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 
                                  'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia', 'split'])], ignore_index=True)
    
df_tosave.to_csv('~/symptom_label_chexpert.csv', mode='a', index=None, header=False)

df_tosave = pd.DataFrame(columns=['subject_id', 'study_id', 'dicom_id', 'path', 
                                  'Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 
                                  'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia', 'split'])

for idx, row in tqdm(filtered_df_test.iterrows(), total=filtered_df_test.shape[0]):
    # check all the subject_id
    current_img_id = row['current_image_id']
    previous_img_id = row['previous_image_id']

    data_cur = df_cxr_record.loc[df_cxr_record['dicom_id']==current_img_id][['subject_id', 'study_id', 'dicom_id', 'path']].values.tolist()[0]
    data_pre = df_cxr_record.loc[df_cxr_record['dicom_id']==previous_img_id][['subject_id', 'study_id', 'dicom_id', 'path']].values.tolist()[0]

    data_cur_path = data_cur[-1]
    data_pre_path = data_pre[-1]
    
    data_cur[-1] = '~/data/A/'+data_cur_path.split('/')[-1]
    data_pre[-1] = '~/data/B/'+data_pre_path.split('/')[-1]

    study_id_cur = data_cur[1]
    study_id_pre = data_pre[1]

    # check corresponding symptoms
    try:
        cxr_symptoms_cur = df_cxr_symptoms.loc[df_cxr_symptoms['study_id']==study_id_cur][['Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia', 'No Finding']].values.tolist()[0]
        if cxr_symptoms_cur[-1] == 1:
            cxr_symptoms_cur = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            cxr_symptoms_cur = cxr_symptoms_cur[:-1]
        cxr_symptoms_pre = df_cxr_symptoms.loc[df_cxr_symptoms['study_id']==study_id_pre][['Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia', 'No Finding']].values.tolist()[0]
        if cxr_symptoms_pre[-1] == 1:
            cxr_symptoms_pre = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            cxr_symptoms_pre = cxr_symptoms_pre[:-1]
    except IndexError:
        not_ava_study_ids.append(study_id_cur)
        continue
    
    data_cur.extend(cxr_symptoms_cur)
    data_pre.extend(cxr_symptoms_pre)
    data_cur.append('test')
    data_pre.append('test')

    df_tosave = pd.concat([df_tosave, pd.DataFrame(np.array(data_cur).reshape(1, 13), columns=['subject_id', 'study_id', 'dicom_id', 'path', 
                                  'Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 
                                  'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia', 'split'])], ignore_index=True)
    df_tosave = pd.concat([df_tosave, pd.DataFrame(np.array(data_pre).reshape(1, 13), columns=['subject_id', 'study_id', 'dicom_id', 'path', 
                                  'Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 
                                  'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia', 'split'])], ignore_index=True)

df_tosave.to_csv('~/symptom_label_chexpert.csv', mode='a', index=None, header=False)


# Generate the symptom label_dict
label_path_prefix = '~/symptom_label_chexpert.csv'
# Get all .npy file names from label_path
label_files = os.listdir(label_path_prefix)
# Iterate over all .npy files to get the label, the key is the label_files, the value is the label, the key should not contain '.npy'
label_dict = {}
for label_file in tqdm(label_files):
    label = np.load(os.path.join(label_path_prefix, label_file))
    label_file = label_file.replace('.npy', '')
    label_dict[label_file] = label.tolist()

# Save the label_dict to a json file
with open('~/label_dict.json', 'w') as f:
    json.dump(label_dict, f)

# # The way to load the label_dict from the json file
# with open('~/label_dict.json', 'r') as f:
#     label_dict = json.load(f)
#     label_dict = {k: np.array(v) for k, v in label_dict.items()}

# with open('~/not_ava_study_ids.txt', 'w') as f:
#     for study_id in not_ava_study_ids:
#         f.write(str(study_id) + '\n')
# f.close() 

