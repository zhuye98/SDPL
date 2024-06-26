import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")
import os
import pdb
print(os.getcwd())
# Read the CSV file into a Pandas DataFrame without the header
df_our = pd.read_csv("~/Ours_SDPL.csv", header=[0])
# df_base = pd.read_csv("./miccai2024_zy/Baseline_CheXRelFormer.csv", header=[0])

syptom = ["Lung Opacity",
          "Pleural Effusion", 
          "Atelectasis",
          "Cardiomegaly",
          "Edema",
          "Pneumothorax",
          "Consolidation",
          "Pneumonia"]


data = df_our.values
label = []
pred = []
mask = []
for i in range(len(data)):
    img_pair = data[i, 0]
    img_pair = img_pair.split(".jpg")[0]
    temp = np.load("~/physionet.org/files/chest-imagenome/1.0.0/data/my_label/{0}.npy".format(img_pair))
    mask.append(1 - np.isnan(temp).astype(np.int_))
    label.append(data[i, 1])
    pred.append(data[i, 2])
    
label = np.array(label, dtype=np.int_)
pred = np.array(pred, dtype=np.int_)   
mask = np.array(mask, dtype=np.int_)

sum = 0
for j in range(mask.shape[1]):

    idx = mask[:, j]
    label_temp = label[idx==1]   
    pred_temp = pred[idx==1] 
    sum = sum + len(label_temp)
    print("----", syptom[j], len(label_temp), "--------")
    # print(precision_score(label_temp, pred_temp, average=None))
    # print(precision_score(label_temp, pred_temp, average="weighted"), precision_score(label_temp, pred_temp, average="macro"))
    # print(recall_score(label_temp, pred_temp, average="weighted"), recall_score(label_temp, pred_temp, average="macro"))
    # print(f1_score(label_temp, pred_temp, average="weighted"), f1_score(label_temp, pred_temp, average="macro"))
    print(round(precision_score(label_temp, pred_temp, average="weighted"), 3))
    print(round(recall_score(label_temp, pred_temp, average="weighted"), 3))
    print(round(f1_score(label_temp, pred_temp, average="weighted"), 3))
    # if syptom[j] == "Atelectasis":
    #     print('stop')
    #     continue


print(len(label), sum)