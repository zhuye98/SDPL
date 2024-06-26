import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import pdb
from prettytable import PrettyTable


pred_path_our = "~/Ours_SDPL.csv"
patient_path = "~/data/symptom_label_chexpert.csv"

df_our = pd.read_csv(pred_path_our, header=[0])
data = df_our.values

label = data[:, 1]
pred = data[:, 2]

label = label.astype(np.int_)
pred = pred.astype(np.int_)

print(len(label), len(label[label==0]), len(label[label==2]), len(label[label==1]))

class_recall = recall_score(label, pred, average=None)
class_pre = precision_score(label, pred, average=None)
class_f1 = f1_score(label, pred, average=None)


table_1 = PrettyTable()
table_1.field_names = ['Metrics\\Classes', 'Worsened', 'Stable', 'Improved']
table_1.add_row(['Precision', round(class_pre[0], 3), round(class_pre[2], 3), round(class_pre[1], 3)])
table_1.add_row(['Recall', round(class_recall[0], 3), round(class_recall[2], 3), round(class_recall[1], 3)])
table_1.add_row(['F1_score', round(class_f1[0], 3), round(class_f1[2], 3), round(class_f1[1], 3)])

table_2 = PrettyTable()
table_2.field_names = ['Metrics\\Classes', ' ']
table_2.add_row(['Weighted_Precision', round(precision_score(label, pred, average="weighted"), 3)])
table_2.add_row(['Weighted_Recall', round(recall_score(label, pred, average="weighted"), 3)])
table_2.add_row(['Weighted_F1_score', round(f1_score(label, pred, average="weighted"), 3)])

print(table_1)
print(table_2)

# print(score_fun(label, pred, average="macro"))

