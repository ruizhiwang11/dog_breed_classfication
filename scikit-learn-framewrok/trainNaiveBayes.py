from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
import os
import csv
import numpy as np
import pandas as pd
import pandas
from sklearn.naive_bayes import GaussianNB

global_features = np.load('train_X.npy')
global_labels = np.load('train_y.npy')

trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal = train_test_split(global_features,
                                                                                          global_labels,
                                                                                          test_size=0.1,
                                                                                          random_state=101)
model = GaussianNB()
kfold = KFold(n_splits=10, random_state=101,shuffle=True)
cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring="accuracy")
msg = "%s: %f (%f)" % ("NB", cv_results.mean(), cv_results.std())
print(msg)

clf = GaussianNB()
clf.fit(trainDataGlobal,trainLabelsGlobal)


test_data = np.load('test_X.npy')
prediction = clf.predict(test_data)
breed_dic = {}
breeds = []
header = []
img_file_names = []

with open("train.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        img_file_name, breed, label = row
        img_file_names.append(img_file_name)
        label = int(label)
        breeds.append(breed)
        breed_dic[label] = breed
with open("sample_submission.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        header = row[1:]
        break

final_arr = np.zeros((len(prediction),120))
count = 0
for x in prediction:
    breed = breed_dic[x]
    print(breed)
    index = header.index(breed)
    print(index)
    final_arr[count,index] = 1
    count += 1
print(len(img_file_names))
print(len(prediction))
data_to_save = np.zeros((len(prediction),121))
data_to_save[:,:-2] = img_file_names
data_to_save[1:,:] = final_arr
print(data_to_save)

# dataset = pd.DataFrame({'id': img_file_names})
# dataset.to_csv("123.csv", index=False)




print(header)


print(prediction)
