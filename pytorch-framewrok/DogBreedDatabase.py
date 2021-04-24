import torch
import os
import csv
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

class DogBreed(Dataset):

    def __init__(self, root,tranform, mode):
        super(DogBreed,self).__init__()
        self.root = root
        self.tranform = tranform
        self.info_df = pd.read_csv('./labels.csv')
        self.images, self.labels, self.dog_breed_dic = self.load_train_csv("./train.csv")

        if mode == "train":
            self.images = self.images[:int(0.9*len(self.images))]
            self.labels = self.labels[:int(0.9*len(self.labels))]
        if mode == "validation":
            self.images = self.images[int(0.9*len(self.images)):]
            self.labels = self.labels[int(0.9*len(self.labels)):]

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        # idx~[0-len(images)]
        img, label = self.images[idx], self.labels[idx]

        tf = self.tranform
        img = tf(img)
        label = torch.tensor(label)


        return img, label


    def load_train_csv(self, filename):
        if not os.path.exists(os.path.join("./", filename)):
            self.info_df["label"] = LabelEncoder().fit_transform(self.info_df["breed"])
            self.info_df.to_csv(filename, index=False,header=None)
            print("written to csv file: ", filename)

        images_path,  labels = [], []
        breed_dic = {}
        with open(os.path.join("./", filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img_file_name, breed, label = row
                images_path.append(os.path.join(self.root, "train", img_file_name+".jpg"))
                label = int(label)
                labels.append(label)
                breed_dic[label] = breed
        assert len(images_path) == len(labels)
        return images_path, labels, breed_dic
