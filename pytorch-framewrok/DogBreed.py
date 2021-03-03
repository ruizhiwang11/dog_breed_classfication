import torch
import os
import random, csv
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset,DataLoader
from    torchvision import transforms
import PIL

class DogBreed(Dataset):

    def __init__(self, root,tranform, mode):
        super(DogBreed,self).__init__()
        self.root = root
        self.tranform = tranform
        self.info_df = pd.read_csv('../scikit-learn-framewrok/labels.csv')
        self.images, self.labels, self.dog_breed_dic = self.load_train_csv("train.csv")

        if mode == "train":
            self.images = self.images[:int(0.8*len(self.images))]
            self.labels = self.labels[:int(0.8*len(self.labels))]
        if mode == "validation":
            self.images = self.images[int(0.8*len(self.images)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        # idx~[0-len(images)]
        img, label = self.images[idx], self.labels[idx]

        tf = self.tranform
        img = tf(img)
        label = torch.tensor(label)


        return img, label

    def denormalize(self, x_hat):
        """To view to image without normalization"""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x

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




def main():
    import visdom
    import time
    import torchvision

    transform = transforms.Compose([
        lambda x: PIL.Image.open(x).convert('L'),  # string path= > image data
        transforms.Resize((int(224 * 1.25), int(224 * 1.25))),
        transforms.RandomRotation(15),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    viz = visdom.Visdom()
    dogbreed = DogBreed("../",transform, "train")
    x, y = next(iter(dogbreed))
    print('sample:', x.shape, y.shape, y)

    # viz.image(dogbreed.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    viz.image(x, win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(dogbreed, batch_size=32, shuffle=True, num_workers=8)

    for x, y in loader:
        # viz.images(dogbreed.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.image(x, win='sample_x', opts=dict(title='sample_x'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))

        time.sleep(10)


if __name__ == '__main__':
    main()