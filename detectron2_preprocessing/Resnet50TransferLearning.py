from torchvision.models import resnet50
import torch

import PIL.Image
from torch import optim, nn
import visdom
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import pandas as pd
import torch.nn.functional as F

from DogBreedDatabase import DogBreed

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self,x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


train_transform = transforms.Compose([
            lambda x: PIL.Image.open(x).convert('RGB'),  # string path= > image data
            transforms.Resize((224,224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
validation_transform = train_transform = transforms.Compose([
            lambda x: PIL.Image.open(x).convert('RGB'),  # string path= > image data
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
batch_size = 32
lr = 0.0003
epochs = 100
comp_df = pd.read_csv('./train.csv', names=["directory","breed","annotation"], header= None)
class_sample_count = comp_df.annotation.value_counts().sort_index().tolist()
weights = 1 / torch.Tensor(class_sample_count)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)

device = torch.device('cuda:0')
torch.manual_seed(1234)

train_db = DogBreed("./", train_transform, mode='train')
val_db = DogBreed("./", validation_transform, mode='validation')

train_loader = DataLoader(train_db, batch_size=batch_size,
                          num_workers=4,shuffle=True)

val_loader = DataLoader(val_db, batch_size=batch_size, num_workers=2)

viz = visdom.Visdom()


def evaluate(model, loader):
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
        print(correct, total)
    return correct / total


class net(torch.nn.Module):
    def __init__(self, base_model, base_out_features, num_classes):
        super(net,self).__init__()
        self.base_model=base_model
        self.linear1 = torch.nn.Linear(base_out_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.output = torch.nn.Linear(512,num_classes)
    def forward(self,x):
        x = F.relu(self.base_model(x))
        # x = F.batch_norm(x)
        x = F.relu(self.linear1(x))
        x = self.bn1(x)
        # x = F.batch_norm(x)
        x = F.dropout(x, 0.9)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x

def main():
    trained_model = resnet50(pretrained=True)

    model_final = nn.Sequential(
        *list(trained_model.children())[0:-1],
        Flatten(),
        nn.Linear(2048,1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(1024, 120),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = optim.SGD(params=model_final.parameters(), momentum=0.9, lr=lr)
    criteon = nn.CrossEntropyLoss()

    global_step = 0
    best_acc, best_epoch = 0, 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))

    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model_final(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        if epochs % 1 == 0:

            val_acc = evaluate(model_final, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model_final.state_dict(), 'resnet50-best-SGD-momentum.mdl')
                viz.line([val_acc], [global_step], win='val_acc', update='append')
    print('best acc:', best_acc, "best epoch:", best_epoch)

if __name__ == '__main__':
    main()
