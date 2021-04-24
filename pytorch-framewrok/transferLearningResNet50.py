
from torchvision.models import resnet50
import torch

import PIL.Image
from torch import optim, nn
import visdom
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from DogBreedDatabase import DogBreed

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self,x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

transform = transforms.Compose([
            lambda x: PIL.Image.open(x).convert('RGB'),  # string path= > image data
            transforms.Resize((int(224 * 1.25), int(224 * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

batch_size = 32
lr = 1e-3
epochs = 50

device = torch.device('cuda:0')
torch.manual_seed(1234)

train_db = DogBreed("../", transform, mode='train')
val_db = DogBreed("../", transform, mode='validation')

train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True,
                          num_workers=4)

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


def main():
    trained_model = resnet50(pretrained=True)
    model = nn.Sequential(
        *list(trained_model.children())[0:-1],
        Flatten(),
        nn.Linear(2048, 1024),
        nn.Dropout(0.3),
        nn.Linear(1024, 120)
    ).to(device)
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=lr)
    criteon = nn.CrossEntropyLoss()

    global_step = 0
    best_acc, best_epoch = 0, 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))

    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        if epochs % 1 == 0:

            val_acc = evaluate(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), 'resnet50-best.mdl')
                viz.line([val_acc], [global_step], win='val_acc', update='append')
    print('best acc:', best_acc, "best epoch:", best_epoch)

if __name__ == '__main__':
    main()
