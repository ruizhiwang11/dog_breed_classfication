import PIL
import torch
from torch import optim, nn
import visdom
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from DogBreedDatabase import DogBreed
from SimpleNN import  SimpleNN

batch_size = 32
lr = 1e-3
epochs = 25

device = torch.device('cuda')
torch.manual_seed(1234)

transform = transforms.Compose([
            lambda x: PIL.Image.open(x).convert('L'),  # string path= > image data
            transforms.Resize((int(128 * 1.25), int(128 * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
        ])

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
            x = x.view(x.size(0), -1)
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
        print(correct, total)
    return correct / total


def main():
    model = SimpleNN()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    global_step = 0
    best_acc, best_epoch = 0, 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))

    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            print(x.shape)
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

                torch.save(model.state_dict(), 'simpleNN-best.mdl')
                viz.line([val_acc], [global_step], win='val_acc', update='append')
    print('best acc:', best_acc, "best epoch:", best_epoch)

if __name__ == '__main__':
    main()
