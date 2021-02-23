import torch
from torch import  nn
from torch.nn import functional as F



class ResBasicBlk(nn.Module):
    """
    Implementation of the ResBasicBlk, which introduces the 'short cut connection'
    """

    def __init__(self, input_channel, output_channel, stride=1):
        """
        :param input_channel:
        :param output_channel:
        """
        super(ResBasicBlk, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)

        self.extra = nn.Sequential()
        if output_channel != input_channel:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_channel)
            )


    def forward(self, x):
        """
        :param x: [batch_size, channel(color images), height, width]
        :return: output tensor
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        """
        short cut connection, more details here https://arxiv.org/pdf/1512.03385.pdf
        extra module: [batch_size, in_channel(color images), height, width] => [batch_size, out_channel(color images), height, width]
        element-wise add
        """
        out = self.extra(x) + out
        out = F.relu(out)

        return out




class CustomizedResNet18(nn.Module):

    def __init__(self, num_class):
        super(CustomizedResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(16)
        )
        # followed 4 blocks
        # [b, 16, h, w] => [b, 32, h ,w]
        self.blk1 = ResBasicBlk(16, 32, stride=3)
        # [b, 32, h, w] => [b, 64, h, w]
        self.blk2 = ResBasicBlk(32, 64, stride=3)
        # # [b, 64, h, w] => [b, 128, h, w]
        self.blk3 = ResBasicBlk(64, 128, stride=2)
        # # [b, 128, h, w] => [b, 256, h, w]
        self.blk4 = ResBasicBlk(128, 256, stride=2)

        # [b, 256, 7, 7]
        self.outlayer = nn.Linear(256*3*3, num_class)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)


        return x



def main():
    blk = ResBasicBlk(64, 128)
    tmp = torch.randn(2, 64, 224, 224)
    out = blk(tmp)
    print('block:', out.shape)


    model = CustomizedResNet18()
    tmp = torch.randn(2, 3, 224, 224)
    out = model(tmp)
    print('resnet:', out.shape)

    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()