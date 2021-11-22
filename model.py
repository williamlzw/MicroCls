import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d


class ConvBNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MicroBlock(nn.Module):
    def __init__(self, nh, kernel_size):
        super().__init__()
        self.conv1 = ConvBNACT(nh, nh, kernel_size, groups=nh, padding=1)
        self.conv2 = ConvBNACT(nh, nh, 1)

    def forward(self, x):
        x = x + self.conv1(x)
        x = self.conv2(x)
        return x


class MicroCls(nn.Module):
    def __init__(self, nh=64, depth=2, nclass=60):
        super().__init__()
        assert(nh >= 2)
        self.conv = ConvBNACT(3, nh, 4, 4)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(MicroBlock(nh, 3))
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(nh, nclass)

    def forward(self, x):
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)    
        x = adaptive_avg_pool2d(x, 1)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    import time
    x = torch.randn(1, 3, 32, 32)
    model = MicroCls(1024, depth=2, nclass=10)
    t0 = time.time()
    out = model(x)
    t1 = time.time()
    print(out.shape, (t1-t0)*1000)
    torch.save(model, 'micro.pth')
    # from torchsummaryX import summary
    # summary(model, x)
