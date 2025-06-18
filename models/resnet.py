from .basic_module import BasicModule
import torch as t
from torch import nn
from torch.nn import functional as F


class ResidualBlock(BasicModule):
    """
    实现子module: Residual Block
    """

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None, use_attention=True):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.right = shortcut
        # 根据开关决定是否添加注意力
        self.attention = CSAttention(outchannel) if use_attention else nn.Identity()

    def forward(self, x):
        out = self.left(x)      # 64*16*16
        out = self.attention(out)                         # 添加注意力1
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(BasicModule):
    """
    实现主module：ResNet34
    ResNet34 包含多个layer，每个layer又包含多个residual block
    用子module来实现residual block，用_make_layer函数来实现layer
    """

    def __init__(self, input_size=3, num_classes=2):
        super(ResNet34, self).__init__()
        self.model_name = "Resnet34"

        # 前几层: 图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # 重复的layer（包装以下），分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(128, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 64, 4, stride=2)
        self.layer3 = self._make_layer(64, 64, 6, stride=2)
        self.layer4 = self._make_layer(64, 64, 3, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类用的全连接
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        构建layer,包含多个residual block
        """
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel),
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)


class ResNet18(BasicModule):
    """
    实现主module: ResNet18
    分类数：8
    """

    def __init__(self, num_classes=8, use_attention=True):
        super(ResNet18, self).__init__()
        self.model_name = "Resnet18"

        # 前几层: 图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 32
            nn.BatchNorm2d(32),  # 32
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),# 移除最大池化层以保留空间信息
        )

        # 重复的layer（包装以下），分别有2个residual block
        self.layer1 = self._make_layer(32, 32, 2, stride=1, use_attention=use_attention)  # 64, 64,
        self.layer2 = self._make_layer(32, 64, 2, stride=2, use_attention=use_attention)  # 64, 128,
        self.layer3 = self._make_layer(64, 128, 2, stride=2, use_attention=use_attention)  # 128, 256
        self.layer4 = self._make_layer(128, 256, 2, stride=2, use_attention=use_attention)  # 256, 512,

        # 自适应池处理最终尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 添加Dropout层（关键修改）
        # self.dropout = nn.Dropout(p=0.5)  # 丢弃概率设为0.7

        # 分类用的全连接
        self.fc = nn.Linear(256, num_classes)  # 512

    def _make_layer(self, inchannel, outchannel, block_num, stride=1, use_attention=True):
        """
        构建layer,包含多个residual block
        只有第一个改通道，后面不动
        """
        shortcut = None
        if stride != 1 or inchannel != outchannel:
            shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
                nn.BatchNorm2d(outchannel),
            )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut, use_attention=use_attention))

        for _ in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel, use_attention=use_attention))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B,256,4,4]

        x = self.adaptive_pool(x)  # [B,256,1,1]
        x = x.view(x.size(0), -1)  # [B,256]

        # x = self.dropout(x)           # 添加Dropout

        return self.fc(x)


class ChannelAttention(BasicModule):
    """
    子网络：通道注意力
    通道注意力模仿人脑的“特征选择”机制，通过评估不同通道的重要性，增强关键特征通道的响应。
    """

    def __init__(self, channel, reduction):
        super(ChannelAttention, self).__init__()

        # 将每个通道的全局空间信息压缩为单个数值，形成通道描述符
        self.maxpool = nn.AdaptiveMaxPool2d(1)# 最大池化关注显著特征
        self.avgpool = nn.AdaptiveAvgPool2d(1)# 平均池化反映整体分布
        # 通过降维-激活-升维的结构，建模通道间的非线性关系。共享权重处理两种池化结果，增强泛化性。
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
        )
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x) #32*32*32 -> 32*1*1
        avg_result = self.avgpool(x)
        max_out = self.se(max_result) #32*1*1 -> 32*1*1
        avg_out = self.se(avg_result)
        return F.sigmoid(max_out + avg_out)


class SpatialAttention(BasicModule):
    """
    子网络：空间注意力
    空间注意力模拟视觉系统的“空间聚焦”机制，定位特征图中的关键区域。
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度压缩，保留空间分布信息，最大响应突出重要区域，平均响应抑制噪声。
        max_result, _ = t.max(x, dim=1, keepdim=True)# 通道最大响应反映显著区域
        avg_result = t.mean(x, dim=1, keepdim=True)# 通道平均响应保留全局信息
        result = t.cat([max_result, avg_result], dim=1)# 拼接特征图：2*H*W
        # 使用较大的卷积核（如7×7）捕获广域空间上下文关系，生成细粒度注意力图。padding保持尺寸不变
        output = self.conv(result)
        return  F.sigmoid(output)
    

class CSAttention(BasicModule):
    '''
    子模块：Convolutional Block Attention Module
    对两个Attention进行串联，channel 在前，spatial在后
    给定一个中间特征图，模块会沿着两个独立的维度（通道和空间）依次推断注意力图，然后将注意力图乘以输入特征图以进行自适应特征修饰。
    即先筛选重要通道，再在空间上聚焦关键区域。
    '''
    def __init__(self, channel=256, reduction=16, kernel_size=7):
        super(CSAttention, self).__init__()

        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        residual = x  # 保存原始输入（用于残差连接）
        out = x * self.ca(x)  # 每个通道的权重作用到所有空间位置
        out = out * self.sa(out)  # 每个位置的权重作用到所有通道
        return out * residual


if __name__ == "__main__":
    from torchsummary import summary

    device = "cuda" if t.cuda.is_available() else "cpu"
    model = ResNet18(use_attention=True).to(device)
    summary(model, (3, 32, 32))
    
