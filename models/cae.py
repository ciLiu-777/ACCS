from .basic_module import BasicModule
import torch as t
from torch import nn
from torch.nn import functional as F


class CAE(BasicModule):
    '''
    实现module: 卷积自编码器
    瓶颈层：64*21*42
    '''
    def __init__(self):
        super(CAE, self).__init__()
        self.model_name = 'CAE'
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(4, stride=4),#64, 32, 32
            nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1)),  # 输出尺寸：(3, 32, 32)，1x1卷积过渡模块
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),  # 输出尺寸：(64, 64, 64)，使用stride=2进行上采样
            nn.ConvTranspose2d(64, 32, 3, stride=(2, 2), padding=1, output_padding=(1,1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=(2, 2), padding=1, output_padding=(1,1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=(2, 2), padding=1, output_padding=(1,1)),   # 输出尺寸: (1, 512, 512)
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__=='__main__':
    from torchsummary import summary

    device = 'cuda' if t.cuda.is_available() else 'cpu'
    model = CAE().encoder.to(device)
    summary(model, (1,512,512))
