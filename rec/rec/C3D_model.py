import torch
import torch.nn as nn
from torchsummary import summary

class C3D(nn.Module):
    # 该代码适用3x16x112x112 3通道 16帧 112*112
    # num_classes表示要分类的数量
    def __init__(self,num_classes=2):
        super(C3D, self).__init__()

        self.block1=nn.Sequential(
            nn.Conv3d(3,64,kernel_size=(3,3,3),padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2)),
        )

        self.block2=nn.Sequential(
            nn.Conv3d(64,128,kernel_size=(3,3,3),padding=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        )

        self.block3=nn.Sequential(
            nn.Conv3d(128,256,kernel_size=(3,3,3),padding=(1,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256,256,kernel_size=(3,3,3),padding=(1,1,1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        )

        self.block4=nn.Sequential(
            nn.Conv3d(256,512,kernel_size=(3,3,3),padding=(1,1,1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512,512,kernel_size=(3,3,3),padding=(1,1,1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        )

        self.block5 = nn.Sequential(
            nn.Conv3d(512,512,kernel_size=(3,3,3),padding=(1,1,1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512,512,kernel_size=(3,3,3),padding=(1,1,1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2),padding=(0,1,1))
        )

        self.block6=nn.Sequential(
            nn.Flatten(),
            # 512*1*4*4
            nn.Linear(8192,4096),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(4096,num_classes)
        )

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)

    # def __load__pretrained_model(self):
    #     wth_dict=torch.load("pth")
    #     print(wth_dict)
    #     self.load_state_dict(wth_dict)
        

    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        x=self.block6(x)
        return x

if __name__ == "__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=C3D(num_classes=2).to(device)
    print(summary(model,(3,16,112,112)))