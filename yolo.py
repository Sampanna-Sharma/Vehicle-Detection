import torch
import torch.nn as nn

class yolo(nn.Module):
    def __init__(self,n_class):
        super(yolo, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #64    
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4))
        #16
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2))

        #7
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        # 5
        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=5, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU())        


        self.layer6 = nn.Sequential(
            nn.Conv2d(8, n_class, kernel_size=1, stride=1),
           # nn.BatchNorm2d(n_class),
            # nn.Sigmoid()
             )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)

        return out
