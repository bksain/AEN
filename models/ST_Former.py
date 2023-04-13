import torch
from torch import nn

from models.S_Former import spatial_transformer
from models.T_Former import temporal_transformer
# from S_Former import spatial_transformer
# from T_Former import temporal_transformer
import math
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
import time

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GenerateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_former = spatial_transformer()
        # for param in self.s_former.parameters():
        #     param.requires_grad_(False)
        # self.s_former = Conformer()

        self.t_former = temporal_transformer()

        #self.fc1 = nn.Linear(8192, 512)
        self.sigmoid = nn.Sigmoid()
        self.GELU = GELU()
        # self.fc = nn.Linear(512, 7)
        self.fc0_0 = nn.Linear(512, 784)

        self.fc0_1 = nn.Linear(512, 784)
        self.fc1_2 = nn.Linear(784, 784)
        self.fc2_3 = nn.Linear(784, 784)

        self.fc0_1_o = nn.Linear(784, 784)
        self.fc1_2_o = nn.Linear(784, 784)
        self.fc2_3_o = nn.Linear(784, 784)

        self.fc1_1 = nn.Linear(784, 784)

        self.fc0 = nn.Linear(512, 7)
        self.fc1 = nn.Linear(784, 3)
        # self.fc2 = nn.Linear(784, 7)
        self.fc3 = nn.Linear(784, 7)
        # self.fc = nn.Linear(256 + 256, 7)


        #2D pretrain
        self.fc2D = nn.Linear(784,7)

    def forward(self, x):

        xs = self.s_former(x)
        #"""
        x = self.t_former(xs)

        # uni-temporal cross spatial att
        x0_1 = torch.matmul(self.fc0_1(x[0]).unsqueeze(2), x[1].unsqueeze(1))
        x0_1 = x0_1 / math.sqrt(x[1].size(-1))
        x0_1 = F.softmax(x0_1, dim=1)
        x0_1 = torch.matmul(x0_1,x[1].unsqueeze(2))[:,:,0]
        x0_1 = self.fc0_1_o(x0_1) + self.fc0_0(x[0])

        # x1_2 = torch.matmul(self.fc1_2(x[1]).unsqueeze(2), x[2].unsqueeze(1))
        # x1_2 = x1_2 / math.sqrt(x[2].size(-1))
        # x1_2 = F.softmax(x1_2, dim=1)
        # x1_2 = torch.matmul(x1_2, x[2].unsqueeze(2))[:, :, 0]
        # x1_2 = self.fc1_2_o(x1_2) + x[2]
        #
        # x2_3 = torch.matmul(self.fc2_3(x[2]).unsqueeze(2), x[3].unsqueeze(1))
        # x2_3 = x2_3 / math.sqrt(x[3].size(-1))
        # x2_3 = F.softmax(x2_3, dim=1)
        # x2_3 = torch.matmul(x2_3, x[3].unsqueeze(2))[:, :, 0]
        # x2_3 = self.fc2_3_o(x2_3) + x[3]

        x2_3 = torch.matmul(self.fc2_3(x0_1).unsqueeze(2), x[3].unsqueeze(1))
        x2_3 = x2_3 / math.sqrt(x[3].size(-1))
        x2_3 = F.softmax(x2_3, dim=1)
        x2_3 = torch.matmul(x2_3, x[3].unsqueeze(2))[:, :, 0]
        x2_3 = self.fc2_3_o(x2_3) + x0_1



        x0 = self.fc0(x[0])
        x1 = self.fc1(x0_1)
        #x2 = self.fc2(x1_2)
        x3 = self.fc3(x2_3)

        #
        PG = F.log_softmax(x0) #7
        PL1 = F.log_softmax(x1) #3
        PL2 = F.log_softmax(x3) #7

        # x = (x0 * 0.7 + x1 * 0.1 + x2 * 0.1 + x3 * 0.1)
        # x = self.fc(x)
        #"""
        
        #2D pretrain
        out2D = self.fc2D(xs[3])
        # PG = []
        # PL1 = []
        # PL2 = []

        return PG, PL1, PL2, out2D, x, xs, x0, x1, x3


class GenerateModel_2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_former = spatial_transformer()
        # for param in self.s_former.parameters():
        #     param.requires_grad_(False)
        

        # 2D pretrain
        self.fc2D = nn.Linear(784, 7)

    def forward(self, x):
        xs = self.s_former(x)
        

        # 2D pretrain
        out2D = self.fc2D(xs[3])
        PG = []
        PL1 = []
        PL2 = []

        return PG, PL1, PL2, out2D


if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 112, 112))
    model = GenerateModel()
    macs, params = get_model_complexity_info(model, (1, 16, 3, 112, 112), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # CNN = CAERSNet()
    # model = CAERSNet_video_Transf(CNN,16,7)

    time_sum = 0
    cnt = 10
    for i in range(cnt):
        start_time = time.time()
        model(img)
        inference_time = time.time() - start_time
        time_sum += inference_time
        print(inference_time)

    print(time_sum/cnt)
