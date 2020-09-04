import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import datasets

# LR model define
class LR(nn.Module):
    def __init__(self, input_dim=None, output_dim=None):
        super(LR, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim) # bias = true by default

    def forward(self, x):
        bt = x.shape[0]
        x = x.view(bt, -1)
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out

def FPR_TPR(thr,score,label):
    FPR, TPR = 0, 0
    for i in range(10): #0~9
        N_pos = len(label[label == i])
        N_neg = len(label[label != i])

        TP = (score[:,i]>thr) * (label == i)
        FP = (score[:,i]>thr) * (label != i)

        TP_num = len(TP[TP])
        FP_num = len(FP[FP])

        FPR += FP_num / N_neg
        TPR += TP_num / N_pos
    FPR /= 10
    TPR /= 10
    return FPR, TPR

dataset = datasets.create(name='mnist', batch_size=128, use_gpu=True, num_workers=4)
testloader = dataset.testloader
model_list = os.listdir('models')
thr_points = np.linspace(0, 1, 1000) 
for name in model_list:
    path = os.path.join('models', name)
    '''
    # for task1
    update_way = name.split('_')[0]
    lr = name.split('_')[1]
    epoch = name.split('_')[2].split('.')[0]
    '''

    '''
    # for task2
    update_way = name.split('_')[0]
    lr = name.split('_')[1]
    epoch = name.split('_')[2]
    reg = name.split('_')[3]
    weight_reg = name.split('_')[-1].split('.pth')[0]
    '''
    # for task3
    update_way = name.split('_')[0]
    lr = name.split('_')[1]
    epoch = name.split('_')[2]
    delta = name.split('_')[-1].split('.pth')[0]

    m = LR(28*28, 10)
    m = torch.nn.DataParallel(m).cuda()
    m.load_state_dict(torch.load(path))
    m.eval()
    prediction = []
    target = []
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.cuda(), labels.cuda()
            outputs = m(data)
            prediction.append(outputs)
            target.append(labels)
    prediction = torch.cat(prediction)
    target = torch.cat(target)

    FPR_points = [] 
    TPR_points = []
    for thr in thr_points:
        FPR, TPR = FPR_TPR(thr, prediction, target)
        FPR_points.append(FPR)
        TPR_points.append(TPR)
    plt.plot(FPR_points, TPR_points, label='lr={}'.format(lr))
    plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.savefig("FPR-TPR_3_3")
   
