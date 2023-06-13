# %%
import example_pendulum
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
#import torchvision
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import csv

from torch.autograd import Variable

# %%
device = 'cuda:0'

# %%
environment = "server"
if environment == 'laptop':
    root_dir =R'C:\Users\87106\OneDrive\sindy\progress'
elif environment == 'desktop':
    root_dir = R'E:\OneDrive\sindy\progress'
elif environment == 'server':
    root_dir = R'/mnt/ssd1/stilrmy/double_pendulum/progress'
#the angle_extractor
AE_save_date = '2023-05-11'
AE_save_ver = '1'
#the angle_t_extractor
AtE_save_date = '2023-05-11'
AtE_save_ver = '1'
#the angle_tt_extractor
AttE_save_date = '2023-05-11'
AttE_save_ver = '1'
#genrate path
AE_path = os.path.join(root_dir,AE_save_date,AE_save_ver,'model.pth')
AtE_path = os.path.join(root_dir,'Angle_t_extractor',AtE_save_date,AtE_save_ver,'model.pth')
AttE_path = os.path.join(root_dir,'Angle_t_extractor',AttE_save_date,AttE_save_ver,'model.pth')

# %%
#initialize the Angle_extractor and load the parameters
class angle_predict(nn.Module):
    def __init__(self):
        super(angle_predict, self).__init__()
        self.fc1 = nn.Linear(2601, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 2)
    def forward(self, x):
        m = nn.ReLU()
        x = self.fc1(x)
        x = m(x)
        x = self.fc2(x)
        x = m(x)
        x = self.fc3(x)
        x = m(x)
        x = self.fc4(x)
        x = m(x)
        x = self.fc5(x) 
        return x
AE = angle_predict()
AE.load_state_dict(torch.load(AE_path))
AE =AE.to(device)

# %%
#initialize the Angle_t_extractor and load the parameters
class angle_t_predict(nn.Module):
    def __init__(self):
        super(angle_predict, self).__init__()
        self.fc1 = nn.Linear(2601, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 2)
    def forward(self, x):
        m = nn.ReLU()
        x = self.fc1(x)
        x = m(x)
        x = self.fc2(x)
        x = m(x)
        x = self.fc3(x)
        x = m(x)
        x = self.fc4(x)
        x = m(x)
        x = self.fc5(x) 
        return x
AtE = angle_t_predict()
AtE.load_state_dict(torch.load(AtE_path))
AtE = AtE.to(device)

# %%
#initialize the Angle_tt_extractor and load the parameters
class angle_tt_predict(nn.Module):
    def __init__(self):
        super(angle_predict, self).__init__()
        self.fc1 = nn.Linear(2601, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 2)
    def forward(self, x):
        m = nn.ReLU()
        x = self.fc1(x)
        x = m(x)
        x = self.fc2(x)
        x = m(x)
        x = self.fc3(x)
        x = m(x)
        x = self.fc4(x)
        x = m(x)
        x = self.fc5(x) 
        return x
AttE = angle_tt_predict()
AttE.load_state_dict(torch.load(AttE_path))
AttE = AttE.to(device)

# %%
def image_process(sample_size,params):
    sample_size = 10
    data = example_pendulum.get_pendulum_data(sample_size,params)
    image = data['x']
    angle = np.zeros(image.shape[0],2)
    angle_t = np.zeros(image.shape[0],2)
    angle_tt = np.zeros(image.shape[0],2)
    for i in range(image.shape[0]-2):
        input = Variable(torch.tensor(image[i,:],dtype=torch.float32).to(device))
        temp = AE.forward(input)
        temp = temp.cpu().detach().numpy()
        angle[i,0]= temp[0]
        angle[i,1]= temp[1]
    
    for i in range(image.shape[0]-2):
        input = Variable(torch.tensor(image[i,:],dtype=torch.float32).to(device))
        temp = AtE.forward(input)
        temp = temp.cpu().detach().numpy()
        angle_t[i,0]= temp[0]
        angle_t[i,1]= temp[1]
    
    for i in range(image.shape[0]-2):
        input = Variable(torch.tensor(image[i,:],dtype=torch.float32).to(device))
        temp = AttE.forward(input)
        temp = temp.cpu().detach().numpy()
        angle_tt[i,0]= temp[0]
        angle_tt[i,1]= temp[1]
    return angle,angle_t,angle_tt

# %%


# %%



