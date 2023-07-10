#!/usr/bin/env python
# coding: utf-8

# In[5]:


import example_pendulum_friction as example_pendulum
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


# In[6]:


device = 'cuda:7'


# In[ ]:


environment = "server"
if environment == 'laptop':
    root_dir =R'C:\Users\87106\OneDrive\sindy\progress'
elif environment == 'desktop':
    root_dir = R'E:\OneDrive\sindy\progress'
elif environment == 'server':
    root_dir = R'/mnt/ssd1/stilrmy/Angle_detector/progress'
#the angle_extractor
AE_save_date = '2023-06-26'
AE_save_ver = '1'
#the angle_t_extractor
AtE_save_date = '2023-06-26'
AtE_save_ver = '1'
#genrate path
AE_path = os.path.join(root_dir,AE_save_date,AE_save_ver,'model.pth')
AtE_path = os.path.join(root_dir,'Angle_t_extractor',AtE_save_date,AtE_save_ver,'model.pth')


# In[8]:


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


# In[10]:


#initialize the Angle_t_extractor and load the parameters
class angle_t_predict(nn.Module):
    def __init__(self):
        super(angle_t_predict, self).__init__()
        self.fc1 = nn.Linear(7803, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 64)
        self.fc6 = nn.Linear(64, 1)
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
        x = m(x)
        x = self.fc6(x) 
        return x
AtE = angle_t_predict()
AtE.load_state_dict(torch.load(AtE_path))
AtE = AtE.to(device)




def image_process(sample_size,params):
    data = example_pendulum.get_pendulum_data(sample_size,params)
    image = data['x']
    angle = np.zeros(image.shape[0]-2)
    angle_t = np.zeros(image.shape[0]-2)
    angle_tt = np.zeros(image.shape[0]-2)
    for i in range(image.shape[0]-2):
        input = Variable(torch.tensor(image[i,:],dtype=torch.float32).to(device))
        temp = AE.forward(input)
        temp = temp.cpu().detach().numpy()
        angle[i] = temp[0]
        angle_tt[i] = temp[1]
    
    for i in range(image.shape[0]-2):
        input = torch.tensor(image[i:i+3,:],dtype=torch.float32).to(device)
        input = input.view(-1)
        input = Variable(input)
        temp = AtE.forward(input)
        temp = temp.cpu().detach().numpy()
        angle_t[i] = temp
    t = data['t']
    print('type of t: ',t.dtype)
    return angle,angle_t,angle_tt

def plotting(sample_size,params):
    data = example_pendulum.get_pendulum_data(sample_size,params)
    image = data['x']
    angle = np.zeros(image.shape[0]-2)
    angle_t = np.zeros(image.shape[0]-2)
    angle_tt = np.zeros(image.shape[0]-2)
    for i in range(image.shape[0]-2):
        input = Variable(torch.tensor(image[i,:],dtype=torch.float32).to(device))
        temp = AE.forward(input)
        temp = temp.cpu().detach().numpy()
        angle[i] = temp[0]
        angle_tt[i] = temp[1]
    
    for i in range(image.shape[0]-2):
        input = torch.tensor(image[i:i+3,:],dtype=torch.float32).to(device)
        input = input.view(-1)
        input = Variable(input)
        temp = AtE.forward(input)
        temp = temp.cpu().detach().numpy()
        angle_t[i] = temp
    t = data['t']
    fig, axs = plt.subplots(3, figsize=(10, 15))
    # Plot angle
    axs[0].plot(t[:498], data['z'][:498], label='True values')
    axs[0].plot(t[:498], angle[:498], label='Predicted values')
    axs[0].set(xlabel='Time', ylabel='Angle')
    axs[0].legend()

    # Plot angle_t
    axs[1].plot(t[:498],data['dz'][:498], label='True values')
    axs[1].plot(t[:498], angle_t[:498], label='Predicted values')
    axs[1].set(xlabel='Time', ylabel='Angle_t')
    axs[1].legend()

    # Plot angle_tt
    axs[2].plot(t[:498],data['ddz'][:498], label='True values')
    axs[2].plot(t[:498], angle_tt[:498], label='Predicted values')
    axs[2].set(xlabel='Time', ylabel='Angle_tt')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('comparison_along_time.png')
    plt.show()
    return
'''
params = {}
params['adding_noise'] = False
params['noise_type'] = 'image_noise'
params['noiselevel'] = 1e-1
params['changing_length'] = False
params['c'] = 1.4e-2
params['g'] = 9.81
params['l'] = 1
plotting(1,params)
'''

