#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys

from sympy import symbols, simplify, derive_by_array
from scipy.integrate import solve_ivp
from xLSINDy_sp import *
from sympy.physics.mechanics import *
from sympy import *
from Data_generator import image_process
import sympy
import torch
import sys
import HLsearch as HL
import example_pendulum
import time


# In[2]:


save = False
environment = "laptop"
sample_size = 10
device = 'cuda:0'


# In[3]:


sys.path.append(r'../../../HLsearch/')
#Saving Directory
if environment == 'laptop':
    root_dir =R'C:\Users\87106\OneDrive\sindy\progress'
elif environment == 'desktop':
    root_dir = R'E:\OneDrive\sindy\progress'
elif environment == 'server':
    root_dir = R'/home/stilrmy/Angle_extractor'
x,dx,ddx = image_process(sample_size)
X = []
Xdot = []
for i in range(len(x)):
    temp_list = [float(x[i]),float(dx[i])]
    X.append(temp_list)
    temp_list = [float(dx[i]),float(ddx[i])]
    Xdot.append(temp_list)
X = np.vstack(X)
print(X.shape)
Xdot = np.vstack(Xdot)


# In[4]:


states_dim = 2
states = ()
states_dot = ()
for i in range(states_dim):
    if(i<states_dim//2):
        states = states + (symbols('x{}'.format(i)),)
        states_dot = states_dot + (symbols('x{}_t'.format(i)),)
    else:
        states = states + (symbols('x{}_t'.format(i-states_dim//2)),)
        states_dot = states_dot + (symbols('x{}_tt'.format(i-states_dim//2)),)
print('states are:',states)
print('states derivatives are: ', states_dot)


# In[5]:


#Turn from sympy to str
states_sym = states
states_dot_sym = states_dot
states = list(str(descr) for descr in states)
states_dot = list(str(descr) for descr in states_dot)


# In[6]:


#build function expression for the library in str
expr= HL.buildFunctionExpressions(2,states_dim,states,use_sine=True)
#expr=['x0', 'x0_t', 'sin(x0)', 'cos(x0)', 'x0**2', 'x0*x0_t', 'x0_t**2', 'x0*sin(x0)', 'x0_t*sin(x0)', 'sin(x0)**2', 'x0*cos(x0)', 'x0_t*cos(x0)', 'sin(x0)*cos(x0)', 'cos(x0)**2']
"a list of candidate function"
print(expr)
expr.pop(5)


# In[7]:


Zeta, Eta, Delta = LagrangianLibraryTensor(X,Xdot,expr,states,states_dot,device,scaling=True)
Eta = Eta.to(device)
Zeta = Zeta.to(device)
Delta = Delta.to(device)


# In[8]:


mask = torch.ones(len(expr),device=device)
xi_L = torch.ones(len(expr), device=device).data.uniform_(-10,10)
prevxi_L = xi_L.clone().detach()


# In[9]:


def loss(pred, targ):
    loss = torch.mean((pred - targ)**2) 
    return loss 


# In[10]:


def clip(w, alpha):
    clipped = torch.minimum(w,alpha)
    clipped = torch.maximum(clipped,-alpha)
    return clipped


# In[11]:


def proxL1norm(w_hat, alpha):
    if(torch.is_tensor(alpha)==False):
        alpha = torch.tensor(alpha)
    w = w_hat - clip(w_hat,alpha)
    return w


# In[12]:


def training_loop(coef, prevcoef, Zeta, Eta, Delta,xdot, bs, lr, lam):
    loss_list = []
    tl = xdot.shape[0]
    n = xdot.shape[1]
    momentum = True
    #if(torch.is_tensor(xdot)==False):
        #xdot = torch.from_numpy(xdot).to(device).float()
    v = coef.clone().detach().to(device).requires_grad_(True)
    prev = prevcoef.clone().detach().to(device).requires_grad_(True)
    for i in range(tl//bs):
        #computing acceleration with momentum
        #if (momentum == True):
            #vhat = (v + ((i - 1) / (i + 2)) * (v - prev)).clone().detach().requires_grad_(True)
       # else:
        vhat = v.requires_grad_(True).clone().detach().requires_grad_(True)
        prev = v
        #Computing loss
        zeta = Zeta[:,i*bs:(i+1)*bs]
        eta = Eta[:,i*bs:(i+1)*bs]
        delta = Delta[:,i*bs:(i+1)*bs]
        x_t = xdot[i*bs:(i+1)*bs,:]
        #forward
        q_tt_pred = lagrangianforward(vhat,zeta,eta,delta,x_t,device)
        q_tt_true = xdot[i*bs:(i+1)*bs,n//2:].T
        q_tt_true = torch.from_numpy(q_tt_true).to(device).float()
        lossval = loss(q_tt_pred, q_tt_true)
        lossval.requires_grad_(True)
        #Backpropagation
        lossval.backward()
        with torch.no_grad():
            v = vhat - lr * vhat.grad
            v = proxL1norm(v, lr * lam)
            # Manually zero the gradients after updating weights
            vhat.grad = None
        loss_list.append(lossval.item())
    return v, prev, torch.tensor(loss_list).mean().item()


# In[13]:


Epoch = 100
i = 1
lr = 1e-3
lam = 0.1
temp = 1000
while(i<=Epoch):
    xi_L , prevxi_L, lossitem= training_loop(xi_L,prevxi_L,Zeta,Eta,Delta,Xdot,128,lr=lr,lam=lam)
    if i %20 == 0:
        print("\n")
        print("Stage 1")
        print("Epoch "+str(i) + "/" + str(Epoch))
        print("Learning rate : ", lr)
        print("Average loss : " , lossitem)
    if(temp <=5e-3):
        break
    if(temp <=1e-1):
        lr = 1e-5
    temp = lossitem
    i+=1


# In[14]:


threshold = 1e-2
surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
expr = np.array(expr)[surv_index].tolist()

xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
prevxi_L = xi_L.clone().detach()
mask = torch.ones(len(expr),device=device)

## obtaining analytical model
xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=2)
L = HL.generateExpression(xi_Lcpu,expr)
print(len(xi_L))
print(L)


# In[ ]:


for stage in range(100):
    
    #Redefine computation after thresholding
    Zeta, Eta, Delta = LagrangianLibraryTensor(X,Xdot,expr,states,states_dot,device,scaling=False)
    Eta = Eta.to(device)
    Zeta = Zeta.to(device)
    Delta = Delta.to(device)

    #Training
    Epoch = 100
    i = 1
    lr = 1e-4
    if(stage==1):
        lam = 0
    else:
        lam = 0.1
    temp = 1000
    while(i<=Epoch):   
        xi_L , prevxi_L, lossitem= training_loop(xi_L,prevxi_L,Zeta,Eta,Delta,Xdot,128,lr=lr,lam=lam)
        if i %50 == 0:
            print("\n")
            print("Stage ",stage)
            print("Epoch "+str(i) + "/" + str(Epoch))
            print("Learning rate : ", lr)
            print("Average loss : " , lossitem)
        temp = lossitem
        if(temp <=1e-6):
            break
        i+=1
    
    ## Thresholding small indices ##
    threshold = 1e-1
    surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
    expr = np.array(expr)[surv_index].tolist()

    xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
    prevxi_L = xi_L.clone().detach()
    mask = torch.ones(len(expr),device=device)

    ## obtaining analytical model
    xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
    L = HL.generateExpression(xi_Lcpu,expr)
    print("expression length:\t",len(xi_L))
    print("Result stage " + str(stage+2) + ":" , L)


# In[ ]:





# In[ ]:




