import numpy as np
import sys 

from autoencoder_v2 import loading,encoder_forward,encoder_derivative
from sympy import symbols, simplify, derive_by_array
from scipy.integrate import solve_ivp
from xLSINDy_sp import *
from sympy.physics.mechanics import *
from sympy import *
import sympy
import torch
import sys
import HLsearch as HL
import example_pendulum
import time





sys.path.append(r'../../../HLsearch/')
device = 'cuda:0'
data = example_pendulum.get_pendulum_data(40)



#Saving Directory
rootdir = "../Single Pendulum/Data/"

num_sample = 100
training = True
save = False
noiselevel = 2e-2
file_route = R'C:\Users\87106\OneDrive\sindy\progress'
load_date = '3-6'
load_ver = 1
widths = [1024,512,128]
params={}
params['ver'] = 1
params['date'] = '3-7'
params['widths'] = widths
params['activation'] = 'sigmoid'
params = loading(params,load_date,load_ver,file_route,device)
image = torch.from_numpy(data['x']).to(torch.float32).to(device)
image_t = torch.from_numpy(data['dx']).to(torch.float32).to(device)
image_tt = torch.from_numpy(data['ddx']).to(torch.float32).to(device)
x = encoder_forward(image,params)
x = x.cpu().detach().numpy()
dx,ddx = encoder_derivative(image,image_t,image_tt,params)
dx = dx.cpu().detach().numpy()
ddx = ddx.cpu().detach().numpy()
X = []
Xdot = []
for i in range(len(x)):
    temp_list = [float(x[i]),float(dx[i])]
    print(temp_list)
    X.append(temp_list)
    temp_list = [float(dx[i]),float(ddx[i])]
    Xdot.append(temp_list)
X = np.vstack(X)
print(X.shape)
Xdot = np.vstack(Xdot)
print(X.shape)





#adding noise
"""
mu, sigma = 0, noiselevel
noise = np.random.normal(mu, sigma, X.shape[0])
for i in range(X.shape[1]):
    X[:,i] = X[:,i]+noise
    Xdot[:,i] = Xdot[:,i]+noise
"""
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


#Turn from sympy to str
states_sym = states
states_dot_sym = states_dot
states = list(str(descr) for descr in states)
states_dot = list(str(descr) for descr in states_dot)


#build function expression for the library in str
expr= HL.buildFunctionExpressions(2,states_dim,states,use_sine=True)
#expr=['x0', 'x0_t', 'sin(x0)', 'cos(x0)', 'x0**2', 'x0*x0_t', 'x0_t**2', 'x0*sin(x0)', 'x0_t*sin(x0)', 'sin(x0)**2', 'x0*cos(x0)', 'x0_t*cos(x0)', 'sin(x0)*cos(x0)', 'cos(x0)**2']
"a list of candidate function"
print(expr)
expr.pop(5)
"?"

device = 'cuda:0'


Zeta, Eta, Delta = LagrangianLibraryTensor(X,Xdot,expr,states,states_dot,scaling=True)
Eta = Eta.to(device)
Zeta = Zeta.to(device)
Delta = Delta.to(device)


mask = torch.ones(len(expr),device=device)
xi_L = torch.ones(len(expr), device=device).data.uniform_(-10,10)
prevxi_L = xi_L.clone().detach()


def loss(pred, targ):
    loss = torch.mean((pred - targ)**2) 
    return loss 


def clip(w, alpha):
    clipped = torch.minimum(w,alpha)
    clipped = torch.maximum(clipped,-alpha)
    return clipped

def proxL1norm(w_hat, alpha):
    if(torch.is_tensor(alpha)==False):
        alpha = torch.tensor(alpha)
    w = w_hat - clip(w_hat,alpha)
    return w


def training_loop(coef, prevcoef, Zeta, Eta, Delta,xdot, bs, lr, lam):
    loss_list = []
    tl = xdot.shape[0]
    n = xdot.shape[1]
    momentum = True
    if(torch.is_tensor(xdot)==False):
        xdot = torch.from_numpy(xdot).to(device).float()
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
    print("Average loss : " , torch.tensor(loss_list).mean().item())
    return v, prev, torch.tensor(loss_list).mean().item()


Epoch = 100
i = 1
lr = 1e-3
lam = 0.1
temp = 1000
while(i<=Epoch):
    print("\n")
    print("Stage 1")
    print("Epoch "+str(i) + "/" + str(Epoch))
    print("Learning rate : ", lr)
    xi_L , prevxi_L, lossitem= training_loop(xi_L,prevxi_L,Zeta,Eta,Delta,Xdot,128,lr=lr,lam=lam)
    if(temp <=5e-3):
        break
    if(temp <=1e-1):
        lr = 1e-5
    temp = lossitem
    i+=1


## Thresholding small indices ##
threshold = 1e-2
surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
expr = np.array(expr)[surv_index].tolist()

xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
prevxi_L = xi_L.clone().detach()
mask = torch.ones(len(expr),device=device)

## obtaining analytical model
xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=2)
L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-2)
print(simplify(L))



## Next round Selection ##
for stage in range(100):
    
    #Redefine computation after thresholding
    Zeta, Eta, Delta = LagrangianLibraryTensor(X,Xdot,expr,states,states_dot,scaling=False)
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
        print("\n")
        print("Stage " + str(stage+2))
        print("Epoch "+str(i) + "/" + str(Epoch))
        print("Learning rate : ", lr)
        xi_L , prevxi_L, lossitem= training_loop(xi_L,prevxi_L,Zeta,Eta,Delta,Xdot,128,lr=lr,lam=lam)
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
    L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-2)
    print("Result stage " + str(stage+2) + ":" , simplify(L))


if(save==True):
    #Saving Equation in string
    text_file = open(rootdir + "lagrangian_" + str(noiselevel)+ "_noise.txt", "w")
    text_file.write(L)
    text_file.close()


# %%



