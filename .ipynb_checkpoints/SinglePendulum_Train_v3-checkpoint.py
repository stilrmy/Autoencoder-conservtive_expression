import numpy as np
import sys 

from autoencoder_v3 import *
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
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable





sys.path.append(r'../../../HLsearch/')
device = 'cuda:0'
data = example_pendulum.get_pendulum_data(5)



#Saving Directory
rootdir = "../Single Pendulum/Data/"
torch.cuda.manual_seed(123)
num_sample = 100
momentum = True
training = True
save = True
loading = False
noiselevel = 2e-2
file_route = R'E:\OneDrive\sindy\progress'
load_date = '3-20'
load_ver = 1
#parameters of the autoencoder
params={}
expr_log = {}
expr_log['stage'] = []
expr_log['expr'] = []
expr_log['loss'] = []
expr_log['Lag'] = []
params['latent_dim'] = 1
params['learning_rate'] = 1e-8
params['learning_rate_sindy'] = 1e-2
params['learning_rate_stage_2'] = 1e-9
params['learning_rate_sindy_stage_2'] = 1e-2
params['ver'] = 3
params['date'] = '3-21'
params['widths'] = [1024,512,128]
params['activation'] = 'sigmoid'
params['loss_weight_decoder'] = 1
params['loss_weight_sindy_x'] = 0.7
params['loss_weight_sindy_image_tt'] = 0.7
params['loss_weight_regularization'] = 0
params['loss_weight_sparsity'] = 0
if loading  == True:
    params,xi_L,expr = loading_coef(params,load_date,load_ver,file_route,device)
else:
    encoder_weights,encoder_biases,decoder_weights,decoder_biases = autoencoder(len(data['x'][1]),params['latent_dim'],params['widths'],device)
    params['encoder_weights'] = encoder_weights
    params['encoder_biases'] = encoder_biases
    params['decoder_weights'] = decoder_weights
    params['decoder_biases'] = decoder_biases
params['stages_1'] = 400
params['threshold_1'] = 0.2
params['stages_2'] = 30
params['threshold_2'] = 0.2
image = torch.from_numpy(data['x']).to(torch.float32).to(device)
image_t = torch.from_numpy(data['dx']).to(torch.float32).to(device)
image_tt = torch.from_numpy(data['ddx']).to(torch.float32).to(device)
theta = torch.from_numpy(data['dz'][0]).to(torch.float32).to(device)


def ae_forward(params,image,image_t,image_tt):
    x = encoder_forward(image,params)
    x = x.cpu().detach().numpy()
    dx,ddx = encoder_derivative(image,image_t,image_tt,params)
    dx = dx.cpu().detach().numpy()
    ddx = ddx.cpu().detach().numpy()
    X = []
    Xdot = []
    for i in range(len(x)):
        temp_list = [float(x[i]),float(dx[i])]
        X.append(temp_list)
        temp_list = [float(dx[i]),float(ddx[i])]
        Xdot.append(temp_list)
    X = np.vstack(X)
    Xdot = np.vstack(Xdot)
    return X,Xdot





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
#expr= HL.buildFunctionExpressions(2,states_dim,states,use_sine=True)
expr=['x0', 'x0_t', 'sin(x0)', 'cos(x0)', 'x0**2', 'x0*x0_t', 'x0_t**2', 'x0*sin(x0)', 'x0_t*sin(x0)', 'sin(x0)**2', 'x0*cos(x0)', 'x0_t*cos(x0)', 'sin(x0)*cos(x0)', 'cos(x0)**2']
#expr=['cos(x0)','x0_t**2']
"a list of candidate function"

#expr.pop(7)
#expr.pop(9)
"?"
print(expr)
device = 'cuda:0'
loss_history =[]


if loading == False:
    mask = torch.ones(len(expr),device=device)
    xi_L = torch.ones(len(expr), device=device,requires_grad=True).data.uniform_(-10,10)
    #xi_L = torch.tensor([9.81,0.5],device=device)
    xi_L = Variable(xi_L,requires_grad=True)
prevxi_L = xi_L.clone().detach()


def loss(coef,x_predict, x, image_predict, image, image_tt_predict, image_tt,theta, params,device):

    x = torch.Tensor(x).to(device)
    x_predict = x_predict.to(device)
    image_predict = image_predict.to(device)
    image = image.to(device)
    image_tt_predict = image_tt_predict.to(device)
    image_tt = image_tt.to(device)
    losses = {}
    losses['rebuild'] = torch.mean(torch.square(x_predict - theta))
    losses['sindy_x'] = torch.mean(torch.square(x_predict - x))
    losses['sindy_image_tt'] = torch.mean(torch.square(image_tt_predict - image_tt))
    losses['decoder'] = torch.mean(torch.square(image_predict - image))
    losses['regularization'] = torch.mean(torch.abs(coef))
    losses['sparsity'] = len(coef) - 2
    loss = params['loss_weight_decoder'] * losses['decoder'] + \
            params['loss_weight_sindy_x'] * losses['sindy_x'] + \
            params['loss_weight_sindy_image_tt'] * losses['sindy_image_tt'] + \
            params['loss_weight_regularization'] * losses['regularization'] + \
            params['loss_weight_sparsity'] * losses['sparsity'] + \
            losses['rebuild']
    return loss 


def clip(w, alpha):
    w.to(device)
    alpha.to(device)
    clipped = torch.minimum(w,alpha)
    clipped = torch.maximum(clipped,-alpha)
    return clipped

def proxL1norm(w_hat, alpha):
    if(torch.is_tensor(alpha)==False):
        alpha = torch.tensor(alpha)
    w_hat.to(device)
    alpha.to(device)
    w = w_hat - clip(w_hat,alpha)
    return w


def training_loop(coef, prevcoef, images , images_t , images_tt ,theta, params, bs,expr,states,states_dot):
    #vhat = coef.clone().detach().to(device).requires_grad_(True)
    coef.requires_grad=True
    opt = torch.optim.Adam(generate_parameter(params), lr=params['learning_rate'])
    opt2 = torch.optim.Adam([coef],lr=params['learning_rate_sindy'])
    loss_list = []
    X,Xdot = ae_forward(params, images, images_t, images_tt)
    tl = Xdot.shape[0]
    n = Xdot.shape[1]
    #momentum = True
    prev = prevcoef.clone().detach().to(device).requires_grad_(True)

    if(torch.is_tensor(Xdot)==False):
        Xdot = torch.from_numpy(Xdot).to(device).float()



    for i in range(tl//bs):
        X, Xdot = ae_forward(params, images, images_t, images_tt)
        Zeta, Eta, Delta = LagrangianLibraryTensor(X, Xdot, expr, states, states_dot,device, scaling=True)
        #Computing loss
        zeta = Zeta[:,i*bs:(i+1)*bs]
        eta = Eta[:,i*bs:(i+1)*bs]
        delta = Delta[:,i*bs:(i+1)*bs]
        x_t = Xdot[i*bs:(i+1)*bs,:]

        #forward
        q_tt_pred = lagrangianforward(coef,zeta,eta,delta,x_t,device)
        q_tt_pred_T = torch.reshape(q_tt_pred,[500,1])
        image = images[i * bs:(i + 1) * bs]
        image_t = images_t[i * bs:(i + 1) * bs]
        image_tt = images_tt[i * bs:(i + 1) * bs]
        x = encoder_forward(image,params)
        image_predict = decoder_forward(x,params)
        dx,ddx = encoder_derivative(image,image_t,image_tt,params)

        image_t_predict,image_tt_predict = decoder_derivative(x,dx,q_tt_pred_T,params)
        q_tt_true = Xdot[i*bs:(i+1)*bs,n//2:].T

        lossval = loss(coef,q_tt_pred,q_tt_true,image_predict,image,image_tt_predict,image_tt,theta,params,device)
        #Backpropagation
        opt.zero_grad()
        opt2.zero_grad()
        lossval.backward()

        #gradient check code

        """
        print(coef.grad)
        for i in range(len(params['widths'])):
            print("e_w",i)
            print(params['encoder_weights'][i].grad)
            print("e_b", i)
            print(params['encoder_biases'][i].grad)
        for i in range(len(params['widths'])):
            print("d_w", i)
            print(params['decoder_weights'][i].grad)
            print("d_b", i)
            print(params['decoder_biases'][i].grad)
        """
        #end

        opt.step()
        opt2.step()

        loss_list.append(lossval.item())
    print("Average loss : " , torch.tensor(loss_list).mean().item())
    return coef, prev, torch.tensor(loss_list).mean().item()

def thresholding(stages,xi_L,prevxi_L,image,image_t,image_tt,theta,params,threshold,expr,loss_history):
    L = str()
    for stage in range(stages):
        loss_sum = 0
        #Training
        Epoch = 100
        i = 1
        lr = params['learning_rate']
        while(i<=Epoch):
            print("\n")
            print("Stage " + str(stage+2))
            print("Epoch "+str(i) + "/" + str(Epoch))
            print("Learning rate : ", lr)
            xi_L , prevxi_L, lossitem= training_loop(xi_L,prevxi_L,image,image_t,image_tt,theta,params,500
                                                     ,expr,states,states_dot)

            loss_history.append(lossitem)
            temp = lossitem
            loss_sum += temp
            if(temp <=1e-6):
                break
            i+=1

        ## Thresholding small indices ##

        origin_len = len(expr)
        print(xi_L)
        surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
        print(surv_index)
        print(expr)
        expr = np.array(expr)[surv_index].tolist()
        new_len = len(expr)

        #print(len(expr))

        xi_L =xi_L[surv_index].clone().detach()
        #xi_L.requires_grad=True

        prevxi_L = xi_L.clone().detach()
        mask = torch.ones(len(expr),device=device)

        ## obtaining analytical model
        xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
        L = HL.generateExpression(xi_Lcpu,expr)
        #if stages % 10 == 0:
            #expr_log['Lag'].append(L)

        if origin_len != new_len:
            expr_log['stage'].append(stage)
            expr_log['expr'].append(expr)
            expr_log['Lag'].append(str(L))
            expr_log['loss'].append(loss_sum/100)

        print("Result stage " + str(stage+2) + ":" , L)
        #if len(expr)<= 2:
           #break
    return xi_L,prevxi_L,expr,L,loss_history

if loading == False:
    threshold = 0.5
    Epoch = 100
    i = 1
    lr = params['learning_rate']
    temp = 1000
    while(i<=Epoch):
        print("\n")
        print("Stage 1")
        print("Epoch "+str(i) + "/" + str(Epoch))
        print("Learning rate : ", lr)
        xi_L , prevxi_L, lossitem= training_loop(xi_L,prevxi_L,image,image_t,image_tt,theta,params,500
                                                 , expr, states, states_dot)

        loss_history.append(lossitem)
        if(temp <=5e-3):
            break
        if(temp <=1e-1):
            lr = 1e-5
        temp = lossitem
        i+=1


    ## Thresholding small indices ##

    threshold = 1e-1
    surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
    expr = np.array(expr)[surv_index].tolist()

    xi_L =xi_L[surv_index].clone().detach()

    #xi_L.requires_grad=True
    prevxi_L = xi_L.clone().detach()
    mask = torch.ones(len(expr),device=device)

    ## obtaining analytical model
    xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=2)
    L = HL.generateExpression(xi_Lcpu,expr)
    print(L)





xi_L,prevxi_L,expr,L,loss_history = thresholding(params['stages_1'],xi_L,prevxi_L,image,image_t,image_tt,theta,params,params['threshold_1'], expr,loss_history)
params['learning_rate'] = params['learning_rate_stage_2']
params['learning_rate_sindy'] = params['learning_rate_sindy_stage_2']
xi_L,prevxi_L,expr,L,loss_history = thresholding(params['stages_2'],xi_L,prevxi_L,image,image_t,image_tt,theta,params,params['threshold_2'], expr, loss_history)

print("history")
for i in range(len(expr_log['stage'])):
    print("stage{}\tloss{}\texpr{}\tLagrangian{}".format(expr_log['stage'][i],expr_log['loss'][i],expr_log['expr'][i],expr_log['Lag'][i]))

#or elements in expr_log['Lag']:
    #print(elements)

plt.plot(loss_history)
plt.show()
if(save==True):
    #Saving Equation in string
    saving(params,xi_L,expr,L)


# %%



