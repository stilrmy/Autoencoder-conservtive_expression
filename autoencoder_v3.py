def __clear_env():
    for key in globals().keys():
        if not key.startswith("__"):# 排除系统内建函数
            globals().pop(key)
__clear_env
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
from torch.autograd import Variable
device = 'cuda:0'
data = example_pendulum.get_pendulum_data(40)
val_data = example_pendulum.get_pendulum_data(10)


def autoencoder(input_dim,latent_dim,widths,device):
    #generate the parameters of the autoencoder
    encoder_weights,encoder_biases = build_network_layers(input_dim,latent_dim,widths,device)
    decoder_weights, decoder_biases = build_network_layers(latent_dim, input_dim, widths[::-1],device)
    return encoder_weights,encoder_biases,decoder_weights,decoder_biases


def build_network_layers(input_dim,output_dim,widths,device):
    #universal function for building network
    weights = []
    biases = []
    last_width = input_dim
    #middle layers
    for i,n_units in enumerate(widths):
        w = torch.Tensor(last_width,n_units,).to(device)
        nn.init.xavier_uniform_(w, gain=1.0)
        w = Variable(w, requires_grad=True)
        b = torch.Tensor(n_units).to(device)
        nn.init.constant_(b, 0.0)
        b = Variable(b, requires_grad=True)
        last_width = n_units
        weights.append(w)
        biases.append(b)
    #latent layer
    w = torch.Tensor(last_width,output_dim).to(device)
    nn.init.xavier_uniform_(w, gain=1.0)
    w = Variable(w,requires_grad=True)
    b = torch.Tensor(output_dim).to(device)
    nn.init.constant_(b, 0.0)
    b = Variable(b, requires_grad=True)
    weights.append(w)
    biases.append(b)
    return weights,biases

def autoencoder_forward(input,params):
    #pass through the autoencoder
    for i,weights in enumerate(params['encoder_weights']):
        input = torch.matmul(input,weights)+params['encoder_biases'][i]
        if params['activation'] == 'sigmoid' and i <len(params['encoder_weights'])-1:
            input = torch.sigmoid(input)
    for i,weights in enumerate(params['decoder_weights']):
        input = torch.matmul(input,weights)+params['decoder_biases'][i]
        if params['activation'] == 'sigmoid'and i <len(params['decoder_weights'])-1:
            input = torch.sigmoid(input)
    return input

def encoder_forward(input,params):
    for i,weights in enumerate(params['encoder_weights']):
        input = torch.matmul(input,weights)+params['encoder_biases'][i]
        if params['activation'] == 'sigmoid' and i <len(params['encoder_weights'])-1:
            input = torch.sigmoid(input)
    return input

def decoder_forward(input,params):
    for i,weights in enumerate(params['decoder_weights']):
        input = torch.matmul(input,weights)+params['decoder_biases'][i]
        if params['activation'] == 'sigmoid' and i <len(params['decoder_weights'])-1:
            input = torch.sigmoid(input)
    return input

def encoder_derivative(image,image_t,image_tt,params):
    #calculate the time/second time derivative of the latent(s) using the autoencoder
    dz = image_t
    ddz = image_tt
    input = image
    weights = params['encoder_weights']
    biases = params['encoder_biases']
    if params['activation'] == 'sigmoid':
        for i in range(len(weights)-1):
            input = torch.matmul(input,weights[i])+biases[i]
            input = torch.sigmoid(input)
            dz_prev = torch.matmul(dz,weights[i])
            sigmoid_derivative = torch.multiply(input,1-input)
            sigmoid_derivative2 = torch.multiply(sigmoid_derivative,1-2*input)
            dz = torch.multiply(sigmoid_derivative,dz_prev)
            ddz = torch.multiply(sigmoid_derivative2,torch.square(dz_prev))\
                  + torch.multiply(sigmoid_derivative,torch.matmul(ddz,weights[i]))
        dz = torch.matmul(dz,weights[-1])
        ddz = torch.matmul(ddz,weights[-1])
    return dz,ddz

def decoder_derivative(z,dz,ddz,params):
    #calculate the time/second time derivative of the latent(s) using the autoencoder
    input = z
    weights = params['decoder_weights']
    biases = params['decoder_biases']
    if params['activation'] == 'sigmoid':
        for i in range(len(weights)-1):
            input = torch.matmul(input,weights[i])+biases[i]
            input = torch.sigmoid(input)
            dz_prev = torch.matmul(dz,weights[i])
            sigmoid_derivative = torch.multiply(input,1-input)
            sigmoid_derivative2 = torch.multiply(sigmoid_derivative,1-2*input)
            dz = torch.multiply(sigmoid_derivative,dz_prev)
            #print(sigmoid_derivative.shape)
            ddz = torch.multiply(sigmoid_derivative2,torch.square(dz_prev))\
                  + torch.multiply(sigmoid_derivative,torch.matmul(ddz,weights[i]))
        dz = torch.matmul(dz,weights[-1])
        ddz = torch.matmul(ddz,weights[-1])
    return dz,ddz

def training(data,params,device):
    opt = torch.optim.Adam(generate_parameter(params), lr=params['learning_rate'])
    total_loss = 0
    total_loss_z = 0
    for j in range(params['epoch_size']//params['batch_size']):
        batch_idxs = np.arange(j * params['batch_size'], (j + 1) * params['batch_size'])
        z = torch.from_numpy(data['z'][batch_idxs]).to(torch.float32).to(device)
        x = torch.from_numpy(data['x'][batch_idxs]).to(torch.float32).to(device)
        dx = torch.from_numpy(data['dx'][batch_idxs]).to(torch.float32).to(device)
        ddx = torch.from_numpy(data['ddx'][batch_idxs]).to(torch.float32).to(device)
        x_predict = autoencoder_forward(x,params)
        z_predict = encoder_forward(x,params)
        dz_predict,ddz_predict = encoder_derivative(x,dx,ddx,params)
        dx_predict,ddx_predict = decoder_derivative(z_predict,dz_predict,ddz_predict,params)
        loss_x = torch.mean((x - x_predict)**2)
        loss_dx = torch.mean((x - dx_predict) ** 2)
        loss_ddx = torch.mean((x - ddx_predict) ** 2)
        loss_z = torch.mean((z - z_predict)**2)
        loss = params['loss_weight_x'] * loss_x \
               + params['loss_weight_dx'] * loss_dx \
               + params['loss_weight_ddx'] * loss_ddx
        total_loss += loss
        total_loss_z += loss_z
        loss.backward()
        opt.step()
        opt.zero_grad()
    avg_loss = total_loss/(params['epoch_size']//params['batch_size'])
    avg_loss_z = total_loss_z/(params['epoch_size']//params['batch_size'])
    return avg_loss,avg_loss_z

def generate_parameter(params):
    encoder_weights = params['encoder_weights']
    encoder_biases = params['encoder_biases']
    params_temp = []
    for i in range(len(params['widths'])):
        params_temp.append(encoder_weights[i])
        params_temp.append(encoder_biases[i])
    decoder_weights = params['decoder_weights']
    decoder_biases = params['decoder_biases']
    for i in range(len(params['widths'])):
        params_temp.append(decoder_weights[i])
        params_temp.append(decoder_biases[i])
    return params_temp

def generate_vhat(vhat):
    vhat = tuple(vhat)
    yield vhat

def validation(data,params,device):
    total_loss = 0
    total_loss_z = 0
    params['epoch_size'] = val_data["x"].shape[0]
    for j in range(params['epoch_size']//params['batch_size']):
        batch_idxs = np.arange(j * params['batch_size'], (j + 1) * params['batch_size'])
        z = torch.from_numpy(data['z'][batch_idxs]).to(torch.float32).to(device)
        x = torch.from_numpy(data['x'][batch_idxs]).to(torch.float32).to(device)
        dx = torch.from_numpy(data['dx'][batch_idxs]).to(torch.float32).to(device)
        ddx = torch.from_numpy(data['ddx'][batch_idxs]).to(torch.float32).to(device)
        x_predict = autoencoder_forward(x,params)
        z_predict = encoder_forward(x,params)
        dz_predict,ddz_predict = encoder_derivative(x,dx,ddx,params)
        dx_predict,ddx_predict = decoder_derivative(z_predict,dz_predict,ddz_predict,params)
        loss_x = torch.mean((x - x_predict)**2)
        loss_dx = torch.mean((x - dx_predict) ** 2)
        loss_ddx = torch.mean((x - ddx_predict) ** 2)
        loss_z = torch.mean((z - z_predict)**2)
        loss = params['loss_weight_x'] * loss_x \
               + params['loss_weight_dx'] * loss_dx \
               + params['loss_weight_ddx'] * loss_ddx
        total_loss += loss
        total_loss_z += loss_z
        print('batch_{} --- image loss: {} --- z loss: {}'.format(j,loss,loss_z))
    avg_loss = total_loss/(params['epoch_size']//params['batch_size'])
    avg_loss_z = total_loss_z/(params['epoch_size']//params['batch_size'])
    return avg_loss,avg_loss_z

def plotting(loss_history,loss_z_history):
    fig, a = plt.subplots(2, 1)
    a[0].plot(loss_history)
    a[0].set_title("loss")
    a[1].plot(loss_z_history)
    a[1].set_title("z_loss")
    plt.subplots_adjust(hspace=1)
    plt.show()
    return

def saving(params,coef,expr,L):
    params_names = [params['encoder_weights'],params['encoder_biases'],params['decoder_weights'],params['decoder_biases']]
    params_names_str = ['encoder_weights','encoder_biases','decoder_weights','decoder_biases']
    for j,param_set in enumerate(params_names):
        for i,elements in enumerate(param_set):
            np.save(r"E:\OneDrive\sindy\progress\{}\{}\autoencoder_{}_layer{}.npy".format(params['date'],params['ver'],params_names_str[j],i),
                   elements.clone().detach().cpu().numpy())
    np.save(r"E:\OneDrive\sindy\progress\{}\{}\coef.npy".format(params['date'],params['ver']),coef.clone().detach().cpu().numpy())
    np.save(r"E:\OneDrive\sindy\progress\{}\{}\expr.npy".format(params['date'], params['ver']),
            expr)
    sub_params = params
    del sub_params['encoder_weights']
    del sub_params['encoder_biases']
    del sub_params['decoder_weights']
    del sub_params['decoder_biases']
    with open(r"E:\OneDrive\sindy\progress\{}\{}\params.txt".format(params['date'],params['ver']),'w') as file:
        file.write(str(sub_params))
        file.write('\r\t')
        file.write(L)
        file.write('\r\t')
        file.close()
    print('params saved')
    return

def loading_coef(params,date,load_number,file_route,device):
    print('loading params')
    params_names_str = ['encoder_weights', 'encoder_biases', 'decoder_weights', 'decoder_biases']
    for param_name in params_names_str:
        params['{}'.format(param_name)] = []
        for i in range(len(params['widths'])+1):
            file_name = 'autoencoder_{}_layer{}.npy'.format(param_name,i)
            route = os.path.join(file_route,date,load_number,file_name)
            temp_loader = np.load(route)
            temp_loader = torch.tensor(temp_loader).to(device)
            temp_loader = Variable(temp_loader, requires_grad=True)
            print(temp_loader.shape)
            params['{}'.format(param_name)].append(temp_loader)
    #coef = np.load(r'{}\{}\{}\coef.npy'.format(file_route,date,load_number))
    #coef = torch.Tensor(coef).to(device)
    #coef = Variable(coef,requires_grad=True)
    #expr = np.load(r'{}\{}\{}\expr.npy'.format(file_route,date,load_number))
    return params

# initialization
# file_route = R'C:\Users\87106\OneDrive\sindy\progress'
"""
save_params = True
load_params = True

load_date = '3-7'
load_ver = 2
widths = [1024,512,128]
params={}
params['ver'] = "3"
params['accumulate_epochs'] = 6000
params['date'] = '3-7'
params['widths'] = widths

if load_params == True:
    params = loading(params,load_date,load_ver,file_route,device)
else:
    encoder_weights,encoder_biases,decoder_weights,decoder_biases = autoencoder(2601,1,widths,device)
    params['encoder_weights'] = encoder_weights
    params['encoder_biases'] = encoder_biases
    params['decoder_weights'] = decoder_weights
    params['decoder_biases'] = decoder_biases

params['activation'] = 'sigmoid'
params['max_epochs'] = 2000
params['epoch_size'] = data["x"].shape[0]
params['batch_size'] = 500
params['learning_rate'] = 0
params['learning_rate_stage1'] = 1e-3
params['learning_rate_stage2'] = 1e-3
params['loss_weight_x'] = 1
params['loss_weight_dx'] = 1
params['loss_weight_ddx'] = 1
params['learning_rate_switching_point'] = 700
loss_history = []
loss_z_history = []
params['learning_rate'] = params['learning_rate_stage1']

for epochs in range(params['max_epochs']):
    if epochs >= params['learning_rate_switching_point']:
        params['learning_rate'] = params['learning_rate_stage2']
    loss,loss_z = training(data,params,device)
    loss_history.append(loss.clone().detach().cpu().numpy())
    loss_z_history.append(loss_z.clone().detach().cpu().numpy())
    print('epoch: {} === loss: {}\t=== loss_z: {}'.format(epochs,loss,loss_z))

validation(val_data,params,device)
if save_params == True:
    saving(params)
plotting(loss_history,loss_z_history)
"""