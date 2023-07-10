import numpy as np
import random
from scipy.integrate import odeint
from PIL import Image

def get_pendulum_data(n_ics,params):
    t,x,dx,ddx,z = generate_pendulum_data(n_ics,params)
    print('sample version: c=-0.14')
    data = {}
    data['t'] = t
    data['x'] = x.reshape((n_ics*t.size, -1))
    data['dx'] = dx.reshape((n_ics*t.size, -1))
    data['ddx'] = ddx.reshape((n_ics*t.size, -1))
    data['z'] = z.reshape((n_ics*t.size, -1))[:,0:1]
    data['dz'] = z.reshape((n_ics*t.size, -1))[:,1:2]
    data['ddz'] = -0.14*data['dz']-9.81*np.sin(data['z'])

    #adding noise
    
    if params['adding_noise'] == True :
        if params['noise_type'] == 'image_noise':
            print('Adding noise to the pendulum data')
            print('noise_type: image noise')
            mu,sigma = 0,params['noiselevel']
            noise = np.random.normal(mu,sigma,data['x'].shape[1])
            for i in range(data['x'].shape[0]):
                data['x'][i] = data['x'][i]+noise
                data['dx'][i] = data['dx'][i] + noise
                data['ddx'][i] = data['ddx'][i] + noise

    return data

def plot(n_ics,params):
    t, x, dx, ddx, X, Xdot = generate_pendulum_data(n_ics,params)
    print(x.shape)
    imglist = []
    for i in range(500):
        image_pend = Image.fromarray(x[0,i,:,:]*255).convert('L')
        center_dot = np.zeros([51,51])
        center_dot[26,26] = 255
        center_dot = Image.fromarray(center_dot+x[0,i,:,:]).convert('L')
        image = Image.merge("RGB",(image_pend,center_dot,image_pend))
        imglist.append(image)
    imglist[0].save('save_name.gif', save_all=True, append_images=imglist, duration=0.1)
    return

def generate_pendulum_data(n_ics,params):
    #100g pendulum
    f  = lambda z, t: [z[1], -0.14*z[1]-9.81*np.sin(z[0])]
    'pendulum with friction'
    t = np.arange(0, 10, .02)
    '500 time steps'
    z = np.zeros((n_ics,t.size,2))
    dz = np.zeros(z.shape)

    z1range = np.array([-np.pi,np.pi])
    z2range = np.array([-2.1,2.1])
    i = 0
    while (i < n_ics):
        z0 = np.array([(z1range[1]-z1range[0])*np.random.rand()+z1range[0],
            (z2range[1]-z2range[0])*np.random.rand()+z2range[0]])
        if np.abs(z0[1]**2/2. - np.cos(z0[0])) > .99:
            continue
        z[i] = odeint(f, z0, t)
        'z.shape:(n_ics,t,2)'
        'theta,theta_dot'
        'theta_dot,theta_ddot'
        i += 1
    if params['adding_noise'] == True :
        if params['noise_type'] == 'angle_noise':
            print('Adding noise to the pendulum data')
            print('noise_type: angle noise')
            mu,sigma = 0,params['noiselevel']
            noise = np.random.normal(mu,sigma,z.shape[1])
            z_noise = np.zeros(z.shape)
            dz_noise = np.zeros(z.shape)
            for i in range(z.shape[0]):
                for j in range(z.shape[2]):
                    z_noise[i,:,j] = z[i,:,j] + noise
            for i in range(dz.shape[0]):
                for j in range(dz.shape[2]):
                    dz_noise[i,:,j] = dz[i,:,j] + noise
            x,dx,ddx = pendulum_to_movie(z_noise, dz_noise)
    x,dx,ddx = pendulum_to_movie(z, dz,params)
            
   

    # n = 51
    # xx,yy = np.meshgrid(np.linspace(-1.5,1.5,n),np.linspace(1.5,-1.5,n))
    # create_image = lambda theta : np.exp(-((xx-np.cos(theta-np.pi/2))**2 + (yy-np.sin(theta-np.pi/2))**2)/.05)
    # argument_derivative = lambda theta,dtheta : -1/.05*(2*(xx - np.cos(theta-np.pi/2))*np.sin(theta-np.pi/2)*dtheta \
    #                                                   + 2*(yy - np.sin(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*dtheta)
    # argument_derivative2 = lambda theta,dtheta,ddtheta : -2/.05*((np.sin(theta-np.pi/2))*np.sin(theta-np.pi/2)*dtheta**2 \
    #                                                            + (xx - np.cos(theta-np.pi/2))*np.cos(theta-np.pi/2)*dtheta**2 \
    #                                                            + (xx - np.cos(theta-np.pi/2))*np.sin(theta-np.pi/2)*ddtheta \
    #                                                            + (-np.cos(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*dtheta**2 \
    #                                                            + (yy - np.sin(theta-np.pi/2))*(np.sin(theta-np.pi/2))*dtheta**2 \
    #                                                            + (yy - np.sin(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*ddtheta)
        
    # x = np.zeros((n_ics, t.size, n, n))
    # dx = np.zeros((n_ics, t.size, n, n))
    # ddx = np.zeros((n_ics, t.size, n, n))
    # for i in range(n_ics):
    #     for j in range(t.size):
    #         z[i,j,0] = wrap_to_pi(z[i,j,0])
    #         x[i,j] = create_image(z[i,j,0])
    #         dx[i,j] = (create_image(z[i,j,0])*argument_derivative(z[i,j,0], dz[i,j,0]))
    #         ddx[i,j] = create_image(z[i,j,0])*((argument_derivative(z[i,j,0], dz[i,j,0]))**2 \
    #                         + argument_derivative2(z[i,j,0], dz[i,j,0], dz[i,j,1]))

    return t,x,dx,ddx,z


def pendulum_to_movie(z, dz, params):
    n_ics = z.shape[0]
    n_samples = z.shape[1]
    n = 51
    y1,y2 = np.meshgrid(np.linspace(-1.5,1.5,n),np.linspace(1.5,-1.5,n))
    create_image = lambda theta,len : np.exp(-((y1-len*np.cos(theta-np.pi/2))**2 + (y2-len*np.sin(theta-np.pi/2))**2)/.05)
    argument_derivative = lambda theta,dtheta : -1/.05*(2*(y1 - len*np.cos(theta-np.pi/2))*len*np.sin(theta-np.pi/2)*dtheta \
                                                      + 2*(y2 - len*np.sin(theta-np.pi/2))*(-len*np.cos(theta-np.pi/2))*dtheta)
    argument_derivative2 = lambda theta,dtheta,ddtheta : -2/.05*((len*np.sin(theta-np.pi/2))*len*np.sin(theta-np.pi/2)*dtheta**2 \
                                                               + (y1 - len*np.cos(theta-np.pi/2))*len*np.cos(theta-np.pi/2)*dtheta**2 \
                                                               + (y1 - len*np.cos(theta-np.pi/2))*len*np.sin(theta-np.pi/2)*ddtheta \
                                                               + (-len*np.cos(theta-np.pi/2))*(-len*np.cos(theta-np.pi/2))*dtheta**2 \
                                                               + (y2 - len*np.sin(theta-np.pi/2))*(len*np.sin(theta-np.pi/2))*dtheta**2 \
                                                               + (y2 - len*np.sin(theta-np.pi/2))*(-len*np.cos(theta-np.pi/2))*ddtheta)

    x = np.zeros((n_ics, n_samples, n, n))
    dx = np.zeros((n_ics, n_samples, n, n))
    ddx = np.zeros((n_ics, n_samples, n, n))
    for i in range(n_ics):
        if params['changing_length'] == True:
            len = random.uniform(0.2,1)
        else:
            len = 1
        for j in range(n_samples):
            z[i,j,0] = wrap_to_pi(z[i,j,0])
            x[i,j] = create_image(z[i,j,0],len)
            dx[i,j] = (create_image(z[i,j,0],len)*argument_derivative(z[i,j,0], dz[i,j,0]))
            ddx[i,j] = create_image(z[i,j,0],len)*((argument_derivative(z[i,j,0], dz[i,j,0]))**2 \
                            + argument_derivative2(z[i,j,0], dz[i,j,0], dz[i,j,1]))
    return x,dx,ddx


def wrap_to_pi(z):
    z_mod = z % (2*np.pi)
    subtract_m = (z_mod > np.pi) * (-2*np.pi)
    return z_mod + subtract_m

