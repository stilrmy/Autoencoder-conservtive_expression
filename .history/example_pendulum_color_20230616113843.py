import numpy as np
import random
from scipy.integrate import odeint
from PIL import Image
from scipy.integrate import solve_ivp
from sympy import derive_by_array

# Pendulum rod lengths (m), bob masses (kg).
L1, L2 = 1, 1
m1, m2 = 1, 1
# The gravitational acceleration (m.s-2).
g = 9.81
tau = 0


def get_pendulum_data(n_ics,params):
    t,x,dx,ddx,X,Xdot = generate_pendulum_data(n_ics,params)

    data = {}
    data['t'] = t
    data['x'] = x.reshape((n_ics*t.size, -1))
    data['dx'] = dx.reshape((n_ics*t.size, -1))
    data['ddx'] = ddx.reshape((n_ics*t.size, -1))
    data['z'] = X[:,0:1]
    data['dz'] = X[:,2:3]
    data['ddz'] = Xdot[:,2:3]

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

def generate_data(func, time, init_values):
    sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time, method='RK45',rtol=1e-10,atol=1e-10)
    return sol.y.T, np.array([func(0,sol.y.T[i,:]) for i in range(sol.y.T.shape[0])],dtype=np.float64)

def doublePendulum(t,y,M=0.0):
    q1,q2,q1_t,q2_t = y
    q1_2t = (-L1*g*m1*np.sin(q1) - L1*g*m2*np.sin(q1) + M + m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t)/2 - m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t + 2*L1*(-L1*np.sin(q1)*q1_t**2 - L2*np.sin(q2)*q2_t**2)*np.cos(q1) - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t + 2*L1*(L1*np.cos(q1)*q1_t**2 + L2*np.cos(q2)*q2_t**2)*np.sin(q1))/2 - m2*(2*L1*L2*np.sin(q1)*np.sin(q2) + 2*L1*L2*np.cos(q1)*np.cos(q2))*(-L2*g*m2*np.sin(q2) + m2*(2*L2*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q2)*q2_t - 2*L2*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q2)*q2_t)/2 - m2*(2*L2*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q2)*q2_t + 2*L2*(-L1*np.sin(q1)*q1_t**2 - L2*np.sin(q2)*q2_t**2)*np.cos(q2) - 2*L2*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q2)*q2_t + 2*L2*(L1*np.cos(q1)*q1_t**2 + L2*np.cos(q2)*q2_t**2)*np.sin(q2))/2 - m2*(2*L1*L2*np.sin(q1)*np.sin(q2) + 2*L1*L2*np.cos(q1)*np.cos(q2))*(-L1*g*m1*np.sin(q1) - L1*g*m2*np.sin(q1) + M + m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t)/2 - m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t + 2*L1*(-L1*np.sin(q1)*q1_t**2 - L2*np.sin(q2)*q2_t**2)*np.cos(q1) - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t + 2*L1*(L1*np.cos(q1)*q1_t**2 + L2*np.cos(q2)*q2_t**2)*np.sin(q1))/2)/(2*(m1*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2 + m2*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2)))/(2*(-m2**2*(2*L1*L2*np.sin(q1)*np.sin(q2) + 2*L1*L2*np.cos(q1)*np.cos(q2))**2/(4*(m1*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2 + m2*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2)) + m2*(2*L2**2*np.sin(q2)**2 + 2*L2**2*np.cos(q2)**2)/2)))/(m1*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2 + m2*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2)
    q2_2t = (-L2*g*m2*np.sin(q2) + m2*(2*L2*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q2)*q2_t - 2*L2*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q2)*q2_t)/2 - m2*(2*L2*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q2)*q2_t + 2*L2*(-L1*np.sin(q1)*q1_t**2 - L2*np.sin(q2)*q2_t**2)*np.cos(q2) - 2*L2*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q2)*q2_t + 2*L2*(L1*np.cos(q1)*q1_t**2 + L2*np.cos(q2)*q2_t**2)*np.sin(q2))/2 - m2*(2*L1*L2*np.sin(q1)*np.sin(q2) + 2*L1*L2*np.cos(q1)*np.cos(q2))*(-L1*g*m1*np.sin(q1) - L1*g*m2*np.sin(q1) + M + m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t)/2 - m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t + 2*L1*(-L1*np.sin(q1)*q1_t**2 - L2*np.sin(q2)*q2_t**2)*np.cos(q1) - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t + 2*L1*(L1*np.cos(q1)*q1_t**2 + L2*np.cos(q2)*q2_t**2)*np.sin(q1))/2)/(2*(m1*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2 + m2*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2)))/(-m2**2*(2*L1*L2*np.sin(q1)*np.sin(q2) + 2*L1*L2*np.cos(q1)*np.cos(q2))**2/(4*(m1*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2 + m2*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2)) + m2*(2*L2**2*np.sin(q2)**2 + 2*L2**2*np.cos(q2)**2)/2)
    return q1_t,q2_t,q1_2t,q2_2t

def generate_pendulum_data(n_ics,params):
    f  = lambda z, t: [z[1], -9.81*np.sin(z[0])]
    'z[0]-theta z[1]-theta_dot'
    t = np.arange(0, 10, .02)
    '500 time steps'

    i = 0
    X, Xdot = [], []
    #shape of X and Xdot: (50000,4)
    #structure of X:q1,q2,q1_t,q2_t
    ##structure of Xdot:q1_t,q2_t,q1_tt,q2_tt
    while (i < n_ics):
        theta1 = np.random.uniform(-np.pi, np.pi)
        thetadot = np.random.uniform(0,0)
        theta2 = np.random.uniform(-np.pi, np.pi)
        y0=np.array([theta1, theta2, thetadot, thetadot])
        x,xdot = generate_data(doublePendulum,t,y0)
        X.append(x)
        Xdot.append(xdot)
        i += 1
    X = np.vstack(X)
    Xdot = np.vstack(Xdot)
    if params['adding_noise'] == True :
        if params['noise_type'] == 'angle_noise':
            print('Adding noise to the pendulum data')
            print('noise_type: angle noise')
            mu,sigma = 0,params['noiselevel']
            noise = np.random.normal(mu,sigma,X.shape[0])
            X_noise = np.zeros(X.shape)
            Xdot_noise = np.zeros(Xdot.shape)
            for i in range(X.shape[1]):
                X_noise[:,i] = X[:,i]+noise
                Xdot_noise[:,i] = Xdot[:,i]+noise
            x,dx,ddx = pendulum_to_movie(X_noise,Xdot_noise,n_ics,params)
    x,dx,ddx = pendulum_to_movie(X,Xdot,n_ics,params)
    return t,x,dx,ddx,X,Xdot


def pendulum_to_movie(X,Xdot,n_ics,params):
    n_samples = 500
    n = 51
    y1,y2 = np.meshgrid(np.linspace(-2.5,2.5,n),np.linspace(2.5,-2.5,n))
    create_image_0 = lambda theta1,theta2,len : np.exp(-((y1-len*np.cos(theta1-np.pi/2))**2 + (y2-len*np.sin(theta1-np.pi/2))**2)/.05)
    create_image_1 = lambda theta1,theta2,len : np.exp(-((y1-len*np.cos(theta2-np.pi/2)-len*np.cos(theta1-np.pi/2))**2 + (y2-len*np.sin(theta2-np.pi/2)-len*np.sin(theta1-np.pi/2))**2)/.05)

    argument_derivative_0 = lambda theta1,dtheta1,theta2,dtheta2,len : -1/.05*(2*(y1 - len*np.cos(theta1-np.pi/2))*len*np.sin(theta1-np.pi/2)*dtheta1
                                                      + 2*(y2 - len*np.sin(theta1-np.pi/2))*(-len*np.cos(theta1-np.pi/2))*dtheta1)\

    argument_derivative_1 = lambda theta1,dtheta1,theta2,dtheta2,len : -1/.05*(2*((y1-len*np.cos(theta1-np.pi/2)-len*np.cos(theta2-np.pi/2))*len*np.sin(theta1-np.pi/2)*dtheta1)-
                                                              2*((y2-len*np.sin(theta1-np.pi/2)-len*np.sin(theta2-np.pi/2))*len*np.cos(theta1-np.pi/2))*dtheta1)\
                                                      -1/.05*(2*((y1-len*np.cos(theta1-np.pi/2)-len*np.cos(theta2-np.pi/2))*len*np.sin(theta2-np.pi/2)*dtheta2)-
                                                              2*((y2-len*np.sin(theta1-np.pi/2)-len*np.sin(theta2-np.pi/2))*len*np.cos(theta2-np.pi/2))*dtheta2)

    argument_derivative2_0 = lambda theta1,dtheta1,ddtheta1,theta2,dtheta2,ddtheta2,len : -2/.05*((len*np.sin(theta1-np.pi/2))*len*np.sin(theta1-np.pi/2)*dtheta1**2
                                                               + (y1 - len*np.cos(theta1-np.pi/2))*len*np.cos(theta1-np.pi/2)*dtheta1**2
                                                               + (y1 - len*np.cos(theta1-np.pi/2))*len*np.sin(theta1-np.pi/2)*ddtheta1
                                                               + (-len*np.cos(theta1-np.pi/2))*(-len*np.cos(theta1-np.pi/2))*dtheta1**2
                                                               + (y2 - len*np.sin(theta1-np.pi/2))*(len*np.sin(theta1-np.pi/2))*dtheta1**2
                                                               + (y2 - len*np.sin(theta1-np.pi/2))*(-len*np.cos(theta1-np.pi/2))*ddtheta1)
    
    argument_derivative2_1 = lambda theta1,dtheta1,ddtheta1,theta2,dtheta2,ddtheta2,len : -2/0.05*(2*np.sin(theta1-np.pi/2)*np.sin(theta2-np.pi/2)+2*np.cos(theta1-np.pi/2)*np.cos(theta1-np.pi/2)**2)*dtheta2*dtheta1-1/0.05*(
                                                               2*(y1*len*np.cos(theta1-np.pi/2)-np.cos(2*theta1-np.pi)*len**2+np.cos(theta1-np.pi/2)*np.cos(theta2-np.pi/2)*len**2)-2*(
                                                               -y2*len*np.sin(theta1-np.pi/2)-np.cos(2*theta1-np.pi)*len**2-np.sin(theta1-np.pi/2)*np.sin(theta2-np.pi/2)*len**2))*dtheta1**2+\
                                                               1/.05*(2*(y1 - len*np.cos(theta1-np.pi/2))*len*np.sin(theta1-np.pi/2)*ddtheta1
                                                               + 2*(y2 - len*np.sin(theta1-np.pi/2))*(-len*np.cos(theta1-np.pi/2))*ddtheta1)\
                                                               -1/0.05*(2*np.sin(theta1-np.pi/2)*np.sin(theta2-np.pi/2)+2*np.cos(theta1-np.pi/2)*np.cos(theta1-np.pi/2)**2)*dtheta2*dtheta1\
                                                               -1/0.05*(2*(y1*len*np.cos(theta2-np.pi/2)-np.cos(theta1-np.pi/2)*np.cos(theta2-np.pi/2)*len**2-np.cos(2*theta2-np.pi)*len**2)
                                                               -2*(-y2*len*np.sin(theta2-np.pi/2)+np.sin(theta1-np.pi/2)*np.sin(theta2-np.pi/2)*len**2-np.cos(2*theta2-np.pi)))*dtheta2**2\
                                                               -1/.05*(2*((y1-len*np.cos(theta1-np.pi/2)-len*np.cos(theta2-np.pi/2))*len*np.sin(theta2-np.pi/2)*ddtheta2)-
                                                               2*((y2-len*np.sin(theta1-np.pi/2)-len*np.sin(theta2-np.pi/2))*len*np.cos(theta2-np.pi/2))*ddtheta2)


                                                              # -2/.05*((len*np.sin(theta2-np.pi/2))*len*np.sin(theta2-np.pi/2)*dtheta2**2
                                                               #+ (y1 - len*np.cos(theta2-np.pi/2))*len*np.cos(theta2-np.pi/2)*dtheta2**2
                                                               #+ (y1 - len*np.cos(theta2-np.pi/2))*len*np.sin(theta2-np.pi/2)*ddtheta2
                                                               #+ (-len*np.cos(theta2-np.pi/2))*(-len*np.cos(theta2-np.pi/2))*dtheta2**2
                                                               #+ (y2 - len*np.sin(theta2-np.pi/2))*(len*np.sin(theta2-np.pi/2))*dtheta2**2
                                                               #+ (y2 - len*np.sin(theta2-np.pi/2))*(-len*np.cos(theta2-np.pi/2))*ddtheta2)

    x = np.zeros((n_ics*n_samples, n, n))
    dx = np.zeros((n_ics*n_samples, n, n))
    ddx = np.zeros((n_ics*n_samples, n, n))
    for i in range(X.shape[0]):
        if params['changing_length'] == True:
            len = random.uniform(0.2,1)
        else:
            len = 1
        x[i, :, :] = create_image(X[i, 0], X[i, 1], len)
        dx[i, :, :] = (create_image(X[i, 0], X[i, 1], len)*argument_derivative(X[i,0],X[i,2],X[i,1],X[i,3],len))
        ddx[i, :, :] = create_image(X[i, 0], X[i, 1], len)*((argument_derivative(X[i,0],X[i,2],X[i,1],X[i,3],len))**2
                    + argument_derivative2(X[i,0],Xdot[i,0],Xdot[i,2],X[i,1],Xdot[i,1],Xdot[i,3],len))
    i , len = 1,1
    return x,dx,ddx

params= {}
params['adding_noise'] = False
params['noise_type'] = 'angle_noise'
params['noiselevel'] = 1e-3
params['changing_length'] = False
plot(1,params)


