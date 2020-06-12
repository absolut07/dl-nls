#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:49:04 2020

@author: nevena
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

import numpy as np 
import tensorflow as tf
import pickle
tf.enable_v2_behavior()#for those like me who haven't yet installed TF2

#tf.config.optimizer.set_jit(True)

#%%
"""parameters"""
n=int(10000)
#layer dim:
s=10
p=50
m=100
q=100
r=100
ni=0.001 #optimizer Adam
pu=0.5 #initial condition percentage of points
stdev=0.1
mean=0
#%%
"""weights and biases"""

W11shape=[1,s]
b1shape=[1,s]
W12shape=[1,s]

W2shape=[s,p]
b2shape=[1,p]

W3shape=[p,m]
b3shape=[1,m]

W4shape=[m,q]
b4shape=[1,q]

W5shape=[q,r]
b5shape=[1,r]

W6shape=[r,r]
b6shape=[1,r]

W7shape=[r,2]
b7shape=[1,2]

W11=tf.Variable(tf.random.truncated_normal(W11shape, stddev=stdev, dtype=tf.float32))
b1=tf.Variable(tf.random.truncated_normal(b1shape, stddev=stdev, dtype=tf.float32))
W12=tf.Variable(tf.random.truncated_normal(W12shape, stddev=stdev, dtype=tf.float32))


W2=tf.Variable(tf.random.truncated_normal(W2shape, mean=mean, stddev=stdev, dtype=tf.float32))
b2=tf.Variable(tf.random.truncated_normal(b2shape, mean=mean, stddev=stdev, dtype=tf.float32))
W3=tf.Variable(tf.random.truncated_normal(W3shape, mean=mean, stddev=stdev, dtype=tf.float32))
b3=tf.Variable(tf.random.truncated_normal(b3shape, mean=mean, stddev=stdev, dtype=tf.float32))
W4=tf.Variable(tf.random.truncated_normal(W4shape, mean=mean, stddev=stdev, dtype=tf.float32))
b4=tf.Variable(tf.random.truncated_normal(b4shape, mean=mean, stddev=stdev, dtype=tf.float32))
W5=tf.Variable(tf.random.truncated_normal(W5shape, mean=mean, stddev=stdev, dtype=tf.float32))
b5=tf.Variable(tf.random.truncated_normal(b5shape, mean=mean, stddev=stdev, dtype=tf.float32))
W6=tf.Variable(tf.random.truncated_normal(W6shape, mean=mean, stddev=stdev, dtype=tf.float32))
b6=tf.Variable(tf.random.truncated_normal(b6shape, mean=mean, stddev=stdev, dtype=tf.float32))
W7=tf.Variable(tf.random.truncated_normal(W7shape, mean=mean, stddev=stdev, dtype=tf.float32))
b7=tf.Variable(tf.random.truncated_normal(b7shape, mean=mean, stddev=stdev, dtype=tf.float32))
#%%
"""input"""
t1=tf.random.uniform([int(0.98*n),1],minval=-1,maxval=1)
t2=tf.random.uniform([int(0.01*n),1],minval=-1,maxval=1)
t3=tf.random.uniform([int(0.01*n),1],minval=-1,maxval=1)
t=tf.concat([t2,t1,t3],0)
    
t_zeros=tf.zeros((int((1-pu)*n),1))
t_ones=tf.ones((int(pu*n),1))
t_ind=tf.concat([t_zeros,t_ones],0)
t_rand_ind=tf.random.shuffle(t_ind)
    
indices2=tf.concat([t_rand_ind,t_rand_ind],1) #for initial condition
    
t=(1-t_rand_ind)*t #puts zeros on places where t_rand_ind (and ind2) is one

    
xm=4*t1+tf.random.uniform([int(0.98*n),1],minval=-5,maxval=5)
x1=4*t2+tf.random.uniform([int(0.01*n),1],minval=-16,maxval=-5)
x2=4*t3+tf.random.uniform([int(0.01*n),1],minval=5,maxval=16)

x=tf.concat([x1,xm,x2],0)

#%%
 
def a(x):
    a1=tf.cos(2*x+np.pi/2)*1/tf.cosh(x)
    a2=-tf.sin(2*x+np.pi/2)*1/tf.cosh(x)
    a=tf.concat((a1,a2), axis=1)
    return a

def run_network(x,t):
    h1=tf.tanh(x*W11+t*W12+b1) #separating variables so that GradientTape can work
    h2=tf.tanh(tf.matmul(h1,W2)+b2)
    h3=tf.tanh(tf.matmul(h2,W3)+b3)
    h4=tf.tanh(tf.matmul(h3,W4)+b4)
    h5=tf.tanh(tf.matmul(h4,W5)+b5)
    h6=tf.tanh(tf.matmul(h5,W6)+b6)
    u=tf.tanh(tf.matmul(h6,W7) + b7)
    return u

def laplacian1(x,t):
    with tf.GradientTape() as g:
        g.watch(x)
        with tf.GradientTape() as gg:
            gg.watch(x)
            u=run_network(x,t)
            u1=u[:,0:1]
        du1_dx = gg.gradient(u1, x)
    d2u1_dx2 = g.gradient(du1_dx, x)
    return d2u1_dx2
def laplacian2(x,t):
    with tf.GradientTape() as g:
        g.watch(x)
        with tf.GradientTape() as gg:
            gg.watch(x)
            u=run_network(x,t)
            u2=u[:,1:]
        du2_dx = gg.gradient(u2, x)
    d2u2_dx2 = g.gradient(du2_dx, x)
    return d2u2_dx2
def u1_t(x,t):
    with tf.GradientTape() as g:
        g.watch(t)
        u=run_network(x,t)
        u1=u[:,0:1]
    du1dX=g.gradient(u1,t)
    return du1dX
def u2_t(x,t):
    with tf.GradientTape() as g:
        g.watch(t)
        u=run_network(x,t)
        u2=u[:,1:]
    du2dX=g.gradient(u2,t)
    return du2dX

def calc_loss():
    u=run_network(x,t)
    u1=u[:,0:1]
    u2=u[:,1:]
    deltas1=-u2_t(x,t)-laplacian1(x,t)-2*u1*(tf.pow(u1,2)+tf.pow(u2,2))
    deltas2=u1_t(x,t)-laplacian2(x,t)-2*u2*(tf.pow(u1,2)+tf.pow(u2,2))
    deltas=[deltas1, deltas2]
    deltas=tf.concat([deltas1,deltas2],1)
    squared_deltas=tf.square(deltas)
    jot1=tf.square(u-ic)
    #jot2=tf.square(u) #boundary condition not necessary
    error=squared_deltas+jot1*indices2#+jot2*indices1
    loss=tf.reduce_mean(error)
    return loss

def solution(x,t):
    res1=tf.cos(2*x-3*t+np.pi/2)*1/tf.cosh(x-4*t)
    res2=-tf.sin(2*x-3*t+np.pi/2)*1/tf.cosh(x-4*t)
    resenje=tf.concat([res1,res2], axis=1)
    return resenje
#%%
sol=solution(x,t)

optimizer = tf.keras.optimizers.Adam(ni)
ic=a(x) #initial condition
u=run_network(x,t) # run network to initialise weights

#trainable_variables=[W11,W12,W7,b1,b7]
trainable_variables=[W11,b1,W12, W2,b2, W3, b3, W4, b4, W5, b5, W6, b6,W7,b7]

#%%
# Train loop
"""track progress in a loop"""
for i in range(10):
    optimizer.minimize(calc_loss, trainable_variables)
    u=run_network(x,t)
    loss_train=calc_loss()
    if i%1==0:
        u1=u[:,0:1]
        umax=tf.reduce_max(u1)
        umin=tf.reduce_min(u1)
        print('iteration:', i, "loss:", loss_train.numpy(), 'u_max:', umax.numpy(), 'u_min:', umin.numpy())
kk=10   
#%%
""" train until certain loss is reached"""

while loss_train>0.045:
    kk=kk+1
    optimizer.minimize(calc_loss, trainable_variables)
    u=run_network(x,t)
    loss_train=calc_loss()
u1=u[:,0:1]
u2=u[:,1:]
umax=tf.reduce_max(u1)
umin=tf.reduce_min(u1)
print('loss', loss_train.numpy(), 'u1max:', umax.numpy(), 'u1min:', umin.numpy(), 'iteracija:', kk)
error=u-sol
sq_error=tf.sqrt(tf.reduce_mean(tf.square(error)))
abs_error=tf.reduce_mean(tf.abs(error))
inf_error=tf.reduce_max(tf.abs(error))
print('l2:', sq_error.numpy(),'srednje abs:', abs_error.numpy(), 'l besk:', inf_error.numpy())

#%%
"""plotting, all or a random subset of points (plotting many points is slow)"""
xs=list(x.numpy())
ts=list(t.numpy())
random1000=np.random.randint(0,n,size=int(0.5*n))
new_x=[]
for k in list(random1000):
    new_x.append(xs[k])
        
new_t=[]
for k in list(random1000):
    new_t.append(ts[k])

newU=u.numpy()
U1=newU[:,0]
newU1=list(U1)      
new_u1=[]
for k in list(random1000):
    new_u1.append(newU1[k]) #the real part
    
U2=newU[:,1]
newU2=list(U2)      
new_u2=[]
for k in list(random1000):
    new_u2.append(newU2[k]) #the imaginary part


ax = plt.axes(projection='3d')

for data in zip(new_x,new_t,new_u1):#(xs,ts,newU1)(new_x,new_t,new_u1)
    one,two,three=data
    ax.scatter(one,two,three)

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('Re(u)')

plt.show()
#plt.savefig('1000it', dpi=300)


#%%
"""plot the actual solution"""

xs = list(x)
ts = list(t)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
real_sol=list(sol.numpy()[:,0])



for data in zip(xs,ts,real_sol):
    one,two,three=data
    ax.scatter(one,two,three)
    
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('Re(u)')
#ax.view_init(30, 45)
#for angle in range(0, 360):
#    ax.view_init(30, angle)
#    plt.show()
#    plt.pause(.001)

plt.show()    
#plt.savefig('real_sol', dpi=300)

#%%
"""saving variables"""

with open('1000it', 'wb') as f:
    pickle.dump([trainable_variables,x,t,indices2], f)

