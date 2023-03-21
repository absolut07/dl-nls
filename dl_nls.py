#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:49:04 2020

@author: nevena
"""
#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import tensorflow as tf
import pickle

#%%
tf.enable_v2_behavior()  # if tf v2 is not installed yet

# tf.config.optimizer.set_jit(True)


def create_weights(s, p, m, q, r, mean, stdev):
    """Initialization of weights and biases"""
    shapes_weights = [
        [1, s],
        [1, s],
        [s, p],
        [p, m],
        [m, q],
        [q, r],
        [r, r],
        [r, 2],
    ]
    shapes_biases = [
        [1, s],
        [1, p],
        [1, m],
        [1, q],
        [1, r],
        [1, r],
        [1, 2],
    ]
    trainable_vars = []
    for shape in shapes_weights:
        trainable_vars.append(
            tf.Variable(
                tf.random.truncated_normal(
                    shape, mean=mean, stddev=stdev, dtype=tf.float32
                )
            )
        )
    for shape in shapes_biases:
        trainable_vars.append(
            tf.Variable(
                tf.random.truncated_normal(
                    shape, mean=mean, stddev=stdev, dtype=tf.float32
                )
            )
        )
    return trainable_vars


def create_input(n, pu):
    """Creating inputs (focusing on the interval around x=4t)"""
    t1 = tf.random.uniform([int(0.98 * n), 1], minval=-1, maxval=1)
    t2 = tf.random.uniform([int(0.01 * n), 1], minval=-1, maxval=1)
    t3 = tf.random.uniform([int(0.01 * n), 1], minval=-1, maxval=1)
    t = tf.concat([t2, t1, t3], 0)

    t_zeros = tf.zeros((int((1 - pu) * n), 1))
    t_ones = tf.ones((int(pu * n), 1))
    t_ind = tf.concat([t_zeros, t_ones], 0)
    t_rand_ind = tf.random.shuffle(t_ind)

    indices_zero_one = tf.concat([t_rand_ind, t_rand_ind], 1)

    t = (1 - t_rand_ind) * t  # puts zeros on places where t_rand_ind (and ind2) is one

    xm = 4 * t1 + tf.random.uniform([int(0.98 * n), 1], minval=-5, maxval=5)
    x1 = 4 * t2 + tf.random.uniform([int(0.01 * n), 1], minval=-16, maxval=-5)
    x2 = 4 * t3 + tf.random.uniform([int(0.01 * n), 1], minval=5, maxval=16)

    x = tf.concat([x1, xm, x2], 0)

    return [x, t, indices_zero_one]


#%%


def a(x):
    a1 = tf.cos(2 * x + np.pi / 2) * 1 / tf.cosh(x)
    a2 = -tf.sin(2 * x + np.pi / 2) * 1 / tf.cosh(x)
    a = tf.concat((a1, a2), axis=1)
    return a


def run_network(x, t, trainable_vars):
    W11, W12, W2, W3, W4, W5, W6, W7, b1, b2, b3, b4, b5, b6, b7 = trainable_vars
    h1 = tf.tanh(
        x * W11 + t * W12 + b1
    )  # separating variables so that GradientTape can work
    h2 = tf.tanh(tf.matmul(h1, W2) + b2)
    h3 = tf.tanh(tf.matmul(h2, W3) + b3)
    h4 = tf.tanh(tf.matmul(h3, W4) + b4)
    h5 = tf.tanh(tf.matmul(h4, W5) + b5)
    h6 = tf.tanh(tf.matmul(h5, W6) + b6)
    u = tf.tanh(tf.matmul(h6, W7) + b7)
    return u


def laplacian1(x, t):
    with tf.GradientTape() as g:
        g.watch(x)
        with tf.GradientTape() as gg:
            gg.watch(x)
            u = run_network(x, t, trainable_vars)
            u1 = u[:, 0:1]
        du1_dx = gg.gradient(u1, x)
    d2u1_dx2 = g.gradient(du1_dx, x)
    return d2u1_dx2


def laplacian2(x, t):
    with tf.GradientTape() as g:
        g.watch(x)
        with tf.GradientTape() as gg:
            gg.watch(x)
            u = run_network(x, t, trainable_vars)
            u2 = u[:, 1:]
        du2_dx = gg.gradient(u2, x)
    d2u2_dx2 = g.gradient(du2_dx, x)
    return d2u2_dx2


def u1_t(x, t):
    with tf.GradientTape() as g:
        g.watch(t)
        u = run_network(x, t, trainable_vars)
        u1 = u[:, 0:1]
    du1dX = g.gradient(u1, t)
    return du1dX


def u2_t(x, t):
    with tf.GradientTape() as g:
        g.watch(t)
        u = run_network(x, t)
        u2 = u[:, 1:]
    du2dX = g.gradient(u2, t, trainable_vars)
    return du2dX


def calc_loss():
    u = run_network(x, t, trainable_vars)
    u1 = u[:, 0:1]
    u2 = u[:, 1:]
    deltas1 = -u2_t(x, t) - laplacian1(x, t) - 2 * u1 * (tf.pow(u1, 2) + tf.pow(u2, 2))
    deltas2 = u1_t(x, t) - laplacian2(x, t) - 2 * u2 * (tf.pow(u1, 2) + tf.pow(u2, 2))
    deltas = [deltas1, deltas2]
    deltas = tf.concat([deltas1, deltas2], 1)
    squared_deltas = tf.square(deltas)
    jot1 = tf.square(u - ic)
    # jot2=tf.square(u) #boundary condition not necessary
    error = squared_deltas + jot1 * indices_zero_one  # +jot2*indices1
    loss = tf.reduce_mean(error)
    return loss


def solution(x, t):
    sol1 = tf.cos(2 * x - 3 * t + np.pi / 2) * 1 / tf.cosh(x - 4 * t)
    sol2 = -tf.sin(2 * x - 3 * t + np.pi / 2) * 1 / tf.cosh(x - 4 * t)
    solution = tf.concat([sol1, sol2], axis=1)
    return solution


#%%
################ parameters ################
n = int(10000)
# layer dim:
s = 10
p = 50
m = 100
q = 100
r = 100
ni = 0.001  # optimizer Adam
pu = 0.5  # initial condition percentage of points
stdev = 0.1
mean = 0

x, t, indices_zero_one = create_input(n, pu)
sol = solution(x, t)
optimizer = tf.keras.optimizers.Adam(ni)
ic = a(x)  # initial condition
u = run_network(x, t)  # run network to initialise weights

trainable_vars = create_weights(s, p, m, q, r, mean, stdev)

#%%
# Train loop
"""track progress in a loop"""
for i in range(10):
    optimizer.minimize(calc_loss, trainable_vars)
    u = run_network(x, t)
    loss_train = calc_loss()
    if i % 1 == 0:
        u1 = u[:, 0:1]
        umax = tf.reduce_max(u1)
        umin = tf.reduce_min(u1)
        print(
            "iteration:",
            i,
            "loss:",
            loss_train.numpy(),
            "u_max:",
            umax.numpy(),
            "u_min:",
            umin.numpy(),
        )

#%%
""" train until certain loss is reached"""
kk = 0
while loss_train > 0.045:
    kk = kk + 1
    optimizer.minimize(calc_loss, trainable_vars)
    u = run_network(x, t)
    loss_train = calc_loss()
u1 = u[:, 0:1]
u2 = u[:, 1:]
umax = tf.reduce_max(u1)
umin = tf.reduce_min(u1)
print(
    "loss",
    loss_train.numpy(),
    "u1max:",
    umax.numpy(),
    "u1min:",
    umin.numpy(),
    "number of iterations:",
    kk,
)
error = u - sol
sq_error = tf.sqrt(tf.reduce_mean(tf.square(error)))
abs_error = tf.reduce_mean(tf.abs(error))
inf_error = tf.reduce_max(tf.abs(error))
print(
    "l2:",
    sq_error.numpy(),
    "mean abs:",
    abs_error.numpy(),
    "l infty:",
    inf_error.numpy(),
)

#%%
########### plotting ##########
# all or a random subset of points (plotting many points is slow)
xn = x.numpy()
tn = t.numpy()
random1000 = np.random.randint(0, n, size=int(0.5 * n))

new_x = xn[random1000]
new_t = tn[random1000]

newU = u.numpy()
U1 = newU[:, 0]
new_u1 = U1[random1000] # the real part

U2 = newU[:, 1]
new_u2 = U2[random1000] # the imaginary part

ax = plt.axes(projection="3d")

for data in zip(new_x, new_t, new_u1):  # (xs,ts,newU1)(new_x,new_t,new_u1)
    one, two, three = data
    ax.scatter(one, two, three)

ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("Re(u)")
plt.show()


#%%
############ plot the actual solution ###########

xs = list(x)
ts = list(t)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
real_sol = list(sol.numpy()[:, 0])


for data in zip(xs, ts, real_sol):
    one, two, three = data
    ax.scatter(one, two, three)

ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("Re(u)")
# ax.view_init(30, 45)
# for angle in range(0, 360):
#    ax.view_init(30, angle)
#    plt.show()
#    plt.pause(.001)

plt.show()

#%%
############ saving variables ############

with open("1000it", "wb") as f:
    pickle.dump([trainable_vars, x, t, indices_zero_one], f)
