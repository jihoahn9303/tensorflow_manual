# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:27:56 2021

@author: jiho Ahn
"""

"""
tf.GradientTape()
연산 그래프를 정의하고, 
이에 따라 자동으로 back propagation을 수월하게 수행하도록 하는 기능.
back propagation을 위해서는, 우선 forward propagation이 필요하다.
이 때, model layer 부터, prediction을 통해 loss 값을 구하는 단계까지
forward propagation한 결과 값들을 저장함으로써, 이를 재사용 할 수 있게 한다.
"""
import tensorflow as tf
import matplotlib.pyplot as plt

t1 = tf.Variable([1, 2, 3], dtype=tf.float32)
t2 = tf.Variable([10, 20, 30], dtype=tf.float32)

# 연산 기록 저장
with tf.GradientTape() as tape:
    t3 = t1 * t2
    t4 = t3 + t2
    
print(t1.numpy())
print(t2.numpy())
print(t3.numpy())
print(t4.numpy())

# 연산 기록에 대해, 편미분 실시
gradients = tape.gradient(t4, [t1, t2, t3])  # dt4/dt1, dt4/dt2, dt4/dt3
#print(gradients)
print("dt1 : ", gradients[0])
print("dt2 : ", gradients[1])
print("dt3 : ", gradients[2])

# %%
t1 = tf.constant([1, 2, 3], dtype=tf.float32)
t2 = tf.Variable([10, 20, 30], dtype=tf.float32)

with tf.GradientTape() as tape:
    t3 = t1 * t2
    
gradients = tape.gradient(t3, [t1, t2]) 

print("dt1 : ", gradients[0])
print("dt2 : ", gradients[1])

# %%
# 간단한 linear regression

x_data = tf.random.normal(shape=(1000,), dtype=tf.float32)
y_data = 3*x_data + 1

w = tf.Variable(-1.)
b = tf.Variable(-1.)

learning_rate = 0.01
w_trace, b_trace = [], []

for x, y in zip(x_data, y_data):
    with tf.GradientTape() as tape:   
        prediction = w*x + b  # model object
        loss = (prediction - y) ** 2  # loss function
        
    gradients = tape.gradient(loss, [w, b])
    
    w_trace.append(w.numpy())
    b_trace.append(b.numpy())
    w = tf.Variable(w - learning_rate * gradients[0])
    b = tf.Variable(b - learning_rate * gradients[1])
    
 # %%
fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(w_trace, label='weight')   
ax.plot(b_trace, label='bias') 

ax.tick_params(labelsize=20)
ax.legend(fontsize=30)
ax.grid()