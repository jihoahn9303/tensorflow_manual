# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 09:03:35 2021

@author: jiho Ahn
"""

import tensorflow as tf
import numpy as np

#print(tf.__version__)

t1 = tf.Variable([1, 2, 3])  # for model
t2 = tf.constant([1, 2, 3])  # for dataset

#print(t1)
#print(t2)
#
#print('===')
#print(type(t1))
#print(type(t2))

# %% tf.constant
#test_list = [1, 2, 3]
#test_np = np.array([1, 2, 3])
#
#t1 = tf.constant(test_list)
#t2 = tf.constant(test_np)
#
#print(t1)
#print(t2)
#
#print(type(t1))
#print(type(t2))

# %%
test_list = [1, 2, 3]
test_np = np.array([1, 2, 3])

t1 = tf.Variable(test_list)
t2 = tf.Variable(test_np)

#print(t1)
#print(t2)
#
#print(type(t1))
#print(type(t2))

# %%
t1 = tf.constant(test_list)
t2 = tf.Variable(test_list)  

t3 = tf.Variable(t1)  # ResourceVariable(variable) -> EagerTensor(constant)

# %%
"""
일반적으로, constant tensor에서 variable tensor로 변환이 가능하다.
하지만, variable tensor에서 constant tensor로 직접적인 변환은 불가능하다.
forward / backward propagation을 통해 weight를 update를 하게 되는데,
이 때, variable을 constant로 변환하는 순간, 이 정보가 다 소멸되기 때문이다.
그러나, convert_to_tensor()를 통해 variable -> constant 변환은 가능하다.
"""
t4 = tf.convert_to_tensor(test_list)  # list -> constant
t5 = tf.convert_to_tensor(test_np)  # numpy -> constant

print(type(t4))
print(type(t5))

t6 = tf.Variable(test_list)
t7 = tf.convert_to_tensor(t6)

print(type(t6))
print(type(t7))

# %%
"""
variable과 variable,
variable과 constant,
constant와 constant 간의 연산 결과는 모두 EagerTensor(constant) 자료형임을 기억하자.
"""
test_list1 = [10, 20, 30]
test_list2 = [1, 2, 3]

t1 = tf.Variable(test_list1)
t2 = tf.Variable(test_list2)

t3 = t1 + t2
print(type(t3))






