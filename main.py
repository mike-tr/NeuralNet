# made by mike

import numpy as np
from neuron import Neuron

# this is an AND table, where A, B input and O is output.
data = np.array([
    # A  B  O
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
])

print(data.shape)

# seperate input and output
x = data[:, :-1]
y = data[:, -1:]

print("--------- data ------------")
print("x :")
print(x)
print("y :")
print(y)

print("--------- Training ------------")

# add bias to x, a.k.a add 1 to each input.
x_bias = np.ones((x.shape[0], 1))
x_train = np.c_[x, x_bias]
n = Neuron(x_train, y)

# Traing give the neuron one input and actual output,
# if for all inputs neuron gives correct answer exit.
for i in range(100):
    print("\n-------- run", i, "--------")
    err = 0
    done = True
    for j in range(x_train.shape[0]):
        pred = n.predict(x_train[j])
        actual = y[j]
        print("prediction :", pred, ", actual :", actual[0])
        change = n.train_one(x_train[j], actual)
        if change:
            err += 1
            done = False
    if done:
        print("\n------ training done ------")
        print("training done on run :", i)
        print("final weights :")
        print(n.weights)
        break
    else:
        print("End of run Num of Errors :", err)
