# Neural network project
My neural network projects in python using numpy


In this project i have implemented Adaline perceptron at first,
then i have implemented Neural network class and BackPropogration.

## About the Neural Network
Using numpy we use matrix multiplciations for training and predicting and can predict using full data-sets or fractions
or even single rows, or mini-batches ( same for training )

we are using momentum model, where the Gradient at time T is equal to the Gradient of the error + Momentum * Gradient at time (t - 1)

Demonstation of the neural network.

Given the following data : 
![bd81b72c7e6d74768409e7247ed175ed](https://user-images.githubusercontent.com/48411662/116869007-870e8380-ac18-11eb-857d-732de4e04b39.png)

## About the data
* Data is -1 < x,y < 1
* given [ d = x^2 + y ^2 ] for [ 0.5 < d < 0.75 ] we classify it as 1 and otherwise its 0, hence we get the ring as shown above.
* in the picture Green represent 1's and gray represent 0's.
* the x,y are at resulution of 0.01, hence we have 40k different point in our total data-set ( 200 possitble values per x and y )

## Training set
* we pick a random sample of size 1000, uniform where 50% of the data is 0's and 50% is 1's

### Demonstrate improvement over training ( first 6 iterarions )
![acc](https://user-images.githubusercontent.com/48411662/116870922-18332980-ac1c-11eb-897d-74865513bf13.png)

### About the figure above
* we run 26 iterations at each iteration we have 250 ( mini-iterations ) and plot the average error of them * 1000
* in mini-iteration, we pick a random ordering of the training-set, then we devide the training set to groups of 32 points and train in those 32 batches.
* A.k.a each iteration has seen the training-set at some order 250 times!

### after 26 iterarions
![final](https://user-images.githubusercontent.com/48411662/116883349-3144d600-ac2e-11eb-88e3-933a3cac5b16.png)
#### we can see in the figure the error as well as the actuall predictions


## How to use:
from NeuralNet import NeuralNetwork <br>
// network that 2 inputs and 2 outputs, 2 hidden layers of size 8 and 4. <br>
network = NeuralNetwork(layers = [3, 8, 4, 2], learning_rate = 0.2, momentum = 0.5) <br>

data = some_data() // get some data <br>
## example of data, Y1 = exactly 1 input is 1, Y2 = exactly 2 inputs is 1.
| A | B | C | | Y1 | Y2 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 || 0 | 0 |
| 0 | 0 | 1 || 1 | 0 |
| 0 | 1 | 0 || 1 | 0 |
| 0 | 1 | 1 || 0 | 1 |
| 1 | 0 | 0 || 1 | 0 |
| 1 | 0 | 1 || 0 | 1 |
| 1 | 1 | 0 || 0 | 1 |
| 1 | 1 | 1 || 0 | 0 |

### Seperation
x_data = data[:, :-2]
y_data = data[:, -2:]

## train
for i in range(iterations):
  network.train(x_data, y_data)
  
## predictions
network.predict(x_data)

predicted : <br>
 [[0.02470556 0.02210182] <br>
 [0.97892623 0.00002558] <br>
 [0.97892575 0.00002489] <br> 
 [0.00000639 0.98381658] <br>
 [0.98224752 0.00002111] <br>
 [0.00000734 0.98049643] <br>
 [0.0000073  0.98155121] <br>
 [0.02406604 0.02782214]] <br>
 
 actual :
 [[0 0]
 [1 0]
 [1 0]
 [0 1]
 [1 0]
 [0 1]
 [0 1]
 [0 0]]


