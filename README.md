# Three-layered-neural-network
three layer feedforward neural network using only numpy

there is a dataset of different type of wine with different types of attribute.
there are 178 wine types with the different cultivar.but at first we dont know which cultivar made them.

we will build a classifier that recognizes the wine based on 13 attributes of the wine.
The input layer (x) consists of 178 neurons.
A1, the first layer, consists of 8 neurons.
A2, the second layer, consists of 5 neurons.
A3, the third and output layer, consists of 3 neurons.

While training neural network, we should propagate forward in Neural Network.
and while detecting error for getting accurate results, we should propagate backward in reverse directions in Neural Network.
so this is implemented by defining two fuctions that is , forward_propagation and backward_propagation.


Libraries used:
NumPy, skicit-learn, pandas

functions used:
random.seed
