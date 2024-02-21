# Activation-Function
1.	Theoretical Understanding: 
Explain the Activation Function, including its equation and graph.
Ans: In an artificial neural network, an activation function of a node is a function that calculates the output of the node based on its individual inputs and their weights.
There are various activation functions used in neural networks, each with its own characteristics. Some common activation functions include Sigmoid, ReLU (Rectified Linear Unit), tanh (Hyperbolic Tangent), and SoftMax.
I will be discussing the sigmoid activation function. It is defined as: σ(x) = 1 / (1 + e-x)
Where:
x is the input to the function.
e is the base of the natural logarithm (Euler's number, approximately equal to 2.71828).
The graph of the sigmoid function is an "S"-shaped curve that ranges from 0 to 1, as shown below:
![image](https://github.com/farahirn99/Activation-Function/assets/89274209/5dc9eb52-ef19-4c3b-bcc3-aca006063ec8)

 
What is an activation function and why do we use them?
Ans: The activation function decides whether a neuron should be activated or not by calculating the weighted sum and further adding bias to it. The purpose of the activation function is to introduce non-linearity into the output of a neuron.
In a neural network, we would update the weights and biases of the neurons based on the error at the output. This is known as back-propagation. Activation functions make the back-propagation possible since the gradients are supplied along with the error to update the weights and biases.

2.	Mathematical Exploration:
Derive the activation function formula and demonstrate its output range.
Ans: (i) Sigmoid activation function. It is defined as: σ(x) = 1 / (1 + e-x)
Where:
x is the input to the function.
e is the base of the natural logarithm (Euler's number, approximately equal to 2.71828).
The graph of the sigmoid function is an "S"-shaped curve that ranges from 0 to 1, as shown below:
![image](https://github.com/farahirn99/Activation-Function/assets/89274209/952214e2-f10c-4a1e-b6fb-6c55e1ee66a6)

(ii) Hyperbolic Tangent Function (tanh) is defined as: tanh(x) = (ex + e-x ) / (ex - e-x ). It ranges from -1 to +1. The graph looks like this: 
![image](https://github.com/farahirn99/Activation-Function/assets/89274209/19428d3f-ca80-44ca-bfa2-cbe4cf901342)

(iii) Rectified linear Unit (ReLU) function is defined as f(x) = max(0, x). It ranges from 0 to ∞. The graph looks like this:
![image](https://github.com/farahirn99/Activation-Function/assets/89274209/2ec8dec4-88fb-4e5d-a816-2d6c35243f43)


Calculate the derivative of the Activation function and explain its significance in the backpropagation process.
Ans: Sigmoid activation function equation: σ(x) = 1 / (1 + e-x)
Derivative: d/dx(σ(x)) =σ(x)(1−σ(x))
Activation functions play a vital role in the backpropagation process of neural networks, enabling the computation of gradients and the adjustment of weights to minimize the error between predicted and actual outputs. Each activation function has its characteristics and significance in training neural networks, influencing the learning dynamics and performance of the model.


4.	Analysis:
Analyze the advantages and disadvantages of using the Activation Function in neural networks.
Ans: Advantages:

Introducing Non-linearity: Activation functions help neural networks approximate complex, non-linear functions. Without activation functions, neural networks would become linear transformations, limiting their training and modelling of complex data patterns.


Gradient Propagation: Activation functions provide derivatives to calculate and propagate gradients backward through the network during training. This allows gradient-based optimisation algorithms like stochastic gradient descent to efficiently optimise network parameters like weights and biases.

Output Range Control: Activation functions can limit neuron output, which can be useful depending on the task. Sigmoid and SoftMax functions normalise outputs to (0, 1) for binary and multi-class classification, respectively.

Sparsity and Efficiency: ReLU and its variants set negative values to zero to induce neural network sparsity. Sparsity improves computation and memory usage, especially in deep networks with many neurons.

Disadvantages:

Vanishing and Exploding Gradients: Deep networks train sigmoid and tanh activation functions, which can cause gradients to vanish or explode. In early network layers, vanishing gradients hinder learning, while exploding gradients cause numerical instability.

Dead Neurons: Neurons can become inactive (output zero) for all inputs during training due to the "dying ReLU" problem. This occurs when a ReLU neuron receives consistently negative input, resulting in a zero gradient, and does not update its weights during training.

Not Zero-Centered: ReLU and its variants produce outputs that are not zero-centered. Since negative input gradients are always zero, this can cause the vanishing gradient problem.

Limited Output Range: Some activation functions have limited output ranges, making them unsuitable for some tasks. ReLU only outputs positive values, which may not be ideal for negative-value tasks.

Discuss the impact of the Activation function on gradient descent and the problem of vanishing gradients.
Ans: 1. Impact on Gradient Descent:

Gradient descent is a fundamental optimization algorithm used to train neural networks by adjusting the weights and biases based on the gradients of the loss function with respect to these parameters. The gradients are calculated using the chain rule during backpropagation.

The activation function plays a crucial role in determining the shape and smoothness of the loss function. Smooth and well-behaved activation functions lead to smooth loss surfaces, making it easier for gradient descent to find the global or local minima efficiently. Activation functions with large regions of non-linearity help the model learn complex patterns in the data.

Additionally, the choice of activation function affects the stability and convergence speed of gradient descent. Activation functions with well-defined derivatives and bounded outputs facilitate stable and efficient training.

2. Problem of Vanishing Gradients:

The vanishing gradient problem arises when gradients propagated backward through the network during training become extremely small, causing the weights in earlier layers to update very slowly or not at all. This problem is particularly prevalent in deep neural networks with many layers, especially when using certain activation functions.

The choice of activation function significantly influences the occurrence of vanishing gradients:

Sigmoid and Tanh Functions: These functions have saturating gradients, meaning that their derivatives approach zero as the absolute value of the input becomes large. Consequently, during backpropagation, the gradients can vanish as they are multiplied across many layers, especially in deep networks, impairing the training of early layers.

ReLU and its Variants: Rectified Linear Unit (ReLU) and its variants have non-saturating gradients for positive inputs, which helps mitigate the vanishing gradient problem to some extent. However, ReLU suffers from the "dying ReLU" problem, where neurons can become inactive (output zero) for all inputs during training, leading to sparse activations and potential information loss.

Impact of Activation Function on Mitigating Vanishing Gradients:

Choosing appropriate activation functions can help alleviate the vanishing gradient problem:
ReLU and Leaky ReLU: ReLU and its variants have non-saturating gradients for positive inputs, which help prevent the vanishing gradient problem. Leaky ReLU allows a small, non-zero gradient for negative inputs, further mitigating the dying ReLU problem.

Initialization Techniques: Initialization techniques like Xavier/Glorot initialization can help mitigate vanishing gradients by initializing weights appropriately, considering the characteristics of the chosen activation function.




