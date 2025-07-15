import numpy as np
import matplotlib.pyplot as plt

# Input range
x = np.linspace(-5, 5, 100)

# Manually define activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def relu(z):
    return np.where(z > 0, z, 0)

def identity(z):
    return z

def softmax(z):
    exp_values = np.exp(z)
    return exp_values / np.sum(exp_values)  # Softmax usually applied to vectors; here we apply it globally for plotting

# Calculate values
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_identity = identity(x)
y_softmax = softmax(x)  # Note: In normal softmax usage, it’s for vectors, here we plot it for understanding.

# Plot all activations
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.plot(x, y_tanh, label='Tanh')
plt.plot(x, y_relu, label='ReLU')
plt.plot(x, y_identity, label='Identity')
plt.plot(x, y_softmax, label='Softmax')

plt.xlabel('Input')
plt.ylabel('Activation')
plt.title('Activation Functions ')
plt.legend()
plt.grid(True)
plt.show()

