import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add header names
headers = ['age', 'sex', 'chest_pain', 'resting_blood_pressure', 'serum_cholestoral', 
           'fasting_blood_sugar', 'resting_ecg_results', 'max_heart_rate_achieved', 
           'exercise_induced_angina', 'oldpeak', 'slope_of_the_peak', 'num_of_major_vessels', 
           'thal', 'heart_disease']

# Load the dataset
heart_df = pd.read_csv("heart.csv", sep=',', names=headers)

# Convert input to numeric values (if any non-numeric values exist)
heart_df = heart_df.apply(pd.to_numeric, errors='coerce')

# Handle missing values (if any)
heart_df.fillna(heart_df.mean(), inplace=True)

# Convert input data to numpy arrays
X = heart_df.drop(columns=['heart_disease'])

# Replace target class with 0 and 1
heart_df['heart_disease'] = heart_df['heart_disease'].replace({1: 0, 2: 1})

y_label = heart_df['heart_disease'].values.reshape(-1, 1)

# Split data into train and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.2, random_state=2)

# Standardize the dataset
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

print(heart_df.dtypes)
print(f"\nShape of train set is {Xtrain.shape}")
print(f"\nShape of test set is {Xtest.shape}")
print(f"\nShape of train label is {ytrain.shape}")
print(f"\nShape of test labels is {ytest.shape}")

# Neural Network Class
class NeuralNet:
    def __init__(self, layers=[13, 8, 1], learning_rate=0.001, iterations=1000):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.layers = layers
        self.X = None
        self.y = None

    def init_weights(self):
        np.random.seed(1)
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1]) * 0.01
        self.params['b1'] = np.zeros((1, self.layers[1]))
        self.params['W2'] = np.random.randn(self.layers[1], self.layers[2]) * 0.01
        self.params['b2'] = np.zeros((1, self.layers[2]))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def dSigmoid(self, Z):
        sig = self.sigmoid(Z)
        return sig * (1 - sig)

    def relu(self, Z):
        return np.maximum(0, Z)

    def dRelu(self, Z):
        return (Z > 0).astype(float)

    def eta(self, x):
        return np.maximum(x, 1e-10)

    def entropy_loss(self, y, yhat):
        epsilon = 1e-10
        yhat = np.clip(yhat, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
        return loss

    def forward_propagation(self):
        Z1 = self.X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        yhat = self.sigmoid(Z2)
        loss = self.entropy_loss(self.y, yhat)

        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1
        return yhat, loss

    def back_propagation(self, yhat):
        m = self.X.shape[0]
        y_inv = 1 - self.y
        yhat_inv = 1 - yhat

        dl_wrt_yhat = (yhat - self.y) / m
        dl_wrt_sig = self.dSigmoid(self.params['Z2'])
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)

        dl_wrt_z1 = dl_wrt_A1 * self.dRelu(self.params['Z1'])
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)

        self.params['W1'] -= self.learning_rate * dl_wrt_w1
        self.params['W2'] -= self.learning_rate * dl_wrt_w2
        self.params['b1'] -= self.learning_rate * dl_wrt_b1
        self.params['b2'] -= self.learning_rate * dl_wrt_b2

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.init_weights()

        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)

            # Learning rate decay (optional)
            if i % 100 == 0 and i != 0:
                self.learning_rate *= 0.9  # Decay learning rate

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, X):
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        yhat = self.sigmoid(Z2)
        return np.round(yhat)

    def acc(self, y, yhat):
        return np.mean(y == yhat) * 100

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss curve during training")
        plt.show()

# Train the model
nn = NeuralNet(layers=[13, 8, 1], learning_rate=0.001, iterations=1000)
nn.fit(Xtrain, ytrain)

# Evaluate the model
y_pred = nn.predict(Xtest)
accuracy = nn.acc(ytest, y_pred)

print(f"Test Accuracy: {accuracy}%")

# Plot the loss curve
nn.plot_loss()

