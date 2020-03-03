import numpy as np
from dlgo.nn.load_mnist import load_data
import matplotlib.pyplot as plt
from dlgo.nn.layers import sigmoid_double


def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)

def predict(x, W, b):
    return sigmoid_double(np.dot(W, x) + b)

def evaluate(data, digit, threshold, W, b):
    correct_predictions = 0
    for x in data:
        if (predict(x[0], W, b) > threshold) == (np.argmax(x[1]) == digit):
            correct_predictions += 1
    return correct_predictions / float(len(data))


train, test = load_data()


W_8 = np.transpose(average_digit(train, 8))
b_8 = -45

# print(predict(train[2][0], W_8, b_8))
# print(predict(train[17][0], W_8, b_8))

print(evaluate(data=train, digit=8, threshold=0.5, W=W_8, b=b_8))
print(evaluate(data=test, digit=8, threshold=0.5, W=W_8, b=b_8))
print(evaluate(data=[x for x in test if np.argmax(x[1]) == 8], digit=8, threshold=0.5, W=W_8, b=b_8))
