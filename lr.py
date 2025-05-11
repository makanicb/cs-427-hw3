#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
from math import log
from math import exp
from math import sqrt

MAX_ITERS = 100


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        example = [int(x) for x in p.split(l.strip())]
        x = example[0:-1]
        y = example[-1]
        data.append((x, y))
    return (data, varnames)


def sigmoid_function(z):
    return 1 / (1 + exp(-z))


def log_loss(y, y_hat):
    return log((1+ exp( -(y * y_hat)))) / log(2)


def vec_norm(vec):
    magnitude = vec_mag(vec)
    if magnitude == 0:
        return [0] * len(vec)
    return vec_scale(vec, 1 / magnitude)


def vec_scale(vec, scalar):
    return [i * scalar for i in vec]


def vec_add(vec_a, vec_b):
    return [a+b for a, b in zip(vec_a, vec_b)]


def vec_sub(vec_a, vec_b):
    return [a-b for a, b in zip(vec_a, vec_b)]


def vec_mag(vec):
    return sqrt(dot(vec, vec))


def compute_gradient_of_weights(x, w, y_hat, y, l2_reg_weight):
    num_examples = len(x)
    nabla = [0] * len(w)
    for i in range(num_examples):
        error = y_hat[i] - y[i]
        scaled_input = vec_scale(x[i], error)
        nabla = vec_add(nabla, scaled_input)

    nabla = vec_scale(nabla, 1 / num_examples)
    penalty = vec_scale(w, l2_reg_weight)
    nabla = vec_add(nabla, penalty)
    return nabla


def compute_gradient_of_bias(y_hat, y):
    num_examples = len(y)
    nabla = 0
    for i in range(num_examples):
        error = y_hat[i] - y[i]
        nabla += error
    return nabla / num_examples


def get_examples_from_data(data):
    return [example for (example, label) in data]


def get_labels_from_data(data):
    return [label for (example, label) in data]


def make_predictions(model, examples):
    return [predict_lr(model, example) for example in examples]


# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
    numvars = len(data[0][0])
    w = [0.0] * numvars
    b = 0.0
    examples = get_examples_from_data(data)
    labels = get_labels_from_data(data)

    for iteration in range(MAX_ITERS):
        model = (w, b)
        # compute predictions
        predictions = make_predictions(model, examples)
        # compute gradient of weights
        nabla_w = compute_gradient_of_weights(examples, w, predictions, labels, l2_reg_weight)
        # compute gradient of bias
        nabla_b = compute_gradient_of_bias(predictions, labels)
        # update weights
        nabla_w = vec_scale(nabla_w, eta)
        w = vec_sub(w, nabla_w)
        # update bias
        b -= eta * nabla_b
        # if gradient was small (<0.0001) then stop
        if vec_mag(nabla_w) < 0.0001: break

    return (w, b)


def dot(vec_a, vec_b):
    sum = 0.0
    for a, b in zip(vec_a, vec_b):
        sum += a * b
    return sum


# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model
    z = dot(x, w) + b
    return sigmoid_function(z)
    # return 0.5 # This is a random probability, fix this according to your solution


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 5):
        print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    eta = float(argv[2])
    lam = float(argv[3])
    modelfile = argv[4]

    # Train model
    (w, b) = train_lr(train, eta, lam)

    # Write model file
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in test:
        prob = predict_lr((w, b), x)
        print(prob)
        if (prob - 0.5) * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
