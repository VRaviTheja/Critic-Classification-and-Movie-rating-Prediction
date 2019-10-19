import numpy as np
import pandas as pd

def train_nn(X, y, num_iterations, D_s):
	D = D_s # number of features
	K = 4 # number of classes

	h = 10 # size of hidden layer
	W = 0.01 * np.random.randn(D,h)
	b = np.zeros((1,h))
	W2 = 0.01 * np.random.randn(h,K)
	b2 = np.zeros((1,K))

	# hyperparameters
	step_size = 1e-0 # learning rate
	reg = 1e-3 # regularization

	# gradient descent loop
	num_examples = X.shape[0]
	for i in range(num_iterations):
		hidden_layer = np.maximum(0, np.dot(X, W) + b)
		scores = np.dot(hidden_layer, W2) + b2

		exp_scores = np.exp(scores)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

		# compute the loss
		correct_logprobs = -np.log(probs[range(num_examples),y])
		data_loss = np.sum(correct_logprobs)/num_examples
		reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
		loss = data_loss + reg_loss
		print("iteration %d: loss %f" % (i, loss))

		dscores = probs
		dscores[range(num_examples),y] -= 1
		dscores /= num_examples

		dW2 = np.dot(hidden_layer.T, dscores)
		db2 = np.sum(dscores, axis=0, keepdims=True)
		dhidden = np.dot(dscores, W2.T)
		dhidden[hidden_layer <= 0] = 0
		dW = np.dot(X.T, dhidden)
		db = np.sum(dhidden, axis=0, keepdims=True)
		# regularization
		dW2 += reg * W2
		dW += reg * W

		# parameter update
		W += -step_size * dW
		b += -step_size * db
		W2 += -step_size * dW2
		b2 += -step_size * db2

	return (W,b,W2,b2)

def predict(X, W, b, W2, b2):
	hidden_layer = np.maximum(0, np.dot(X, W) + b)
	scores = np.dot(hidden_layer, W2) + b2
	predicted_class = np.argmax(scores, axis=1)
	return predicted_class

def get_accuracy(y_true, y_preds):
	print(np.mean(y_true == y_preds))

def save_weights(W,b,W2,b2):
	np.savetxt('W.txt', W, fmt='%f')
	np.savetxt('b.txt', b, fmt='%f')
	np.savetxt('W2.txt', W2, fmt='%f')
	np.savetxt('b2.txt', b2, fmt='%f')

def load_weiehgts():
	W = np.loadtxt('W.txt', dtype=float)
	b = np.loadtxt('b.txt', dtype=float)
	W2 = np.loadtxt('W2.txt', dtype=float)
	b2 = np.loadtxt('b2.txt', dtype=float)

	return (W,b,W2,b2)


