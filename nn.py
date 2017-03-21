import numpy as np 

x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0],[1],[1],[0]])
w1 = np.random.normal(size=(3,5))
w2 = np.random.normal(size=(5,1))
lr = 1e-2

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
	return sigmoid(x)*(1.0 - sigmoid(x))

for epoch in range(60000):
	h = x.dot(w1)
	h_activated = sigmoid(h)
	y_hat = h_activated.dot(w2)

	loss = np.mean(0.5 * np.square(y_hat - y))
	if epoch % 5000 == 0:
		print ('epoch: {}	loss: {}'.format(str(epoch), str(loss)))

	grad_y_hat = (y_hat - y)
	grad_w2 = h_activated.T.dot(grad_y_hat)
	grad_w1 = x.T.dot(sigmoid_prime(h) * (np.dot(grad_y_hat, w2.T)))

	w1 -= lr * grad_w1
	w2 -= lr * grad_w2

print y_hat