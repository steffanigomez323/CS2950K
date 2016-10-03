from __future__ import division
import struct as sys
import numpy as np

def softmax(e, layer):
	numerator = np.exp(e)
	denominator = 0
	for i in range(10):
		denominator += np.exp(layer[i][0])
	return numerator / denominator

def error(pc):
	return -1 * np.log(pc)

def partialElogits(c, probabilities):
	elogits = np.matrix([0] * 10, dtype=float)
	elogits.shape = (10, 1)
	for i in range(10):
		if i == c:
			elogits[i, 0] = probabilities[i][0] - 1
		else:
			elogits[i, 0] = probabilities[i][0]
	return elogits

def partialEweights(img, partial):
	weights = np.matrix([0] * 10 * 784, dtype=float)
	weights.shape = (10, 784)
	for i in range(10):
		weights[i] = partial[i][0] * img
	return weights


i = open('train-images-idx3-ubyte', 'rb')
l = open('train-labels-idx1-ubyte', 'rb')

weights = np.matrix([0] * 7840, dtype=float)
weights.shape = (10, 784)
betas = np.matrix([0] * 10, dtype=float)
betas.shape = (10, 1)
learningrate = 0.5

byts = i.read(16)
header = l.read(8)

imgs = []
labls = []

for count in range(60000):
	byts = i.read(784)
	labl = l.read(1)
	img1 = sys.unpack('784B', byts)
	img = np.array(img1, dtype=float) / 255.0
	imgs.append(img)
	label1 = sys.unpack('1B', labl)
	label = np.array(label1, dtype=float)
	labls.append(label)

i.close()
l.close()

sample = np.random.choice(60000, size=60000, replace=False)

for count in range(60000):
	if count % 1000 == 0:
		print count
	img = imgs[sample[count]]
	label = labls[sample[count]]
	img.shape = (784, 1)
	hiddenlayer = np.mat(weights) * np.mat(img)
	hiddenlayer += np.mat(betas)
	probabilities = np.matrix([0] * 10, dtype=float)
	probabilities.shape = (10, 1)
	for x in range(10):
		probabilities[x][0] = (softmax(hiddenlayer[x][0], hiddenlayer))
	e = error(probabilities[int(label[0])])
	partiallogits = partialElogits(int(label[0]), probabilities)
	img.shape = (1, 784)
	partialweights = partialEweights(img, partiallogits)
	weights += learningrate * -1 * partialweights
	betas += learningrate * -1 * partiallogits

t = open('t10k-images-idx3-ubyte', 'rb')
tl = open('t10k-labels-idx1-ubyte', 'rb')

modellabels = []
labels = []

print "testing"

byts = t.read(16)
header = tl.read(8)

for count in range(10000):
	if count % 1000 == 0:
		print count
	byts = t.read(784)
	labl = tl.read(1)
	label1 = sys.unpack('1B', labl)
	label = np.array(label1, dtype=float)
	labels.append(label[0])
	img1 = sys.unpack('784B', byts)
	img = np.array(img1, dtype=float) / 255.0
	img.shape = (784, 1)
	hiddenlayer = np.mat(weights) * np.mat(img)
	hiddenlayer += np.mat(betas)
	probabilities = []
	for i in range(10):
		probabilities.append(softmax(hiddenlayer[i][0], hiddenlayer))
	maximum = max(probabilities)
	index = probabilities.index(maximum)
	modellabels.append(index)

correct = 0

for x in range(10000):
	if modellabels[x] == labels[x]:
		correct += 1

print correct / 10000

