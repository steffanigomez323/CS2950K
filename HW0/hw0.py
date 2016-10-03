import struct as sys
import numpy as np

i = open('train-images-idx3-ubyte', 'rb')
l = open('train-labels-idx1-ubyte', 'rb')
byts = i.read(16)
l.read(8)
images06 = []
images013 = []
images16 = []
images113 = []
images26 = []
images213 = []
images36 = []
images313 = []
images46 = []
images413 = []
images56 = []
images513 = []
images66 = []
images613 = []
images76 = []
images713 = []
images86 = []
images813 = []
images96 = []
images913 = []
for count in range(60000):
	byts = i.read(784)
	labl = l.read(1)
	img1 = sys.unpack('784B', byts)
	img = np.array(img1, dtype=float)
	label1 = sys.unpack('1B', labl)
	label = np.array(label1, dtype=float)
	#imagematrix = [img[0:28], img[28:56], img[56:84], img[84:112], img[112:140], img[140:168], img[168:196], img[196:224], img[224:252], img[252:280], img[280:308], img[308:336], img[336:364], img[364:392], img[392:420], img[420:448], img[448:476], img[476:504], img[504:532], img[532:560], img[560:588], img[588:616], img[616:644], img[644:672], img[672:700], img[700:728], img[728:756], img[756:784]]
	if label[0] == 0:
		images06.append(img[144])
		images013.append(img[347])
	elif label[0] == 1:
		images16.append(img[144])
		images113.append(img[347])
	elif label[0] == 2:
		images26.append(img[145])
		images213.append(img[348])
	elif label[0] == 3:
		images36.append(img[145])
		images313.append(img[348])
	elif label[0] == 4:
		images46.append(img[145])
		images413.append(img[348])
	elif label[0] == 5:
		images56.append(img[145])
		images513.append(img[348])
	elif label[0] == 6:
		images66.append(img[145])
		images613.append(img[348])
	elif label[0] == 7:
		images76.append(img[145])
		images713.append(img[348])
	elif label[0] == 8:
		images86.append(img[145])
		images813.append(img[348])
	elif label[0] == 9:
		images96.append(img[145])
		images913.append(img[348])
print "digit 0 average (6, 6): " + str(np.average(images06))
print "digit 0 average (13, 13): " + str(np.average(images013))
print "digit 1 average (6, 6): " + str(np.average(images16))
print "digit 1 average (13, 13): " + str(np.average(images113))
print "digit 2 average (6, 6): " + str(np.average(images26))
print "digit 2 average (13, 13): " + str(np.average(images213))
print "digit 3 average (6, 6): " + str(np.average(images36))
print "digit 3 average (13, 13): " + str(np.average(images313))
print "digit 4 average (6, 6): " + str(np.average(images46))
print "digit 4 average (13, 13): " + str(np.average(images413))
print "digit 5 average (6, 6): " + str(np.average(images56))
print "digit 5 average (13, 13): " + str(np.average(images513))
print "digit 6 average (6, 6): " + str(np.average(images66))
print "digit 6 average (13, 13): " + str(np.average(images613))
print "digit 7 average (6, 6): " + str(np.average(images76))
print "digit 7 average (13, 13): " + str(np.average(images713))
print "digit 8 average (6, 6): " + str(np.average(images86))
print "digit 8 average (13, 13): " + str(np.average(images813))
print "digit 9 average (6, 6): " + str(np.average(images96))
print "digit 9 average (13, 13): " + str(np.average(images913))
