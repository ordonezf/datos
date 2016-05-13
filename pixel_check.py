import csv
from sys import argv
import matplotlib.pyplot as plt
import numpy as np

COLS = 784

def discard_pixels(zeroes_in_column):
	test = open("test.csv")
	train = open("train.csv")

	rtest = csv.reader(test)
	rtrain = csv.reader(train)

	next(rtest)
	next(rtrain)

	dic = {}

	for x in range(COLS):
		dic[str(x)] = 0

	for x in rtest:
		pn = 0
		for pixel in x:
			if pixel == '0':
				dic[str(pn)] += 1
			pn += 1

	for x in rtrain:
		pn = 0
		for pixel in x[1:]:
			if pixel == '0':
				dic[str(pn)] += 1
			pn += 1

	dic_new = {}
	for x in dic:
		if dic[x] < zeroes_in_column:
			dic_new[int(x)] = dic[x]

	test.close()
	train.close()
	return ["pixel{}".format(i) for i in sorted(dic_new.keys())]