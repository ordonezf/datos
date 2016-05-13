import matplotlib.pyplot as plt
import pixel_check as pc
import numpy as np
import csv

zeroes_in_column = 70000 * 0.99
DIMENSIONS = 101
TEST = 42000

dims = pc.discard_pixels(zeroes_in_column)

f1 = open("train.csv","r")
f2 = open("test.csv", "r")
r1 = csv.DictReader(f1)
r2 = csv.DictReader(f2)

new_train = open("svd_train.csv", "w")
new_test = open("svd_test.csv", "w")
w1 = csv.writer(new_train)
w2 = csv.writer(new_test)

fields1 = ["label"] + ["col {}".format(x) for x in range(DIMENSIONS)]
fields2 = ["col {}".format(x) for x in range(DIMENSIONS)]

l = []
labels = []
for row in r1:
	labels.append(row["label"])
	l.append([int(row[pixel]) for pixel in dims])
for row in r2:
	l.append([int(row[pixel]) for pixel in dims])

m = np.array(l)

u, s, v = np.linalg.svd(m, full_matrices=False)

w1.writerow(fields1)
for row,label in zip(u[:TEST],labels):
	w1.writerow([label]+[pixel for pixel in row[:DIMENSIONS]])

w2.writerow(fields2)
for row in u[TEST:]:
	w2.writerow([pixel for pixel in row[:DIMENSIONS]])



f1.close()
f2.close()
new_test.close()
new_train.close()
