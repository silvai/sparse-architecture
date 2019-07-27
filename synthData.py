import sys
import csv
from numpy import random
import numpy as np

sums = []
products = []
filename = "../data/testData.csv"
file_A = "../data/data_A.csv"

rows = 1000
cols = 5
num_terms = 3

if len(sys.argv) >= 2 and len(sys.argv[1]) >= 3:
    rows = int(sys.argv[1][0])
    cols = int(sys.argv[1][2:])
if len(sys.argv) >= 3:
    num_terms = int(sys.argv[2])
if len(sys.argv) >= 4:
    output_data_size = int(sys.argv[3]) + rows

output_data = []
input_data = random.rand(rows, cols)
# A = random.rand(500, rows)
A = np.array([[2, 3, 4, 5, 4],
              [4, 4, 3, 2, 3],
              [6, 2, 7, 6, 7],
              [6, 4, 8, 6, 8],
              [8, 6, 8, 1, 8]])

for i in range(len(input_data)):
    output_data.append(np.matmul(A, input_data[i]))

output_data = np.array(output_data)

print("Input Data\n", input_data)
print("----------------------")
print("A\n", A)
print("----------------------")
print("Output Data\n", output_data)
print("----------------------")
print(output_data.shape, " = ", A.shape, " x ", input_data.shape)


in_out_data = []
for i in range(len(input_data)):
    in_out_data.append([input_data[i], output_data[i]])

with open(filename, 'w') as csv_file:
    writer = csv.writer(csv_file)
    # for i in range(len(input_data)):
    #     # writer.writerow((input_data[i], output_data[i]))
    for x in input_data:
        writer.writerow(x)
    for y in output_data:
        writer.writerow(y)
print('Data writen to ', filename)

with open(file_A, 'w') as csv_file:
    writer = csv.writer(csv_file)
    for a in A:
        writer.writerow(a)
