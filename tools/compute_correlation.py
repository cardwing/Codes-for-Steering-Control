import numpy as np
mae_error = []
gt = []
index = []
num = 0
num_index = 0
total_num = 0
with open("mae_testing.txt", "r") as f:
    for item in f.readlines():
        if float(item)*180/3.1415 >= 30:
            mae_error.append(float(item))
            index.append(num)
            num = num + 1
print num
total_num = num
num = 0 
with open("seperate_testing_ground_truth_label.txt", "r") as g:
    for item in g.readlines():
        if index[num_index] == num:
            num_index = num_index + 1
            gt.append(abs(float(item)))
        if num_index == total_num:
            break
        num = num + 1

print np.corrcoef(mae_error, gt)
