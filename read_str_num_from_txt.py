mae = 0.0
num = 0
item_list = []
image_mae = []
item_str = ''
with open("valid-predictions-epoch14.txt", "r") as out:
    for line in out.readlines():
        num_data = 0
        image_name = []
        item_list = line.split(" ")
        len_str = len(item_list)
        for i in range(len_str):
            if len(item_list[len_str - i - 1]) != 0: # len('') = 0
                num_data = num_data + 1
            if num_data == 2:
                image_mae.append(item_list[len_str - i - 1])
                break
        print item_list[len_str - i - 1]
        num = num + 1

print "num:%d" %num
