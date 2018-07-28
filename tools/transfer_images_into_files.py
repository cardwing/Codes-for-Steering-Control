from PIL import Image

mae_error = []
num = 0
with open("mae_testing.txt", "r") as f:
    for item in f.readlines():
        mae_error.append(float(item) * 1.0 * 180 / 3.1415)
        num = num + 1

image_name = []
with open("image_name_testing.txt", "r") as g:
    for item in g.readlines():
        image_name.append(item.strip())

count = 0
for i in range(num):
    if mae_error[i] >= 10:
        im = Image.open(image_name[i])
        if mae_error[i] > 30:
            im.save('bad_predict_images_30_180/' + image_name[i][-24:])
        if mae_error[i] <= 30:
            im.save('bad_predict_images_10_30/' + image_name[i][-24:])   
        count = count + 1
        print count

