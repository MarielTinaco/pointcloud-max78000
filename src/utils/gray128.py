from PIL import Image, ImageOps
import numpy as np
import os
import sys
import cv2

im = (Image.open('dataset/DepthMap/monitor/1669184496_C.png'))
# im = (Image.open('dataset/DepthMap/chair/1669185773_C.png'))
size = (64*2,64*2)
img = im.resize(size)
#img = ImageOps.fit(im, size, Image.ANTIALIAS)
np1_img = np.array(img)
new_img = Image.fromarray(np1_img)
new_img.save("dataset/DepthMap/new_img.png")

src = cv2.imread('dataset/DepthMap/new_img.png', cv2.IMREAD_UNCHANGED)

# In case of grayScale images the len(img.shape) == 2
if len(src.shape) > 2 and src.shape[2] == 4:
    #convert the image from RGBA2RGB
    # src = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
    src = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)

# #Get red channel from img
# red = src[:,:,2]
# red = red - 128
# red_f = red.flatten()

# #Get green channel from img
# green = src[:,:,1]
# green = green - 128
# green_f = green.flatten()

# #Get blue channel from img
# blue = src[:,:,0]
# blue = blue - 128
# blue_f = blue.flatten()

#Get blue channel from img

width = src.shape[0]
height = src.shape[1]

assert width % 4 == 0, "Width must be divisible by 4"
assert height % 4 == 0, "Height must be divisible by 4"

empty = np.zeros(shape=(int(width/4), int(height)), dtype=np.int64)

arra = np.full(shape=src.shape, fill_value=128)
norm = src - arra

norm_f = norm.flatten()

new_img = []
storage = []

for index, pixel in enumerate(norm_f):
    storage.append(pixel)
    if (index + 1) % 4 == 0:
        new_img.append(storage)
        storage = []
        # input()

arr_result = []

for array in new_img:
    first = int(array[3]) & 0xFF
    second = int(array[2]) & 0xFF
    third = int(array[1]) & 0xFF
    fourth = int(array[0]) & 0xFF

    result = first | second << 8 | third << 16 | fourth << 24

    arr_result.append(result)    

out_arr_result = np.asarray(arr_result, dtype=np.uint32)

# print(gray)

# gray = gray - 128
# gray_f = gray.flatten()

# arr_result = []

# # 0x00bbggrr
# for i in range(len(gray_f)):
# 	# result = red_f[i] | blue_f[i]<<8 | green_f[i]<<16
# 	arr_result.append((result))

#convert list to numpy array
# out_arr_result = np.asarray(arr_result, dtype=np.uint32)


#Write out data to the header file
with open('C:/Users/MTinaco/Dev/ai8x-synthesis/sdk/Examples/MAX78000/CNN/depthmap/sampledata.h', 'w') as outfile:
	outfile.write('#define SAMPLE_INPUT_0 { \\')
	outfile.write('\n')

	for i in range(len(out_arr_result)):
		if i==0:
			outfile.write('\t0x{0:08x},\t'.format((out_arr_result[i])))

		else :
			d = i%8
			if(d!=0):
				outfile.write('0x{0:08x},\t'.format((out_arr_result[i])))
			else:
				outfile.write('\\')
				outfile.write('\n\t')
				outfile.write('0x{0:08x},\t'.format((out_arr_result[i])))

	outfile.write('\\')			
	outfile.write('\n')
	outfile.write('}')
	outfile.write('\n')

sys.stdout.close()

