from PIL import Image, ImageOps
import numpy as np
import os
import sys
import cv2
import random

SOURCES = [os.getcwd()+'/assets/DepthMap_PCD/DepthMap/test',
			os.getcwd()+'/assets/DepthMap_TOF/DepthMap/train']

for SRC in SOURCES:
	subtypes = os.listdir(SRC)
	for s in subtypes:
		with open(os.getcwd()+'/src/CNN/depthmap/sampledata_'+str(s)+'.h', 'w+') as outfile:
			fnames = os.listdir(SRC+'/'+s)

			indices = np.arange(0,len(fnames),1)
			random.shuffle(indices)
			try:
				chosen = indices[0:20]
			except:
				chosen = indices

			
			out_arr_result = []
			for ix,val in enumerate(chosen):
				#Write out data to the header file
				im = Image.open(SRC+'/'+s+'/'+fnames[val])
				size = (64*2,64*2)
				img = im.resize(size)
				np1_img = np.array(img)
				new_img = Image.fromarray(np1_img)
				new_img.save("new_img.png")

				src = cv2.imread('new_img.png', cv2.IMREAD_UNCHANGED)

				if len(src.shape) > 2 and src.shape[2] == 4:
					src = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)

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

				
				arr_result = []
				for array in new_img:
					first = int(array[3]) & 0xFF
					second = int(array[2]) & 0xFF
					third = int(array[1]) & 0xFF
					fourth = int(array[0]) & 0xFF

					result = first | second << 8 | third << 16 | fourth << 24
	
					arr_result.append(result)  
			
				out_arr_result.append(np.asarray(arr_result, dtype=np.uint32))
			
			# print(s)
			# print(out_arr_result[0])
			# input()
			out_arr_len =np.shape(out_arr_result)[0]

			outfile.write('#define SAMPLE_INPUT_'+str(s)+' { \\')
			outfile.write('\n')

			for nsamp in range(out_arr_len):
				for i in range(len(out_arr_result[nsamp])):
					if i==0:
						outfile.write('\n')
						outfile.write('{')
						outfile.write('\n')
						outfile.write('\t0x{0:08x},\t'.format((out_arr_result[nsamp][i])))

					else :
						d = i%8
						if(d!=0):
							outfile.write('0x{0:08x},\t'.format((out_arr_result[nsamp][i])))
						else:
							outfile.write('\\')
							outfile.write('\n\t')
							outfile.write('0x{0:08x},\t'.format((out_arr_result[nsamp][i])))

					if i == len(out_arr_result[nsamp])-1:
						outfile.write('\n')
						outfile.write('},')

			outfile.write('\\')			
			outfile.write('\n')
			outfile.write('}')
			outfile.write('\n')
			outfile.write('\n')
			outfile.write('#define SAMPLE_SIZE_'+str(s)+' '+str(out_arr_len))
			outfile.write('\n')

			sys.stdout.close()
			

