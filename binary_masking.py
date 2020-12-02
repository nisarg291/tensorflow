# import cv2
# import numpy as np

# # Load image, create mask, and draw white circle on mask
# image = cv2.imread('benign.png')
# mask = np.zeros(image.shape, dtype=np.uint8)
# mask = cv2.circle(mask, (260, 300), 225, (255,255,255), -1) 

# # Mask input image with binary mask
# result = cv2.bitwise_and(image, mask)
# # Color background white
# result[mask==0] = 255 # Optional

# cv2.imshow('image', image)
# cv2.imshow('mask', mask)
# cv2.imshow('result', result)
# cv2.waitKey()
import cv2
import numpy as np

arr =  cv2.imread('benign.png') #3 channel image
mask_3d = np.zeros(shape=(471,562,3))
# mask[1,1] = 1 # binary mask
# mask_3d = np.stack((mask,mask,mask),axis=0) #3 channel mask

## Answer 1
# Simply multiply the image array with the mask

masked_arr = arr*mask_3d

## Answer 2
# Use the where function in numpy

masked_arr = np.where(mask_3d==1,arr,mask_3d)

#Both answer gives
print(masked_arr)
cv2.imshow("masked_arr",masked_arr)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
  
# originalImage = cv2.imread('benign.jpg')
# grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
  
# (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
 
# cv2.imshow('Black white image', blackAndWhiteImage)
# cv2.imshow('Original image',originalImage)
# cv2.imshow('Gray image', grayImage)
  
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# for i in range(0,437):
arr =  cv2.imread('malignant.png') #3 channel image
lower_black = np.array([0,0,0], dtype = "uint16")
upper_black = np.array([90,90,90], dtype = "uint16")
black_mask = cv2.inRange(arr, lower_black, upper_black)
cv2.imshow('mask0',black_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()