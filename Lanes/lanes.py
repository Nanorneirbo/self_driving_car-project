import cv2
import numpy as np
import matplotlib.pyplot as plt

# function to run the canny 
def canny(image):
	# in - a new image
	# out - greyscael, smoothed, canny image. 
	#greyscale the image
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
	#blur the image
	blur = cv2.GaussianBlur(gray, (5,5),0)
	# canny - detect big changes in gradients using derivatives
	canny = cv2.Canny(blur, 50, 150)
	return canny


def regionofinterest(image):
	# in - the image 
	# out - a filled mask showing the lane we want 
	# height of image = rows in the array
	height = image.shape[0]
	# actually just one but fillpoly dont like that
	polygons = np.array([
	[(200,height), (1100, height), (550,250)]
	])
	# create an array of zeros same shape as image
	mask = np.zeros_like(image)
	# fill the region of interest with white only
	cv2.fillPoly(mask, polygons, 255)
	#use bitwise and to clear out uninteresting stuff
	masked_image = cv2.bitwise_and(image,mask)
	return masked_image
	
def display_lines(image,lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			x1,y1,x2,y2 = line.reshape(4)
			# draw line blue and ten thick 
			cv2.line(line_image, (x1,y1), (x2,y2),(255,0,0), 10)
	return line_image

def make_coordinates(image, line_parameters):
	slope, intercept = line_parameters
	print (image.shape)
	# y's start at bottom x's use slope and intercet. 
	y1 = image.shape[0]
	# 3/5 is an estimate 
	y2 = int(y1 * 3/5)
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	return np.array([x1, y1, x2, y2])

	
def average_slope_intercept(image,lines):
	# in - the original image and the lines image

	left_fit =[]
	right_fit =[]
	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)

		# get the slope using a polynomial. 
	
		parameters = np.polyfit((x1, x2), (y1,y2), 1)
		slope = parameters[0]
		intercept = parameters[1]
		# negative = for slope left hand side, positive is on right
		if slope < 0 :
			left_fit.append((slope, intercept))
		else :
			right_fit.append((slope,intercept))
	
	# test the lines - print(left_fit)
	#print(right_fit)
	# average the sides and make sure to do along the right access(0)
	
	left_fit_average = np.average(left_fit, axis = 0)
	right_fit_average = np.average(right_fit, axis = 0)
	
	
	#print(left_fit_average, 'left')
	#print(right_fit_average, 'right')
	
	# need to get the coordinates - created a function above
	left_line = make_coordinates(image, left_fit_average)
	right_line = make_coordinates(image, right_fit_average)
	# return the lines
	return np.array([left_line, right_line])
	
	
# input road image
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
# changed so as not to reuse canny twice
canny_image = canny(lane_image)
cropped_image = regionofinterest(canny_image)
# 2 pixels single degree precision threshhold - empty array - min line - gaps we fill.
# make sure to initialise the array to empty with []
lines = cv2.HoughLinesP(cropped_image, 2,np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)

averaged_lines = average_slope_intercept(lane_image,lines)

line_image = display_lines(lane_image, averaged_lines)
# show region of interest in cv2
#cv2.imshow('result',regionofinterest(canny))
#line image show
#cv2.imshow('result', line_image)

# comnine the line and colour image weight original by 0.8 - to darken it a bit
# add the line image to it and weight it normally
# combo image needs a gamma as final argument
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1,1)
#cv2.imshow('result', combo_image)

cv2.imshow('result', line_image)

# show it using matplot
#plt.imshow(canny)
#plt.show()

cv2.waitKey(0)