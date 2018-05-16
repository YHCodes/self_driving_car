#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

"""
Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:

cv2.inRange() for color selection
cv2.fillPoly() for regions selection
cv2.line() to draw lines on an image given endpoints
cv2.addWeighted() to coadd / overlay two images cv2.cvtColor() to grayscale or change color cv2.imwrite() to output images to file
cv2.bitwise_and() to apply a mask to an image
"""
import math


def grayscale(img):
    """Applies the Grayscale transform"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness=2)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def get_color_thresholds(image):
    """ Select"""
    # Define color selection criteria
    red_threshold = 150
    green_threshold = 150
    blue_threshold = 50
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    # Mask pixels below the threshold
    color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                    (image[:,:,1] < rgb_threshold[1]) | \
                    (image[:,:,2] < rgb_threshold[2])

    return color_thresholds


def get_region_thresholds(image):
    ysize = image.shape[0]
    xsize = image.shape[1]

    # Define the vertices of a triangular mask.
    left_bottom = [0, 539]
    right_bottom = [959, 539]
    apex = [475, 320]

    # Perform a linear fit (y=Ax+B) to each of the three sides of the triangle
    # np.polyfit returns the coefficients [A, B] of the fit
    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)


    # Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

    return region_thresholds

import os
files = os.listdir("test_images/")
kernel_size = 5
low_threshold = 50
high_threshold = 150

# define the hough transfor parameters
rho = 1
theta = np.pi / 180
threshold = 1
min_line_len = 10
max_line_gap = 1
idx = 0

for image_file in files:
    image_file = 'test_images/' + image_file
    image = mpimg.imread(image_file)

    region_thresholds = get_region_thresholds(image)

    gray_image = grayscale(image)
    blur_image = gaussian_blur(gray_image, kernel_size)
    masked_edges = canny(blur_image, low_threshold, high_threshold)
    hough_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    hough_img[~region_thresholds] = [0, 0, 0]

    combo = weighted_img(hough_img, image)

    #line_image   = np.copy(image)

    color_thresholds = get_color_thresholds(image)

    # Mask color and region selection

    #line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

    plt.imshow(hough_img)
    plt.show()



"""------------"""
import os

files = os.listdir("test_images/")

for file in files:
    if file[0:6] != "output":
        img = mpimg.imread("test_images/" + file)
        gray = grayscale(img)
        gray = gaussian_blur(gray, 3)
        edges = canny(gray, 50, 150)

        imshape = img.shape
        vertices = np.array([[(.51 * imshape[1], imshape[0] * .58), (.49 * imshape[1], imshape[0] * 0.58),
                              (0, imshape[0]), (imshape[1], imshape[0])]], dtype=np.int32)
        target = region_of_interest(edges, vertices)
        lines = hough_lines(target, 1, np.pi / 180, 35, 5, 2)

        result = weighted_img(lines, img)

        plt.imshow(result, cmap='gray')

        r, g, b = cv2.split(result)
        result = cv2.merge((b, g, r))

        cv2.imwrite("test_images/output_" + file, result)


########################
def average_slope_intercept(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    # add more weight to longer lines    
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    
    return left_lane, right_lane # (slope, intercept), (slope, intercept)

def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None
    
    slope, intercept = line
    
    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    
    y1 = image.shape[0] # bottom of the image
    y2 = y1*0.6         # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    
    return left_line, right_line

    
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

def hough_lines2(image):
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

lane_images = []  
files = os.listdir("test_images/")
for file in files:
    if file[0:6] != "output":
        print(file)
        img = mpimg.imread("test_images/"+file)
        gray = grayscale(img)
        gray = gaussian_blur(gray, 3)
        edges = canny(gray, 50, 150)
        
        imshape = img.shape
        vertices = np.array([[(.51*imshape[1], imshape[0]*.58), (.49*imshape[1], imshape[0]*0.58), (0, imshape[0]), (imshape[1], imshape[0])]], dtype=np.int32)
        target = region_of_interest(edges, vertices)
        lines = hough_lines2(target)
        
        lines = lane_lines(img, lines)
        lane_image = draw_lane_lines(img, lines)
        lane_images.append(lane_image)
        r,g,b = cv2.split(lane_image)
        result = cv2.merge((b,g,r))
        cv2.imwrite("test_images_output/"+file, result)
show_images(lane_images)

