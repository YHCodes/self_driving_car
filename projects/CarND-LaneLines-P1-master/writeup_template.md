# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

1. I converted the images to grayscale by `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`
2. Then I blured the gray images by `cv2.GaussianBlur()` function
3. Then I used `cv2.canny()` function to detect edges from the blur image
4. Then I used `cv2.HoughLinesP()` to detect road lane lines of the images
5. Finally  I used `cv2.addWeighted()` to combine the detecting of the road lane lines and raw image.



In order to draw a single line on the left and right lanes, I modified the draw_lines() function by averaging



If you'd like to include images to show how the pipeline works, here is how to include an image: 




### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
