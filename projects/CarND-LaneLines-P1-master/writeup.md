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



In order to draw a single line on the left and right lanes, I modified the draw_lines() function by averaging and extrapolating lines.



If you'd like to include images to show how the pipeline works, here is how to include an image: 

![solidWhiteCurve.jpg](https://github.com/YHCodes/self_driving_car/blob/master/projects/CarND-LaneLines-P1-master/test_images_output/solidWhiteCurve.jpg?raw=true)

![solidYellowCurve.jpg](https://github.com/YHCodes/self_driving_car/blob/master/projects/CarND-LaneLines-P1-master/test_images_output/solidYellowCurve.jpg?raw=true)




### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the car driving at night, the background is really dark, will have a bad influence.

Another shortcoming could be when the road have shadow of building or tree, it may cause the detecting road lane lines distortion.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to improve code robustness by eliminating background nose.

Another potential improvement could be to filter more unrelated objects.s
