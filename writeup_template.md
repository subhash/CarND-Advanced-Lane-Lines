##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[distorted]: ./output_images/distorted.png "Distorted"
[undistorted]: ./output_images/undistorted.png "Undistorted"
[undistorted_road]: ./output_images/undistorted_road.png "Undistorted road"
[before_warp]: ./output_images/before_warped.png "Before warping"
[warped]: ./output_images/warped.png "After warping"
[binary_threshold]: ./output_images/binary_test_images.png "Thresholded test images"
[sliding_window]: ./output_images/sliding-window.png "Sliding window"
[sliding_window_fit]: ./output_images/sliding-window-fit.png "Sliding window fit"
[annotated]: ./output_images/annotated.png "Annotated"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

[calibrate_camera()](https://github.com/subhash/CarND-Advanced-Lane-Lines/blob/master/lane-detection.py#L88)

The central idea behind calibration is to provide samples of relative differences between known points captured from various perspectives and distances so that we can calibrate the distortion caused by the lenses. The known points are encapsulated in `object_points`, which is simply the known corners of a chessboard of size (9,6). The perceived points are drawn from various images by using the function `findChessboardCorners` which returns true if it is able to find the required number of corners. With the valid set of found corners (`image_points`), we submit an equivalent set of `object_points` and derive the camera matrix and distortion coefficients. Supplying these parameters to the `cv2.undistort()` allows us to "undistort" any image captured by the same camera. For eg.

![alt text][distorted]
![alt text][undistorted]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
The first step to process a driving video is to undistort each of the images
![alt text][undistorted_road]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

[threshold_binary()](https://github.com/subhash/CarND-Advanced-Lane-Lines/blob/master/lane-detection.py#L146)

The steps to arrive at a binary thresholded image are:
 * Convert image to HLS color space and extract channels L and S. The S channel makes lane lanes stand out and the L channel is useful to extract the "lightedness" of short markings
 * Select pixels in these channels that pass a threshold value (75% for S and 90% for L)
 * Detect x and y gradients and calculate corresponding magnitude and direction values for them in the thresholded channels
 * Threshold the gradient values and use a combination:

 ```python
    grad_combined = np.logical_and(x_grad, y_grad)
    dir_combined = np.logical_and(dir_grad, mag_grad)
    combined = np.logical_or(grad_combined, dir_combined)
 ```

![alt text][binary_threshold]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

[warp_image()](https://github.com/subhash/CarND-Advanced-Lane-Lines/blob/master/lane-detection.py#L155)
In order to perform a perspective transform, I hardcoded the source and destination points to coincide with four points on the lane.

```python
maxy, maxx = self.image.shape[0], self.image.shape[1]
src = np.float32([(575, 464), (707, 464), (258, 682), (1049, 682)])
dst = np.float32([(450, 0), (maxx - 450, 0), (450, maxy), (maxx - 450, maxy)])
```

I tested the perspective transform on a straight line image and the lane markings appeared parallel in the warped image, as expected

![alt text][before_warp]
![alt text][warped]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

[search_for_fit()](https://github.com/subhash/CarND-Advanced-Lane-Lines/blob/master/lane-detection.py#L207)

I implemented a sliding window algorithm to detect pixels for the left and right lanes. With the detected points, I fit a 2nd degree polynomial for each of the lanes

![alt text][sliding_window]
![alt text][sliding_window_fit]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

[offset_from_centre()](https://github.com/subhash/CarND-Advanced-Lane-Lines/blob/master/lane-detection.py#L69)
[radius_of_curvature()](https://github.com/subhash/CarND-Advanced-Lane-Lines/blob/master/lane-detection.py#L65)

For calculating the radius curvature, I averaged the left and right lane fits. The position of vehicle is basically the difference between the centre of the image and perceived centre of the lane in real-world metrics. We use the left and right fits to estimate the x-value at the bottom of the image and calculate the midpoint between these as the lane center.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

[lane_mask()](https://github.com/subhash/CarND-Advanced-Lane-Lines/blob/master/lane-detection.py#L238)

We generate a mask by painting a polygon from the left lane to the right lane and unwarping it. This mask when superimposed on the original image represents the detected lane 

![alt text][annotated]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/TRjsNwnnH5s)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* Shadows and lighting differences posed the biggest challenges
* The failback mechanism I used is to reuse either the last best fit or the other lane's fit. Both these techniques can fail if the car is passing through a shadowed tunnel with differing curvature
* I could average last few fits to make a more robust solution
