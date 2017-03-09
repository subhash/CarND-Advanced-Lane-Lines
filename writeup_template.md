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

`lane_detection.py - l32 - calibrateCamera()`

The central idea behind calibration is to provide samples of relative differences between known points captured from various perspectives and distances so that we can calibrate the distortion caused by the lenses. The known points are encapsulated in `object_points`, which is simply the known corners of a chessboard of size (9,6). The perceived points are drawn from various images by using the function `findChessboardCorners` which returns true if it is able to find the required number of corners. With the valid set of found corners (`image_points`), we submit an equivalent set of `object_points` and derive the camera matrix and distortion coefficients. Supplying these parameters to the `cv2.undistort()` allows us to "undistort" any image captured by the same camera. For eg.

![alt text][distorted]
![alt text][undistorted]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
The first step to process a driving video is to undistort each of the images
![alt text][undistorted_road]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

`lane_detection.py - l138 - threshold_pipeline()`

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

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][before_warp]
![alt text][warped]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][sliding_window]
![alt text][sliding_window_fit]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][annotated]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

