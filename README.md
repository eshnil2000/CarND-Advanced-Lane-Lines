# Advanced Lane Finding

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
The purpose of the project is to detect the lanes from the video stream and shade the areas between the identified lanes. This was achieved by setting up a Computer Vision processing "pipeline" through which each of the frames of the video was put through to gather the end result video.

Below is the list of techniques used to detect lanes.

Camera Calibration: Transformation between 2D image points to 3D object points.
Distortion Correction: Consistent representation of the geometrical shape of objects.
Perpective Transform: Warping images to effectively view them from a different angle or direction.
Edge Detection: Sobel Operator, Magnitude Gradient, Directional Gradient, and HLS Color Space with Color thresholding

I also used Region of Interest & Gaussian Blurring to focus & better detect the lane lines.

# Files & Code Quality
These are  key files: 
* [Advanced_Lane_Lines.ipynb](./model.ipynb) (script used to setup & execute the pipeline)
* [white.mp4](./white.mp4) (a video recording of the lane lines detected along with the shaded region between the lane lines)
* [README](./README.md) (this readme file has the write up for the project!)

The Project
---
The goals / steps of this project are the following:
* Build an image processing pipeline to detect lane lines
* Transform/ warp the image to get a Bird's eyeview of the lane lines
* Draw shaded region between the lane lines, calculate radius of curvature
* Transform/ warp the image back to original, mapping the shaded region back onto the original image

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.

#### 3. Submission code is usable and readable

The Advanced_Lane_Lines.ipynb [Advanced_Lane_Lines.ipynb](./Advanced_Lane_Lines.ipynb) file contains the image processing pipeline.

### 1. Computed the camera matrix and distortion coefficients
First i used the test chessboard images [calibration_wide.ipynb](./calibration_wide) to find Chessboard corners and saved this data for calibration & distortion correction later.
```
# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

```
### 2. # Select Region of Interest
Since the lane lines are typically found in a cone section of the center of the image, I then blacked out most of the image except the center cone section (defined by the left,right & apex vertices. This helped cut out a lot of background noise image.
```
left_bottom = [1/8*image.shape[1],image.shape[0]]
 right_bottom = [7/8*image.shape[1],image.shape[0]]
 apex = [image.shape[1]/2,image.shape[0]/2]
   
 vertices= np.array([[left_bottom,right_bottom,apex ]],dtype=np.int32)
 img=region_of_interest(image,vertices)
 ```

[![Before ROI](https://raw.githubusercontent.com/eshnil2000/CarND-Behavioral-Cloning/master/model.png)


