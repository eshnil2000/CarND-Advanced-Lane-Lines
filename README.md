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

#### Submission code is usable and readable

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

[![Before ROI](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/before_ROI.png)

[![After ROI](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/ROI.png)


### 3. Change color space to HLS, filter image for lane lines using Sobel
Next, I changed color space to better detect lane lines, and used Sobel filtering. 

```
img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
 #img=image
 # Convert to HLS color space and separate the  channel
 hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    
l_channel = hls[:,:,1]
s_channel = hls[:,:,2]
# Sobel x
sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
# Threshold x gradient
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
# Threshold color channel
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
s_final=np.zeros_like(s_channel)
s_final[(sxbinary==1) & (s_binary ==1)] = 1
```

Then I used these points to perform Distortion Correction using the Camera matrix calculated earlier
```
result, mtx,dist = cal_undistort(sxbinary, objpoints, imgpoints)
```

```
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    #img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    #undist = np.copy(img)  # Delete this line
    undist=dst
    return undist, mtx, dist
```
[![Before Distortion Correction](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/original_chess.png)

[![After Distortion Correction](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/undistorted.png)

### 4. Perform Warp transform to get Bird's eye view, perform windowed Polynomial fit
To get accurate representation of the lane line perspective, i warped the original image, by selecting 4 points on the original image representing roughly the 4 corners of the lane and transforming them so that the lane lines appear parallel to each other in the Bird's eye view

```
height,width = result.shape[:2]

    # define source and destination points for transform
    src = np.float32([(750,450),(550,450),

                      (1050,710),
                      (80,710), 
                      ])
    dst = np.float32([(width-450,0),
                      (450,0),
                      (width-450,height),
                      (450,height)])

    result, M, Minv=warp_transform(result,src,dst)
```

Since the lane lines maybe discontinuous, I used a windowed polynomial fitting algorithm to trace out the complete left and right lane lines. This code was taken from the sample provided by Udacity. The technique involves dividing up the image into vertical windows, computing the histogram to find lane lines in the window, computing the histogram in subsequent windows and then fitting a polynomial smoothed out to fit the windows together.
```
result,left_fitx,right_fitx,ploty,left_curveradius,right_curveradius=window_polyfit(result)

```

[![Original image](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/pre_pipeline.png)

[![After Perspective transform, windowed polynomial fit](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/pipeline.png)

In this same step, I calculate the approximate lane curvature radius and the position of the car assuming camera is mounted at center of the car. 

### 5. Shade region between lane lines, warp image using inverse perspective transform
Now that the lane lines are detected, I filled in the area between the lane lines with the cv2.fillPoly command. Next, i performed an inverse perspective transform to warp the co-ordinates of the fillPoly to that of the original image, and then superimposed this fill on the original image
```
# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
```

[![Original image](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/pre_shade.png)

[![After Shading, inverse perspective transform](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/post_shade.png)

In the same step, i overlayed the curvature and position information onto the image.

Finally, this same pipeline was run on the sample project video, processing each frame at a time, and then compiling an output video using the MoviePy package.

[![Original video](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/project_video.mp4)

[![Processed video](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/white_bu.mp4)






