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
* [project_out.mp4](./white.mp4) (a video recording of the lane lines detected along with the shaded region between the lane lines)
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

The Advanced_Lane_Lines.ipynb [Advanced_Lane_Lines_Updated.ipynb](./Advanced_Lane_Lines.ipynb) file contains the image processing pipeline.

### 1. Computed the camera matrix and distortion coefficients
First i used the test chessboard images [calibration](./camera_cal) to find Chessboard corners and saved this data for calibration & distortion correction later.
```
# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

```
![Calibration](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/Cam_Calibration.png)

### 2. # Select Region of Interest
Since the lane lines are typically found in a cone section of the center of the image, I then blacked out most of the image except the center cone section (defined by the left,right & apex vertices. This helped cut out a lot of background noise image.
```
left_bottom = [1/8*image.shape[1],image.shape[0]]
 right_bottom = [7/8*image.shape[1],image.shape[0]]
 apex = [image.shape[1]/2,image.shape[0]/2]
   
 vertices= np.array([[left_bottom,right_bottom,apex ]],dtype=np.int32)
 img=region_of_interest(image,vertices)
 ```

![Before ROI](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/before_ROI.png)


### 3. Change color space to HLS, filter image for lane lines using Sobel
Next, I changed color space to better detect lane lines, and used Sobel filtering. 

Initially, my right lane curvature was out of whack, curving dramatically to the right at the top of the image, primarily due to lot of noise pick up from the neighboring lanes. I had applied only Sobel filter in X direction.
```	

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
Later, I added in Sobel filtering in both x & y directions, and also added in Directional thresholding as well as magnitude thresholding that seemed to get rid of the noise and my right line detection was much more reasonable, especially at the top of the image. The left line detection was stable throughout because there was a solid yellow line which was picked up just as well with or without applying the additional filtering in the Y direction.

```
	
 	# Define sobel kernel size
    ksize = 7
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(10, 255))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(60, 255))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(40, 255))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(.65, 1.05))
    # Combine all the thresholding information
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # Threshold color channel
    s_binary = np.zeros_like(combined)
    s_binary[(s > 160) & (s < 255)] = 1
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors    
    color_binary = np.zeros_like(combined)
    color_binary[(s_binary > 0) | (combined > 0)] = 1

    
```
![S-channel](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/HLS.png)

![Sobel binary threshold](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/Sobel_Binary.png)

### SOBEL FLTER IN X DIRECTION ONLY
![Before](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/Before_Sobel_Adjustment.png)

### SOBEL FLTER IN X & Y  DIRECTION ONLY
![After](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/After_Sobel_Adjustment.png)

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
![Before Distortion Correction](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/undistorted.png)


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

![Original image](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/pre_pipeline.png)


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

![Original image](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/result_images/pre_shade.png)


In the same step, i overlayed the curvature and position information onto the image.

Finally, this same pipeline was run on the sample project video, processing each frame at a time, and then compiling an output video using the MoviePy package.

### The final pipeline was established in 3 stages:
```
def pipeline(img):
```
```
# Undistort calibrated image, apply Sobel & Color Filters
    color_binary,combined,s_binary,blur=pipeline1(img)

# Warp into Bird's Eye view perspective
    warp, M, Min,src,dst=pipeline2(color_binary)
    
# Fit a windowed polynomial to the warped lane lines, compute curvatures and position, compute Polygon path
    out_img,left_fitx,right_fitx,ploty,left_curveradius,right_curveradius=window_polyfit(warp)

# Warp again into the original perspective with overlaid polygonal path between lane lines
    result= pipeline3(out_img,left_fitx,right_fitx,ploty,left_curveradius,right_curveradius,img)
    
    return result
```

![Original video](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/project_video.mp4)

![Processed video](https://raw.githubusercontent.com/eshnil2000/CarND-Advanced-Lane-Lines/master/final_out.mp4)

### 6. Discussion Points
As can be seen in the resultant video, there is some fluctuation in the shaded region processing due to noise moving from frame to frame especially on the right hand side lane. The left hand side lane with the yellow color is reasonably stable In some frames where there is a transition from yellow to white noise specs, the video shaded region seems to fluctuate. This means under different weather conditions/ shadows, this algorithm may have a tough time keeping track. Also, on roads with poorly marked lane lines or no lane lines, the algorithm would do a poor job keeping to the lane lines.

I could also implement some smoothing/averaging between frames which would reduce the fluctuations between frames.

In a few areas with shadows/ change of color, the lane detection seems to falter especially on the right hand side top edge and show extra curvature due to noise from shadows. This is an area of improvement where I can filter out some of this noise by averaging across multiple frames and setting maximum distance between the left and right lanes.





