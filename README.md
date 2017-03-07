#Vehicle Detection

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/HOG_example_car.jpg
[image3]: ./output_images/HOG_example_notcar.jpg
[image4]: ./output_images/sliding_windows.jpg
[image5]: ./output_images/sliding_window.jpg
[image6]: ./output_images/video_frame1.png
[image7]: ./output_images/video_frame2.png
[image8]: ./output_images/video_frame3.png
[image9]: ./output_images/video_frame4.png
[image10]: ./output_images/video_frame5.png
[image11]: ./output_images/video_frame6.png
[image12]: ./output_images/video_final_frame.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Histogram of Oriented Gradients (HOG)

####1. HOG features extraction from the training images.

The code for this step is contained in the `get_hog_features()` function in the "helper functions" section and "Reading images of two classes for training and testing" section of the `solution.ipny` IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` .png images taken from the GTI vehicle image database and the KITTI vision benchmark suite. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]
![alt text][image3]

####2. HOG parameters selection.

The code for this step is contained in the "The feature parameters" section of the `solution.ipny` IPython notebook.

HOG feature extraction is done usig get_hog_features() function. This function takes image together with the HOG parameters returning HOG extracted features. Images, before feature exetraction are convered to the YCrCb color space. HOG features are extracted for each channel. Ilustration of the HOG features extraction can be seen in the above section.

Parameers were selected based on the exploration, also performed as part of one of the lessons during perparation for this project. For the final selection/verification of parameters and its values I used linear SVN classifier and the training accuracy. The best results I have achieved using parameters shown below

```python
color_space = 'YCrCb'
orient = 8
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (16, 16)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True
```

####3. Classifier training.

The code for this step is contained in the "Classifier training" section of the `solution.ipny` IPython notebook.

My SVN classifier is trained using previously loaded data out of which 10% is used as a test data set. Data is shuffeled before training. The classifier extracts features, normlizes them and stack on top of each other. Next to the HOG feature extraction I have also used histogram of color values in an image combined with spacial binning of color. Using the above parameters reulsted in the following outcome:

```
Using: 8 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 5568
16.43 Seconds to train SVC...
Test Accuracy of SVC =  0.9876
```

###Sliding Window Search

####1. Sliding window search.

The code for this step is contained in the `slide_window()` function in the "helper functions" section of the `solution.ipny` IPython notebook.

To check for the existance of a car I used sliding window technique. In order to imporove performance I have chosed to only slide throught he lower half of the image, as this is the part where we expect to see cars. For the window size I have chosed size of 96x96 and overlap fraction of 75%. Those values were set based on some exploratory testing and discovery.

![alt text][image4]

####2. Pipeline example.

The code for this step is contained in the `search_windows()` function in the "helper functions" section of the `solution.ipny` IPython notebook.

As described above, I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a good result. Here are some example images:

![alt text][image5]
---

### Video Implementation

####1. Vehicle detection video output.

Here's a [link to my video result](https://www.dropbox.com/s/5ksq88r2gncd2j8/result_project_video.mp4?dl=0)


####2. Filter for false positives and overlapping bounding boxes.

The code for this step is contained in the `process_heat()` function in the "pipeline" section of the `solution.ipny` IPython notebook.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. Furhter, assuming that each blob corresponded to a vehicle I constructed bounding boxes to cover the area of each blob detected.

Even though identifying individual blobs helped tremendously to reduce amount of false positives in the solution, I was still faced with some in my video. As a way of further reducing them down I have collected bounding boxes across number of frames using `deque` and then applied heatmap and "labeling" on this list. The effect can be seen in the final video and is also discussed in the final part of this writeup.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames, their corresponding heatmaps and the output of `scipy.ndimage.measurements.label()` for each image

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image12]



---

###Discussion & future consideartions

1. The white car is tricky to be kept detected as it gets farther. I believe that this could be fixed by increasing the overlap of windows when applying sliding windows on a frame, e.g. to 90%. This on the other hand would come as very expensive computation task and would increase the time of the video processing significantly. In the end I have compromised on the 75% overlap which yielded satisfactory results.
2. Retrieving HOG features seems to be quite expensive as well although it is something that can't be avoided as it directly correlates to the accuracy of our classifier. Applying deep learning or some sort of hybrid solution here could provide more flexibility in tuning up parameters.
3. In order to reduce false positives I have used heatmap and data "labeling" in order to create bounding boxes. This helped a lot but did not fully removed false positives. Further, with applying this pipeline on each individaul frame caused boxes around cars to wobble lot which was not desired. As a step to improve it I tried to "cache" hot windows in a global `deque` accross few frames and create bouding boxes on a larger set of windows. This helped in reducing the wobbling effect but did not help with reducing other outstanding false positives. As a final solution I have "cached" with `deque` bounding boxes across few frames and then applied heatmap and "labeling" on top if. This hepled eliminating false positives but on the other hand made the bounding boxes not alway wrapping the car in its fullnes. Definitely it seems like a right direction but it could see some further improvements.
4. Sometimes you can notice that there is a detection of cars driving opposite direction. As much as it may seem to be a false postive it is acutal correct detection. This could be reduced by reducing the space for sliding window on the x axis.
