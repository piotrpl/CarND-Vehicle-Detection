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
[image4]: ./output_images/slide_window_example.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Histogram of Oriented Gradients (HOG)

####1. HOG features extraction from the training images.

The code for this step is contained in the get_hog_features() function in the "helper functions" section of the `some_file.py` IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` .png images taken from the GTI vehicle image database and the KITTI vision benchmark suite. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]
![alt text][image3]

####2. HOG parameters selection.

The code for this step is contained in the get_hog_features() function in the "helper functions" section of the `some_file.py` IPython notebook.

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

The code for this step is contained in the get_hog_features() function in the "helper functions" section of the `some_file.py` IPython notebook.

My SVN classifier is trained using previously loaded data out of which 10% is used as a test data set. Data is shuffeled before training. The classifier extracts features, normlizes them and stack on top of each other. Next to the HOG feature extraction I have also used histogram of color values in an image combined with spacial binning of color. Using the above parameters reulsted in the following outcome:

```
Using: 8 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 5568
16.43 Seconds to train SVC...
Test Accuracy of SVC =  0.9876
```

###Sliding Window Search

####1. Sliding window search.

The code for this step is contained in the get_hog_features() function in the "helper functions" section of the `some_file.py` IPython notebook.

To check for the existance of a car I used sliding window technique. In order to imporove performance I have chosed to only slide throught he lower half of the image, as this is the part where we expect to see cars. For the window size I have chosed size of 96x96 and overlap fraction of 75%. Those values were set based on some exploratory testing and discovery.

![alt text][image3]

####2. Pipeline example.

As described above, I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a good result. Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

