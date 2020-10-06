# image_analysis_STATGR5293
Assignment repo

## Getting Started
### Software Requirement
- Python 3.7
- Numpy
- OpenCv 4.4.0
- sklearn 0.22.2

### Installing Dependencies
Please first install [Anaconda](https://anaconda.org) and create an Anaconda environment with above packages installed.
```
conda create --name image_analysis 
```

After you create the environment, please activate it.
```
source activate image_analysis
```

### Usage
Assignment code are stored in different subdirectories. 
```
cd <image_analysis dir>/assignment1
python assignment1.py
```

For assignment1, input image is stored at ```<image_analysis dir>/assignment1/Homework1```. The output image is stored at ```<image_analysis dir>/assignment1/output```.

## Assignment1
Find faces using kmeans algorithm. First convert 3-d image to a 2-d array with shape (length*width, 3), so that we can use kmeans to put each pixel into different clusters.
```
kmeans = KMeans(n_clusters=k, random_state=0).fit(img_2d)
```

We then need to indicate which cluster represents faces. Define a standard BGR pixel which is the average skin color of people in the picture. The cluster with close average BGR value is selected to be the face cluster. I write a ```get_face_label()``` function to this task.
```
kmeans = KMeans(n_clusters=k, random_state=0).fit(img_2d)

std=[154.44781282, 173.70050865, 226.22034588]

true_label=get_face_label(img_2d,labels,k)
```
After we have the label of the face cluster, we need to find four corners to draw the bonding box for a face. I write a ```find_corners()``` function to find top/bottom left/right corners for one face. For example, if the image contains 4 faces, the function should be called four times. The function take an approximate range of pixels where the face can be located, and it finds the max/min x/y coordinates in the face cluster. 
```
x_min, x_max, y_min, y_max = find_corners(img_shape_label, 75, 175, 50, 160, true_label)
```
Then final step is to draw the bounding box using four corners we just found.
```
img_orig = cv2.line(img_orig, top_left, bottom_left,line_color)
img_orig = cv2.line(img_orig, top_left, top_right,line_color)
img_orig = cv2.line(img_orig, bottom_left, bottom_right,line_color)
img_orig = cv2.line(img_orig, bottom_right, top_right,line_color)
```

The output of the code contains two images. One is the orignial input image with bounding boxes indicating where the faces are. Another image is a binary image with bounding boxes. 


