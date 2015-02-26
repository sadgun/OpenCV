#pragma once

#include <iostream>

// Include OpenCV interface
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

// Search for objects in the image.
void detectObjects(const Mat &SrcImg, CascadeClassifier &Cascade, Rect &facerect, int scaledWidth = 320);

