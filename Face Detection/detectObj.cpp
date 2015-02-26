// Program to detect the face and return the rectangle coordinates of the face (if face is found) else
// return an invalid rectangle

#include "detectObj.h"

void detectObjects(const Mat &SrcImg, CascadeClassifier &Cascade, Rect &facerect, int scaledWidth){

    // Search for the biggest object in the image
    int flags =  CASCADE_FIND_BIGGEST_OBJECT ; // CASCADE_SCALE_IMAGE | CASCADE_DO_ROUGH_SEARCH;
    // Smallest object size.
    Size minFeatureSize = Size(20, 20);
    // search factor size
    float searchScaleFactor = 1.1f;
    // Number of detections that should be filtered out.
    // A value of 2 will yield both good + bad detections, a value of 6 will yield only good detections,
    // some of them are missed
    int minNeighbors = 4;

    Mat smallimg;
    float scale = SrcImg.cols/(float) scaledWidth;
    if (SrcImg.cols > scaledWidth){
        int scaledHeight = cvRound(SrcImg.rows/scale);
        resize(SrcImg, smallimg, cvSize(scaledWidth, scaledHeight));
    }
    else
        smallimg = SrcImg;

    Mat Gray;
    if (smallimg.channels() == 3)
        cvtColor(smallimg, Gray, CV_BGR2GRAY);
    else if (smallimg.channels() == 4)
        cvtColor(smallimg, Gray, CV_BGRA2GRAY);
    else
        Gray = smallimg;

    Mat ImgHistEqual;
    equalizeHist(Gray, ImgHistEqual);

    vector<Rect> objects;
    Cascade.detectMultiScale(ImgHistEqual, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);

    if (SrcImg.cols > scaledWidth){
        for (int i = 0; i < (int)objects.size(); i++ ){
            objects[i].x = cvRound(objects[i].x * scale);
            objects[i].y = cvRound(objects[i].y * scale);
            objects[i].width = cvRound(objects[i].width * scale);
            objects[i].height = cvRound(objects[i].height * scale);
    }
    }
    for (int i = 0; i < (int)objects.size(); i++ ) {
        if (objects[i].x < 0)
            objects[i].x = 0;
        if (objects[i].y < 0)
            objects[i].y = 0;
        if (objects[i].x + objects[i].width > SrcImg.cols)
            objects[i].x = SrcImg.cols - objects[i].width;
        if (objects[i].y + objects[i].height > SrcImg.rows)
            objects[i].y = SrcImg.rows - objects[i].height;
    }

    if (objects.size() > 0){
        // Return the only detected object.
        facerect = (Rect)objects.at(0);
    }
    else
    {
        // Return an invalid rect.
        facerect = Rect(-1,-1,-1,-1);
    }
}


