// Cascade Classifier file, used for Face Detection.
const char *faceCascadeFilename = "haarcascade_frontalface_alt_tree.xml";    // Haar face detector.
//const char *faceCascadeFilename = "lbpcascade_frontalface.xml";     // LBP face detector.

// Set camera resolution
const int DESIRED_CAMERA_WIDTH = 640;
const int DESIRED_CAMERA_HEIGHT = 480;

#include <iostream>

// Include OpenCV interface
#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

// Include user defined header files.
#include "detectObj.h"

using namespace std;
using namespace cv;

#if !defined VK_ESCAPE
    #define VK_ESCAPE 0x1B
#endif


// Load the face detector XML classifier
void InitDetect(CascadeClassifier &faceDetector){
    try {
        faceDetector.load(faceCascadeFilename);
    } catch (cv::Exception &e) {}
    if (faceDetector.empty()){
        cerr << "ERROR: Could not load the face detector classifier [" << faceCascadeFilename << "]" << endl;
        exit(1);
    }
    cout << "Loaded the face detector classifier [" << faceCascadeFilename << "]" << endl;
}

// Load the camera
void InitCam(VideoCapture &capture, int cameraNumber){
    try{
        capture.open(cameraNumber);
    } catch (cv::Exception &e) {}
    if (!capture.isOpened()){
        cerr << "ERROR: Failed to access to camera!" << endl;
        exit(1);
    }
    cout << "Access to camera " << cameraNumber << " successful." << endl;
}

void facedetection(VideoCapture &capture, CascadeClassifier &faceDetector){

    while(1){

        string windowName = "Face Detection";
        namedWindow(windowName, WINDOW_AUTOSIZE);

        // Grab the frame from the camera
        Mat frame;
        capture.read(frame);
        if(frame.empty()){
            cerr << "ERROR: Failed to grab the frame" << endl;
            exit(1);
        }
        Mat displayedFrame;
        frame.copyTo(displayedFrame);

        Rect FaceRect;
        detectObjects(frame, faceDetector, FaceRect);     // Detect the face

        if (FaceRect.width > 0){
            Mat faceImg = frame(FaceRect);
            rectangle(displayedFrame, FaceRect, CV_RGB(255, 01, 0), 2, CV_AA);
            imshow(windowName, displayedFrame);
        }
        else
            imshow(windowName, frame);

        char keypress = waitKey(20);
        if (keypress == VK_ESCAPE) {
            // Quit the program!
            break;
        }
    }


}

int main(int argc, char **argv)
{
    CascadeClassifier faceDetector;
    VideoCapture capture;

    // Load the face detection classifier
    InitDetect(faceDetector);

    // Get the camera number from the user
    int cameraNumber = 0;
    if (argc > 1)
        cameraNumber = atoi(argv[1]);

    // Access the camera
    InitCam(capture, cameraNumber);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

    // Detect face using the specified face detection classifier.
    facedetection(capture, faceDetector);


    return 0;
}
