#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include "mixformer/mixformer_trt.h"
#include "kcf/kcftracker.hpp"
#include <utils.h>

using namespace std;
using namespace cv;
using namespace kcf;

Rect2d bbox;
bool bboxSelected = false;
bool drawing = false;
Point point1, point2;
bool trackerInitialized = false; 


/**
 * @brief Mouse callback function for selecting and drawing a bounding box on the frame.
 * 
 * This function allows users to select a bounding box (bbox) for tracking by clicking and dragging the mouse
 * over the video fram  e. The user can re-select a new bbox by clicking again.
 * 
 * @param event Type of the mouse event (e.g., left button down, mouse move).
 * @param x The current x-coordinate of the mouse pointer.
 * @param y The current y-coordinate of the mouse pointer.
 * @param flags Additional flags for the event (unused in this function).
 * @param userdata Pointer to the current video frame (cast to Mat*).
 */
void mouseCallback(int event, int x, int y, int, void* userdata)
{
    Mat& frame = *(Mat*)userdata;

    if (event == EVENT_LBUTTONDOWN) {
        point1 = Point(x, y);
        drawing = true;
    }
    else if (event == EVENT_MOUSEMOVE && drawing) {
        point2 = Point(x, y);
        Mat tempFrame = frame.clone();
        rectangle(tempFrame, point1, point2, Scalar(255, 0, 0), 2);
        imshow("Tracking", tempFrame);
    }
    else if (event == EVENT_LBUTTONUP) {
        point2 = Point(x, y);
        drawing = false;

        if (abs(point1.x - point2.x) < 10 && abs(point1.y - point2.y) < 10) {
            bbox = Rect2d();   
            bboxSelected = false; 
            trackerInitialized = false;
        }
        else{
            bbox = Rect2d(point1, point2);
            bboxSelected = true;
            trackerInitialized = false; 
        }
    }
}


void track(MixformerTRT *tracker, KCFTracker *kcftracker, int bitsThreshold)
{
    // Open the default camera (index 0). Change index if necessary.
    VideoCapture cap(0);  

    if (!cap.isOpened()) {
        cerr << "Error: Unable to open camera." << endl;
        return;
    }

    Mat frame;
    Mat previousFrameBbox; // Variable to store the previous frame
    bool mfTracked = false;
    bool ok = false;
    bool terminateEarly = false;
    vector<float> fpsValues;

    int frameNumber = 0;
    float mixformerTriggers = 0;

    namedWindow("Tracking");
    setMouseCallback("Tracking", mouseCallback, &frame); 

    while (!terminateEarly) {
        // Capture a frame from the camera
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Empty frame captured." << endl;
            break;
        }

        // If the object is selected or reselected, initialize the trackers
        if (bboxSelected && !trackerInitialized) {
            kcftracker->init(frame, bbox);
            tracker->init(frame, rect2mfbbox(bbox));
            previousFrameBbox = frame(bbox).clone(); 
            trackerInitialized = true;
        }

        // If the trackers are initialized, start tracking
        if (trackerInitialized) {
            double timer = (double)cv::getTickCount();

            // reinitialize KCF tracker after MixFormer tracking step
            if (mfTracked){
                kcftracker->init(frame, bbox);
                mfTracked = false;
            }

            Mat currentFrameBbox = frame(bbox);
            uint64_t hashCurrentFrame = calculateAverageHash(currentFrameBbox);
            uint64_t hashPreviousFrame = calculateAverageHash(previousFrameBbox);
            // Compute the XOR between the current and previous frame hashes to сount the number of bits that have changed
            int differingBits = __builtin_popcountll(hashCurrentFrame ^ hashPreviousFrame);

            if (differingBits >= bitsThreshold) {
                mixformerTriggers++;
                bbox = mfbbox2rect(tracker->track(frame));
                mfTracked = true;
                ok = true;
            } else {
                ok = kcftracker->update(frame, bbox);
            }

            float fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);
            fpsValues.push_back(fps);

            if (ok) {
                rectangle(frame, bbox, Scalar(0, 255, 0), 2, 1);
            } else {
                putText(frame, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
            }

            putText(frame, "FPS: " + std::to_string(int(fps)), Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(170, 50, 50), 2);
            frameNumber++;
        }

        imshow("Tracking", frame);

        if (waitKey(1) == 'q') {
            terminateEarly = true;
        }

        // Boundary checks for bounding box
        bbox.x = std::max(0, static_cast<int>(bbox.x));
        bbox.y = std::max(0, static_cast<int>(bbox.y));
        bbox.width = std::max(4, static_cast<int>(bbox.width));
        bbox.height = std::max(4, static_cast<int>(bbox.height));

        if (bbox.x + bbox.width > frame.size().width){
            bbox.width = frame.size().width - bbox.x;
        }
         
        if (bbox.y + bbox.height > frame.size().height){
            bbox.height = frame.size().height - bbox.y;
        }

        previousFrameBbox = frame(bbox).clone(); // remember this bbox for differingBits calculation on the next frame   
    }

    float meanFps = std::accumulate(fpsValues.begin(), fpsValues.end(), 0.0) / fpsValues.size();

    std::cout << "Average MixFormer trigger rate: " << mixformerTriggers / frameNumber * 100 << "%" << std::endl;
    std::cout << "Average FPS during tracking: " << meanFps << std::endl;
    
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s [modelPath] [bitsThreshold]\n", argv[0]);
        return -1;
    }

    std::string modelPath = argv[1];
    int bitsThreshold = std::stoi(argv[2]);

    MixformerTRT *Mixformerer = new MixformerTRT(modelPath);

    bool HOG = true, FIXEDWINDOW = true, MULTISCALE = true, LAB = true, DSST = false;
    KCFTracker *kcftracker = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);

    track(Mixformerer, kcftracker, bitsThreshold);

    return 0;
}
