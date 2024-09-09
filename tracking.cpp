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



Rect2d selectObject(Mat& frame)
{
    // Use OpenCV's selectROI function to let the user select the object
    Rect2d bbox = selectROI("Select Object", frame, false, false);

    if (bbox.width == 0 || bbox.height == 0) {
        cerr << "Error: No object selected." << endl;
        return Rect2d(); // Return an empty rectangle
    }

    // Optionally, draw the rectangle on the frame to show the selection
    rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
    imshow("Select Object", frame);
    waitKey(0); // Wait for user to press any key after selection

    return bbox;
}


void track(MixformerTRT *tracker, KCFTracker *kcftracker, int bitsThreshold, const std::string& resultsFile, const std::string& fpsFile)
{
    // Open the default camera (index 0). Change index if necessary.
    VideoCapture cap(0); 

    if (!cap.isOpened()) {
        cerr << "Error: Unable to open camera." << endl;
        return;
    }

    Mat frame;
    Mat previousFrameBbox; // Variable to store the previous frame
    Rect2d bbox;
    bool trackerInitialized = false;
    bool mfTracked = false;
    bool ok = false;
    bool terminateEarly = false;
    vector<float> fpsValues;

    int frameNumber = 0;
    float mixformerTriggers = 0;
    float sequenceCount = 1; // Fixed value for a real-time stream

    ofstream outputFile(resultsFile);
    if (!outputFile.is_open()) {
        cerr << "Error: Could not create output file." << endl;
        return;
    }

    // Skip initial frames
    int startFrameNumber = 2;
    for (int i = 0; i < startFrameNumber; ++i) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Empty frame captured during frame skipping." << endl;
            return;
        }
    }

    while (!terminateEarly) {
        // Capture a frame from the camera
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Empty frame captured." << endl;
            break;
        }

        // Debug log frame number and size
        std::cout << "Processing frame: " << frameNumber << ", Size: " << frame.size() << std::endl;

        // Check if we are on frame 200
        if (frameNumber == 200) {
            std::cout << "Debug: Captured frame 200" << std::endl;
            imwrite("frame_200.jpg", frame); // Save the frame for inspection
        }
        // Initialize the tracker with the first frame and bounding box
        if (!trackerInitialized) {
            bbox = selectObject(frame); 
            if (bbox.width == 0 || bbox.height == 0) {
                cerr << "Error: No valid bounding box selected." << endl;
                return;
            }
            kcftracker->init(frame, bbox);
            tracker->init(frame, rect2mfbbox(bbox));
            previousFrameBbox = frame(bbox).clone(); // Initialize previous frame
            trackerInitialized = true;
            continue; // Skip the rest of the loop
        }

        // Check if the bounding box is within frame boundaries
        if (bbox.x < 0 || bbox.y < 0 || bbox.x + bbox.width > frame.cols || bbox.y + bbox.height > frame.rows) {
            cerr << "Error: Bounding box is out of frame boundaries." << endl;
            continue; // Skip this frame
        }

        // Start timer for FPS calculation
        double timer = (double)cv::getTickCount();

        // Check if we need to switch to MixFormer
        Mat currentFrameBbox = frame(bbox);
        uint64_t hashCurrentFrame = calculateAverageHash(currentFrameBbox);
        uint64_t hashPreviousFrame = calculateAverageHash(previousFrameBbox);
        int differingBits = __builtin_popcountll(hashCurrentFrame ^ hashPreviousFrame);

        if (differingBits >= bitsThreshold) {
            mixformerTriggers++;
            bbox = mfbbox2rect(tracker->track(frame));
            mfTracked = true;
            ok = true;
        } else {
            ok = kcftracker->update(frame, bbox);
        }

        // Calculate FPS
        float fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);

        // Display results on the frame
        if (ok) {
            rectangle(frame, bbox, Scalar(0, 255, 0), 2, 1);
            outputFile << int(bbox.x) << "," << int(bbox.y) << "," << int(bbox.width) << "," << int(bbox.height) << std::endl;
        } else {
            outputFile << "NaN, NaN, NaN, NaN" << std::endl;
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }

        putText(frame, "FPS: " + std::to_string(int(fps)), Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(170, 50, 50), 2);
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

        frameNumber++;
        fpsValues.push_back(fps);
        previousFrameBbox = frame(bbox).clone();
    }

    float meanFps = std::accumulate(fpsValues.begin(), fpsValues.end(), 0.0) / fpsValues.size();
    std::ofstream fps(fpsFile, std::ios::app);
    if (fps.is_open()) {
        fps << "Camera, " << meanFps << "\n";
        fps.close();
    } else {
        cerr << "Unable to open fps_results file for writing." << endl;
    }

    outputFile.close();
    cout << "Average MixFormer trigger rate: " << mixformerTriggers / (frameNumber - 1) << endl;
}


int main(int argc, char** argv)
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s [modelPath] [bitsThreshold] [resultsDir] [fpsFile]\n", argv[0]);
        return -1;
    }

    std::string modelPath = argv[1];
    int bitsThreshold = std::stoi(argv[2]);
    std::string resultsFile = argv[3];
    std::string fpsFile = argv[4];

    MixformerTRT *Mixformerer;
    Mixformerer = new MixformerTRT(modelPath);

    bool HOG = true, FIXEDWINDOW = true, MULTISCALE = true, LAB = true, DSST = false;
    KCFTracker *kcftracker;
    kcftracker = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);

    track(Mixformerer, kcftracker, bitsThreshold, resultsFile, fpsFile);

    return 0;
}
