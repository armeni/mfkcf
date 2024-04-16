#include <iostream>
#include <fstream>
#include <string>

#include <iomanip>

// #include <opencv2/opencv.hpp>

#include "kcftracker.hpp"
#include "mixformer_trt.h"

using namespace std;
using namespace cv;
using namespace kcf;


#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

uint64_t calculateAverageHash(const cv::Mat& roi) {
    // Resize the ROI to a fixed size (e.g., 8x8) for simplicity
    cv::Mat resized;
    cv::resize(roi, resized, cv::Size(12, 12));

    // Convert the resized image to grayscale
    cv::cvtColor(resized, resized, cv::COLOR_BGR2GRAY);

    // Calculate the average pixel value
    int sum = 0;
    for (int i = 0; i < resized.rows; ++i) {
        for (int j = 0; j < resized.cols; ++j) {
            sum += resized.at<uchar>(i, j);
        }
    }

    // Calculate the average pixel value
    int average = sum / (resized.rows * resized.cols);

    // Generate the hash based on pixel values above/below average
    uint64_t hash = 0;
    for (int i = 0; i < resized.rows; ++i) {
        for (int j = 0; j < resized.cols; ++j) {
            hash <<= 1;
            if (resized.at<uchar>(i, j) >= average) {
                hash |= 1;
            }
        }
    }

    return hash;
}


int main(int argc, char **argv)
{
    // KCF tracker declaration
    bool HOG = true, FIXEDWINDOW = false, MULTISCALE = true, LAB = true, DSST = false;
    KCFTracker kcftracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);

    // MixFormerV2 tracker declaration (with .engine model)
    std::string model_path = "/home/uavlab20/mfkcf/model/mixformer_v2_sim.engine";
    MixformerTRT *Mixformerer;
    Mixformerer = new MixformerTRT(model_path);

    // first frame reading
    VideoCapture video("/home/uavlab20/Videos/car.mp4");
    Mat frame;
    video.read(frame);
    bool trackerInitialized = false;
    
    // getting bounding box
    Rect2d bbox(855, 325, 55, 20);

    // bbox = selectROI(frame, false);
    
    rectangle(frame, bbox, Scalar(255, 0, 0 ), 2, 1 ); // show bounding box on the first frame
    imshow("Tracking", frame);
    
    int frameNumber = 0;
    
    Mat previous_frame_bbox = frame(bbox);
    Mat previous_frame = frame;

    // ofstream outputFile("differingBits.txt"); // output file for hash difference values 
    bool trackedMf = false;
    bool ok = true;
    while(video.read(frame))
    {            
        DrOBB mf_bbox;  
        if (!trackerInitialized) {
            kcftracker.init(frame, bbox);
            trackerInitialized = true;
            
            mf_bbox.box.x0 = bbox.x;
            mf_bbox.box.x1 = bbox.x+bbox.width;
            mf_bbox.box.y0 = bbox.y;
            mf_bbox.box.y1 = bbox.y+bbox.height;
            
            Mixformerer->init(frame, mf_bbox);
        }
        
        if (trackedMf){
            kcftracker.init(frame, bbox);
            trackedMf = false;
        }

        // Start timer for count fps
        double timer = (double)getTickCount();

        
        Mat current_frame_bbox = frame(bbox);
        Mat current_frame = frame;

        uint64_t hashCurrentFrame = calculateAverageHash(current_frame_bbox);
        uint64_t hashPreviousFrame = calculateAverageHash(previous_frame_bbox);
        int differingBits = __builtin_popcountll(hashCurrentFrame ^ hashPreviousFrame);

        // outputFile << differingBits << endl; // output differingBits to the file
        
        if (differingBits > 25) {
            std::cout << "Frame number: " << frameNumber << std::endl;
            // std::cout << "Switch to MixFormer" << std::endl;
            // std::cout << "Box before MF: " << bbox << std::endl;
            
            mf_bbox = Mixformerer->track(frame);
            
            bbox.x = int(mf_bbox.box.x0);
            bbox.y = int(mf_bbox.box.y0);
            bbox.width = int(mf_bbox.box.x1 - mf_bbox.box.x0);
            bbox.height = int(mf_bbox.box.y1 - mf_bbox.box.y0);
            // std::cout << "Switch to KCF" << std::endl;
            // std::cout << "Box after MF: " << bbox << std::endl;
            trackedMf = true;
        } 
        else {
            // Update the tracking result
            bool ok = kcftracker.update(frame, bbox);

            if (!ok && !trackerInitialized) {
                cout << "KCF tracker initialization failed!" << endl;
                break;
            }
        }

        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
        
        if (ok)
        {
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        }
        else
        {
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }

        
        // Display FPS on frame
        putText(frame, "FPS KCF: " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(170,50,50), 2);

        // Display frame.
        imshow("Tracking", frame);

        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
        waitKey(10);
        
        previous_frame_bbox = frame(bbox).clone();   
        frameNumber++;
    }   

    return 0;
}