#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <numeric>

#include <iomanip>

// #include <opencv2/opencv.hpp>

#include "kcftracker.hpp"
#include "mixformer_trt.h"

using namespace std;
using namespace cv;
using namespace kcf;
namespace fs = std::filesystem;

#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

uint64_t calculateAverageHash(const cv::Mat& roi) {
    // Resize the ROI to a fixed size (e.g., 8x8) for simplicity
    cv::Mat resized;
    cv::resize(roi, resized, cv::Size(16, 16));

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
    int frameNumber;
    float x, y, w, h;
    std::string s;
    ifstream *groundtruth;
    ostringstream osfile;
    std::string sequencesDir = "/home/uavlab20/tracking/Datasets/VisDrone2019-SOT-test-dev/sequences/";

    for (const auto & entry : fs::directory_iterator(sequencesDir))
    {
        osfile.str("");
        std::vector<float> fps_values;
        string sequenceName = entry.path().stem().string();
        cout << "Sequence name: " << sequenceName << endl; 
        groundtruth = new ifstream("/home/uavlab20/tracking/Datasets/VisDrone2019-SOT-test-dev/annotations/" + sequenceName + ".txt");

        frameNumber = 1;
        getline(*groundtruth, s, ',');
        x = atof(s.c_str());
        getline(*groundtruth, s, ',');
        y = atof(s.c_str());
        getline(*groundtruth, s, ',');
        w = atof(s.c_str());
        getline(*groundtruth, s);
        h = atof(s.c_str());
        osfile << sequencesDir << sequenceName << "/img" << setw(7) << setfill('0') << frameNumber << ".jpg";
        Rect2d bboxGroundtruth(x, y, w, h);

        cv::Mat frame = cv::imread(osfile.str().c_str(), IMREAD_UNCHANGED);
        
        if (!frame.data)
        {
            cout << "Could not open or find the image" << std::endl;
            return -1;
        }
        
        rectangle(frame, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);

        // KCF tracker declaration
        bool HOG = true, FIXEDWINDOW = true, MULTISCALE = true, LAB = true, DSST = false;
        KCFTracker kcftracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);

        // MixFormerV2 tracker declaration (with .engine model)
        std::string model_path = "/home/uavlab20/mfkcf/model/mixformer_v2_sim.engine";
        MixformerTRT *Mixformerer;
        Mixformerer = new MixformerTRT(model_path);
        
        Rect2d bbox((int)bboxGroundtruth.x, (int)bboxGroundtruth.y, (int)bboxGroundtruth.width, (int)bboxGroundtruth.height);
        
        string resultsDir = "/home/uavlab20/exp_mfkcf/mf_results_VisDrone/";
        string outputFilePath = resultsDir + sequenceName + ".txt";

        // Open output file for writing
        ofstream outputFile(outputFilePath);
        if (!outputFile.is_open())
        {
            cerr << "Could not create output file for sequence: " << sequenceName << endl;
            continue; 
        }

        bool trackerInitialized = false;
        
        // Mat previous_frame_bbox = frame(bbox);
        // Mat previous_frame = frame;

        // ofstream outputFile("differingBits.txt"); // output file for hash difference values 
        bool trackedMf = false;
        bool ok = true;


        while(frame.data)
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
            
            // if (trackedMf){
            //     kcftracker.init(frame, bbox);
            //     trackedMf = false;
            // }

            // Start timer for count fps
            double timer = (double)getTickCount();

            
            // Mat current_frame_bbox = frame(bbox);
            // Mat current_frame = frame;

            // uint64_t hashCurrentFrame = calculateAverageHash(current_frame_bbox);
            // uint64_t hashPreviousFrame = calculateAverageHash(previous_frame_bbox);
            // int differingBits = __builtin_popcountll(hashCurrentFrame ^ hashPreviousFrame);
            mf_bbox = Mixformerer->track(frame);


            // TODO: when there is no bbox - sample_target() doesn't work = error 
            bbox.x = int(mf_bbox.box.x0);
            bbox.y = int(mf_bbox.box.y0);
            bbox.width = int(mf_bbox.box.x1 - mf_bbox.box.x0);
            bbox.height = int(mf_bbox.box.y1 - mf_bbox.box.y0);
            
            // if (differingBits >= 0) {
            //     // std::cout << "Frame number: " << frameNumber << std::endl;
            //     // std::cout << "Switch to MixFormer" << std::endl;
            //     // std::cout << "Box before MF: " << bbox << std::endl;
                
            //     mf_bbox = Mixformerer->track(frame);
                
            //     bbox.x = int(mf_bbox.box.x0);
            //     bbox.y = int(mf_bbox.box.y0);
            //     bbox.width = int(mf_bbox.box.x1 - mf_bbox.box.x0);
            //     bbox.height = int(mf_bbox.box.y1 - mf_bbox.box.y0);
            //     // std::cout << "Switch to KCF" << std::endl;
            //     // std::cout << "Box after MF: " << bbox << std::endl;
            //     trackedMf = true;
            // } 
            // else {
            //     // Update the tracking result
            //     bool ok = kcftracker.update(frame, bbox);

            //     if (!ok && !trackerInitialized) {
            //         cout << "KCF tracker initialization failed!" << endl;
            //         break;
            //     }
            // }

            // Calculate Frames per second (FPS)
            float fps = getTickFrequency() / ((double)getTickCount() - timer);
            
            if (ok)
            {
                rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
                outputFile << int(bbox.x) << "," << int(bbox.y) << "," << int(bbox.width) << "," << int(bbox.height) << endl;
            }
            else
            {
                outputFile << "NaN,NaN,NaN,NaN" << endl;
                putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
            }
            rectangle(frame, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
            // Display FPS on frame
            putText(frame, "FPS KCF: " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(170,50,50), 2);

            // Display frame.
            imshow("Tracking", frame);

            int k = waitKey(1);
            if(k == 27)
            {
                break;
            }
            // previous_frame_bbox = frame(bbox).clone();   
            frameNumber++;

            osfile.str("");
           
            fps_values.push_back(fps);
            getline(*groundtruth, s, ',');
            x = atof(s.c_str());
            getline(*groundtruth, s, ',');
            y = atof(s.c_str());
            getline(*groundtruth, s, ',');
            w = atof(s.c_str());
            getline(*groundtruth, s);
            h = atof(s.c_str());

            osfile << sequencesDir << sequenceName << "/img" << setw(7) << setfill('0') << frameNumber << ".jpg";

            bboxGroundtruth.x = x;
            bboxGroundtruth.y = y;
            bboxGroundtruth.width = w;
            bboxGroundtruth.height = h;
            frame = cv::imread(osfile.str().c_str(), IMREAD_UNCHANGED);
        } 

        float mean_fps = std::accumulate(fps_values.begin(), fps_values.end(), 0.0) / fps_values.size();

        std::ofstream fps_file("/home/uavlab20/exp_mfkcf/fps/mf_VisDrone_fps_results.txt", std::ios::app); 

        if (fps_file.is_open())
        {
            fps_file << sequenceName << ", " << mean_fps << "\n";
            fps_file.close(); 
        }
        else
        {
            cerr << "Unable to open fps_results file for writing." << endl;
        }
        outputFile.close();

    }   
    
    return 0;
}