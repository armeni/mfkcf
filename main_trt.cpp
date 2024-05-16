#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>

#include "mixformer_trt.h"
#include "kcftracker.hpp"

using namespace std;
using namespace cv;
using namespace kcf;

namespace fs = std::filesystem;

cv::Rect2d mfbbox2rect(DrOBB mf_bbox){
    cv::Rect2d bbox;
    bbox.x = int(mf_bbox.box.x0);
    bbox.y = int(mf_bbox.box.y0);
    bbox.width = int(mf_bbox.box.x1 - mf_bbox.box.x0);
    bbox.height = int(mf_bbox.box.y1 - mf_bbox.box.y0);

    return bbox;
}

DrOBB rect2mfbbox(Rect2d bbox){
    DrOBB mf_bbox;
    mf_bbox.box.x0 = bbox.x;
    mf_bbox.box.x1 = bbox.x+bbox.width;
    mf_bbox.box.y0 = bbox.y;
    mf_bbox.box.y1 = bbox.y+bbox.height;
    return mf_bbox;
}

#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

uint64_t calculateAverageHash(const cv::Mat& roi) { 
    // Resize the ROI to a fixed size (e.g., 8x8) for simplicity
    cv::Mat resized;
    // cv::Mat gray;

    cv::resize(roi, resized, cv::Size(8, 8));
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

void track(MixformerTRT *tracker, KCFTracker *kcftracker)
{
    int frameNumber;
    float x, y, w, h;
    std::string s;
    ifstream *groundtruth;
    ostringstream osfile;

    // std::string sequencesDir = "/home/uavlab20/tracking/Datasets/UAV123/sequences/";
    // std::string sequencesDir = "/home/uavlab20/tracking/Datasets/VisDrone2019-SOT-train-2/sequences/";
    // std::string sequencesDir = "/home/uavlab20/tracking/Datasets/VisDrone2019-SOT-train-1/sequences/";
    std::string sequencesDir = "/home/uavlab20/tracking/Datasets/VisDrone2019-SOT-test-dev/sequences/";
    float percentOfMf = 0;
    float ss = 0;
    for (const auto & entry : fs::directory_iterator(sequencesDir))
    {
        osfile.str("");
        std::vector<float> fps_values;
        
        string sequenceName = entry.path().stem().string();
        std::cout << "Sequence name: " << sequenceName << endl; 
        ss++;
        // groundtruth = new ifstream("/home/uavlab20/tracking/Datasets/UAV123/anno/UAV123/" + sequenceName + ".txt");
        // groundtruth = new ifstream("/home/uavlab20/tracking/Datasets/VisDrone2019-SOT-train-2/annotations/" + sequenceName + ".txt");
        // groundtruth = new ifstream("/home/uavlab20/tracking/Datasets/VisDrone2019-SOT-train-1/annotations/" + sequenceName + ".txt");
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
        // osfile << sequencesDir << sequenceName << "/" << setw(6) << setfill('0') << frameNumber << ".jpg";

        Rect2d bboxGroundtruth(x, y, w, h);

        cv::Mat frame = cv::imread(osfile.str().c_str(), IMREAD_UNCHANGED);

        if (!frame.data)
        {
            std::cout<< "Could not open or find the image" << std::endl;
            return;
        }

        // rectangle(frame, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);

        cv::Rect2d bbox(int(bboxGroundtruth.x), int(bboxGroundtruth.y), int(bboxGroundtruth.width), int(bboxGroundtruth.height));

        string resultsDir = "/home/uavlab20/exp_mfkcf/mfkcf3pr_results_VisDrone/";
        string outputFilePath = resultsDir + sequenceName + ".txt";

        // Open output file for writing
        ofstream outputFile(outputFilePath);
        if (!outputFile.is_open())
        {
            cerr << "Could not create output file for sequence: " << sequenceName << endl;
            continue; 
        }

        bool trackerInitialized = false;
        Mat previous_frame_bbox = frame(bbox);
        Mat previous_frame = frame;
        bool trackedMf = false;
        bool ok = false;
        int numberOfMf = 0;
        bool terminateEarly = false;
        int bits = 0;

        while (frame.data && !terminateEarly)
        {
            if(!trackerInitialized){
                // Initialize tracker with first frame and rect.
                kcftracker->init(frame, bbox);
                tracker->init(frame, rect2mfbbox(bbox));   
                trackerInitialized = true;
            }

            // Start timer
            double timer = (double)cv::getTickCount();
            
            if (trackedMf){
                kcftracker->init(frame, bbox);
                trackedMf = false;
            }

            Mat current_frame_bbox = frame(bbox);
            Mat current_frame = frame;
            uint64_t hashCurrentFrame = calculateAverageHash(current_frame_bbox);
            uint64_t hashPreviousFrame = calculateAverageHash(previous_frame_bbox);
            int differingBits = __builtin_popcountll(hashCurrentFrame ^ hashPreviousFrame);
            // bits+=differingBits;

            if (differingBits > 10) {
                // std::cout << "Frame number: " << frameNumber << std::endl;
                // std::cout << "Switch to MixFormer" << std::endl;
                numberOfMf++;

                // Update tracker.
                bbox = mfbbox2rect(tracker->track(frame));
                
                trackedMf = true;
            } else {
                // Update the tracking result by KCF
                ok = kcftracker->update(frame, bbox);
                if (!ok && !trackerInitialized) {
                    std::cout << "KCF tracker initialization failed!" << std::endl;
                    break;
                }
            }
            
            // Calculate Frames per second (FPS)
            float fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);

            if (fps < 5){
                terminateEarly = true;
                int file_count = 0;
                for (const auto& entry : fs::directory_iterator(sequencesDir + sequenceName)) {
                    if (entry.is_regular_file()) {
                        ++file_count;
                    }
                }
                fps_values.insert(fps_values.end(), file_count - frameNumber, 0.0f);

            }            
            
            // Boundary judgment.
            // if (ok){
            cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2, 1);
            outputFile << int(bbox.x) << "," << int(bbox.y) << "," << int(bbox.width) << "," << int(bbox.height) << std::endl;
            // } else {
            //     outputFile << "NaN,NaN,NaN,NaN" << std::endl;
            //     putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
            // }

            // Display FPS on frame
            putText(frame, "FPS: " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(170,50,50), 2);

            // Display result.
            cv::imshow("Tracking", frame);

            // Exit if 'q' pressed.
            if (cv::waitKey(1) == 'q')
            {
                break;
            }

            if (bbox.x < 0){
                bbox.x = 0;
            } 
            if (bbox.y < 0){
                bbox.y = 0;
            }
            if (bbox.height < 4){
                bbox.height = 4;
            } 

            if (bbox.width < 4){
                bbox.width = 4;
            }
            // if (bbox.width / bbox.height > 8){
            //     bbox.width /= 2;
            // }
            if (bbox.x + bbox.width > frame.size().width){
                bbox.width = frame.size().width - bbox.x;
            } 
            if (bbox.y + bbox.height > frame.size().height){
                bbox.height = frame.size().height - bbox.y;
            }
            
            frameNumber++;

            if ((frameNumber - 1) % 3 ==0){
                previous_frame_bbox = frame(bbox).clone();   
            }
            fps_values.push_back(fps);
            osfile.str("");
            osfile << sequencesDir << sequenceName << "/img" << setw(7) << setfill('0') << frameNumber << ".jpg";
            // osfile << sequencesDir << sequenceName << "/" << setw(6) << setfill('0') << frameNumber << ".jpg";

            frame = cv::imread(osfile.str().c_str(), IMREAD_UNCHANGED);
        }

        float mean_fps = std::accumulate(fps_values.begin(), fps_values.end(), 0.0) / fps_values.size();

        std::ofstream fps_file("/home/uavlab20/exp_mfkcf/fps/mfkcf3pr_VisDrone_fps_results.txt", std::ios::app); 

        if (fps_file.is_open())
        {
            fps_file << sequenceName << ", " << mean_fps << "\n";
            fps_file.close(); 
        }
        else
        {
            cerr << "Unable to open fps_results file for writing." << endl;
        }
        percentOfMf += float(numberOfMf) / float(frameNumber - 1);
        cout << percentOfMf / ss << endl;
        outputFile.close();
    }
}


int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [modelpath] [videopath(file or camera)]\n", argv[0]);
        return -1;
    }

    // Get model path.
    std::string model_path = argv[1]; 


    // Build tracker.
    MixformerTRT *Mixformerer;
    Mixformerer = new MixformerTRT(model_path);

    bool HOG = true, FIXEDWINDOW = true, MULTISCALE = true, LAB = true, DSST = false;
    KCFTracker *kcftracker;
    kcftracker = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);


    track(Mixformerer, kcftracker);

    return 0;
}
