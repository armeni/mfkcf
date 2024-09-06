#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>

#include "mixformer/mixformer_trt.h"
#include "kcf/kcftracker.hpp"
#include <utils.h>

using namespace std;
using namespace cv;
using namespace kcf;

namespace fs = std::filesystem;

void track(MixformerTRT *tracker, KCFTracker *kcftracker, int bitsThreshold, const std::string& sequencesDir, const std::string& annotationsDir, const std::string& resultsDir, const std::string& fpsFile)
{
    int frameNumber;
    float x, y, w, h;
    std::string s;
    ifstream *groundtruth;
    ostringstream osfile;
    float mfTriggerRate = 0;
    float sequenceCount = 0; 

    for (const auto & entry : fs::directory_iterator(sequencesDir))
    {
        osfile.str("");
        std::vector<float> fpsValues;
        frameNumber = 1;
        sequenceCount++;
    
        string sequenceName = entry.path().stem().string();
        std::cout << "Sequence name: " << sequenceName << endl; 
        groundtruth = new ifstream(annotationsDir + sequenceName + ".txt");
            
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
            std::cout<< "Could not open or find the image" << std::endl;
            return;
        }

        // rectangle(frame, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);

        cv::Rect2d bbox(int(bboxGroundtruth.x), int(bboxGroundtruth.y), int(bboxGroundtruth.width), int(bboxGroundtruth.height));
        string outputFilePath = resultsDir + sequenceName + ".txt";

        // Open output file for writing
        ofstream outputFile(outputFilePath);

        if (!outputFile.is_open())
        {
            cerr << "Could not create output file for sequence: " << sequenceName << endl;
            continue; 
        }

        bool trackerInitialized = false;
        bool mfTracked = false;
        bool ok = false;
        bool terminateEarly = false;
        Mat previousFrameBbox = frame(bbox);
        Mat previousFrame = frame;
        int mixformerTriggers = 0;

        while (frame.data && !terminateEarly)
        {
            if(!trackerInitialized){
                // Initialize trackers with first frame and rectangle
                kcftracker->init(frame, bbox);
                tracker->init(frame, rect2mfbbox(bbox));   
                trackerInitialized = true;
            }

            // Start timer
            double timer = (double)cv::getTickCount();
            
            // reinitialize KCF tracker after MixFormer tracking step
            if (mfTracked){
                kcftracker->init(frame, bbox);
                mfTracked = false;
            }

            Mat currentFrameBbox = frame(bbox);
            Mat currentFrame = frame;           
            uint64_t hashCurrentFrame = calculateAverageHash(currentFrameBbox);
            uint64_t hashPreviousFrame = calculateAverageHash(previousFrameBbox);
            // Compute the XOR between the current and previous frame hashes to сount the number of bits that have changed
            int differingBits = __builtin_popcountll(hashCurrentFrame ^ hashPreviousFrame); 

            if (differingBits >= bitsThreshold) {
                // std::cout << "Switch to MixFormer" << std::endl;
                mixformerTriggers++;
                bbox = mfbbox2rect(tracker->track(frame));
                mfTracked = true;
                ok = true;
            } else {
                ok = kcftracker->update(frame, bbox);
                if (!ok && !trackerInitialized) {
                    std::cout << "KCF tracker initialization failed!" << std::endl;
                    break;
                }
            }
            
            // Calculate Frames per second (FPS)
            float fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);

            // Stops tracking if the FPS drops below 1 and fills the remaining frames with 0 FPS values
            if (fps < 1){
                terminateEarly = true;
                int frameCount = 0;
                for (const auto& entry : fs::directory_iterator(sequencesDir + sequenceName)) {
                    if (entry.is_regular_file()) {
                        ++frameCount;
                    }
                }
                fpsValues.insert(fpsValues.end(), frameCount - frameNumber, 0.0f);

            }            

            if (ok){
                cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2, 1);
                outputFile << int(bbox.x) << "," << int(bbox.y) << "," << int(bbox.width) << "," << int(bbox.height) << std::endl;
            } else {
                outputFile << "NaN, NaN, NaN, NaN" << std::endl;
                putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
            }

            putText(frame, "FPS: " + SSTR(int(fps)), Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(170, 50, 50), 2);
            cv::imshow("Tracking", frame);

            if (cv::waitKey(1) == 'q')
            {
                break;
            }

            // Boundary judgement
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

            previousFrameBbox = frame(bbox).clone(); // remember this bbox for differingBits calculation on the next frame   
            fpsValues.push_back(fps);

            osfile.str("");
            osfile << sequencesDir << sequenceName << "/img" << setw(7) << setfill('0') << frameNumber << ".jpg";
            frame = cv::imread(osfile.str().c_str(), IMREAD_UNCHANGED); // read the next frame
        }

        float meanFps = std::accumulate(fpsValues.begin(), fpsValues.end(), 0.0) / fpsValues.size();
        std::ofstream fps(fpsFile, std::ios::app);  

        if (fps.is_open())
        {
            fps << sequenceName << ", " << meanFps << "\n";
            fps.close(); 
        }
        else
        {
            cerr << "Unable to open fps_results file for writing." << endl;
        }

        outputFile.close();
        mfTriggerRate += float(mixformerTriggers) / float(frameNumber - 1);
    }
    cout << "Average MixFormer trigger rate: " << mfTriggerRate / sequenceCount << endl;
}


int main(int argc, char** argv)
{
    if (argc != 7)
    {
        fprintf(stderr, "Usage: %s [modelPath] [bitsThreshold] [sequencesDir] [annotationsDir] [resultsDir] [fpsFile]\n", argv[0]);
        return -1;
    }

    std::string modelPath = argv[1];
    int bitsThreshold = std::stoi(argv[2]);
    std::string sequencesDir = argv[3];
    std::string annotationsDir = argv[4];
    std::string resultsDir = argv[5];
    std::string fpsFile = argv[6];


    MixformerTRT *Mixformerer;
    Mixformerer = new MixformerTRT(modelPath);

    bool HOG = true, FIXEDWINDOW = true, MULTISCALE = true, LAB = true, DSST = false;
    KCFTracker *kcftracker;
    kcftracker = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);


    track(Mixformerer, kcftracker, bitsThreshold, sequencesDir, annotationsDir, resultsDir, fpsFile);

    return 0;
}
