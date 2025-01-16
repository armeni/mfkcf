#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include "mixformer/mixformer_trt.h"
#include "kcf/kcftracker.hpp"
#include <utils.h>
#include <cstdio>

using namespace std;
using namespace cv;
using namespace kcf;
namespace py = pybind11;

class TrackerMain {
public:
    TrackerMain() : threshold(5) {}
    TrackerMain(int threshold) : threshold(threshold) {}
    void init() {
        freopen("log.txt", "w", stderr);
        std::string modelPath = "/home/orin2/mfkcf/models/mixformer_v2.engine";
        tracker = new MixformerTRT(modelPath);  

        bool HOG = true, FIXEDWINDOW = true, MULTISCALE = true, LAB = true, DSST = false;
        kcftracker = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);  
    }

    std::vector<int> track(cv::Mat img, bool is_first = false, std::vector<int> first_box = {}) {
        if (is_first) {
            // Initialize the tracker with the provided first box
            std::vector<int> cfg = {
                first_box[0], first_box[1], 
                first_box[2] - first_box[0], 
                first_box[3] - first_box[1]
            };

            bbox = cv::Rect2f(cfg[0], cfg[1], cfg[2], cfg[3]);
            
            tracker->init(img, rect2mfbbox(bbox));
            kcftracker->init(img, bbox);
            previousFrameBbox = img(bbox);
            
            return {first_box[0], first_box[1], first_box[2], first_box[3]}; 
        } else {
            // reinitialize KCF tracker after MixFormer tracking step
            if (mfTracked && threshold > 0){
                kcftracker->init(img, bbox);
                mfTracked = false;
            }

            Mat currentFrameBbox = img(bbox);
            uint64_t hashCurrentFrame = calculateAverageHash(currentFrameBbox);
            uint64_t hashPreviousFrame = calculateAverageHash(previousFrameBbox);
            // Compute the XOR between the current and previous frame hashes to сount the number of bits that have changed
            int differingBits = __builtin_popcountll(hashCurrentFrame ^ hashPreviousFrame);
            
            if (differingBits >= threshold) {
                bbox = mfbbox2rect(tracker->track(img));
                mfTracked = true;
            } else {
                bbox = kcftracker->update(img, bbox);
            }

            bbox.x = std::max(0.f, bbox.x);
            bbox.y = std::max(0.f, bbox.y);
            bbox.width = std::max(10.f, bbox.width);
            bbox.height = std::max(10.f, bbox.height);

            if (bbox.x + bbox.width > img.size().width){
                bbox.width = img.size().width - bbox.x;
            }

            if (bbox.y + bbox.height > img.size().height){
                bbox.height = img.size().height - bbox.y;
            }

            previousFrameBbox = currentFrameBbox;

            return {
                static_cast<int>(bbox.x),
                static_cast<int>(bbox.y),
                static_cast<int>(bbox.x + bbox.width),
                static_cast<int>(bbox.y + bbox.height)
            };
        }
    }

private:
    MixformerTRT* tracker;  
    KCFTracker* kcftracker; 
    bool mfTracked = false;
    cv::Mat previousFrameBbox;
    cv::Mat img;
    cv::Rect2f bbox;
    int threshold; 
};


cv::Mat numpy_to_cv_mat(py::array_t<uint8_t> img) {
    py::buffer_info buf_info = img.request();
    
    if (buf_info.ndim != 3 || buf_info.shape[2] != 3) {
        throw std::runtime_error("Expected a 3-channel image");
    }
    
    cv::Mat mat(buf_info.shape[0], buf_info.shape[1], CV_8UC3, buf_info.ptr);
    return mat;
}

PYBIND11_MODULE(tracker_module, m) {
    py::class_<TrackerMain>(m, "TrackerMain")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("threshold"))
        .def("init", &TrackerMain::init)
        .def("track", [](TrackerMain& self, py::array_t<uint8_t> img, bool is_first, std::vector<int> first_box) {
            cv::Mat image_cv = numpy_to_cv_mat(img); 
            return self.track(cv::Mat(image_cv), is_first, first_box);
        });
}