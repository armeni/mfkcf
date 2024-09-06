#include <iostream>
#include <opencv2/opencv.hpp>
#include "mixformer/mixformer_trt.h"

using namespace cv;


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


/**
 * @brief Computes a 64-bit average hash of an image's grayscale values.
 * 
 * This function resizes the input ROI to 8x8, converts it to grayscale, 
 * and generates a hash by comparing each pixel to the average intensity.
 * 
 * @param image The input image (color, BGR) as a cv::Mat.
 * @return uint64_t A 64-bit hash value.
 * 
 * @note The input should be a color image; it's converted to grayscale internally.
 */
uint64_t calculateAverageHash(const cv::Mat& image) { 
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(8, 8));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2GRAY);

    // Calculate the average pixel value
    int sum = 0;
    for (int i = 0; i < resized.rows; ++i) {
        for (int j = 0; j < resized.cols; ++j) {
            sum += resized.at<uchar>(i, j);
        }
    }

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