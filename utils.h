#include <opencv2/opencv.hpp>
#include <string>
#include "mixformer/mixformer_trt.h"


// converts a value to a string using string streams
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

cv::Rect2f mfbbox2rect(DrOBB mf_bbox);
DrOBB rect2mfbbox(cv::Rect2f bbox);
uint64_t calculateAverageHash(const cv::Mat& roi);