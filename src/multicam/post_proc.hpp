#ifndef __POST_PROC_HPP
#define __POST_PROC_HPP

#include <opencv2/opencv.hpp>
#include "flir_inst.hpp"

cv::Mat PostProcess(flir::ImagePtr pImg);

#endif
