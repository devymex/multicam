#ifndef __MULCAM_HPP
#define __MULCAM_HPP

#include <cstdint>
#include <vector>
#include <opencv2/opencv.hpp>

class MultipleCameras {
public:
	MultipleCameras(const std::string &strTriggerDevice = "");

	~MultipleCameras();

	void Initialize(uint32_t nExpoMicroSec);

	void GetImages(std::vector<cv::Mat> &images);

private:
	class MultipleCamerasImpl;
	MultipleCamerasImpl *m_pImpl;
};

#endif