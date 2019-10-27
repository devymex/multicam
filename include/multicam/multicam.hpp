#ifndef __MULCAM_HPP
#define __MULCAM_HPP

#include <cstdint>
#include <vector>
#include <opencv2/opencv.hpp>

#include "trigger.hpp"

class MultipleCameras {
public:
	MultipleCameras(Trigger *pTrigger);

	~MultipleCameras();

	void Initialize(uint32_t nExpoMicroSec, float fGain = -1.f);

	void GetImages(std::vector<cv::Mat> &images);

private:
	class MultipleCamerasImpl;
	MultipleCamerasImpl *m_pImpl;
};

#endif