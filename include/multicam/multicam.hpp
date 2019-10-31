#ifndef __MULCAM_HPP
#define __MULCAM_HPP

#include <cstdint>
#include <vector>
#include <opencv2/opencv.hpp>

struct CAMERA_INFO {
	std::string strModelType;
	std::string strDeviceSN;
};

class MultipleCameras {
public:
	MultipleCameras(const std::string &strTriggerDevice = "");

	~MultipleCameras();

	void Initialize(uint32_t nExpoMicroSec, const std::string &strConfRoot);

	void GetImages(std::vector<cv::Mat> &images);

	uint32_t GetCameraCount() const;

	CAMERA_INFO GetCameraInfo(uint32_t iCam) const;

private:
	class MultipleCamerasImpl;
	MultipleCamerasImpl *m_pImpl;
};

#endif