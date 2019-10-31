#ifndef __CAM_CONF_HPP
#define __CAM_CONF_HPP

#include <string>
#include "flir_inst.hpp"

class CameraProperties {
public:
	CameraProperties(flir::CameraPtr pCam);

	std::string GetModelType();

	std::string GetDeviceSN();

	void SetPixelFormat(const std::string &strPixelFormat);

	// Set to auto framerate if less than 0
	// Set to maximum framerate if less than 1.f
	void SetFrameRate(float fFrameRate);

	// "Software", "Line0", "Off"
	void SetTriggerDevice(const std::string &strTriggerDevice);

	// [1.f to 3.9] Auto WB if either less than 1.f
	void SetWhiteBalance(float fBlueRatio, float fRedRatio);

	// [0.0, 399.0] Auto Seturation if less than 0
	void SetSaturation(float fSaturation);

	// [10, ?] Auto Seturation if less than 0
	void SetExposure(float fMicroseconds);

	// [0.0, 24.0] Auto Gain if less than 0
	void SetGain(float fGain);

	// [0.5, 3.0] Turn off gamma if less than 0
	void SetGamma(float fGamma);

	// Resolution
	void SetResolution(int32_t nWidth, int32_t nHeight);

	void SetBufferSize(int32_t nSize);

private:
	enum NodeMapType {NORMAL, TLDEV, STREAM};
	flir::GenApi::INodeMap& __NodeMap(NodeMapType nmt);

	flir::CameraPtr m_pCam;
};

#endif //__CAM_CONF_HPP