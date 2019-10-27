#ifndef __CAM_CONF_HPP
#define __CAM_CONF_HPP

#include <string>
#include "flir_inst.hpp"

class CameraConfig {
public:
	CameraConfig(flir::GenApi::INodeMap &nodeMap);

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

private:
	flir::GenApi::INodeMap &m_NodeMap;
};

#endif //__CAM_CONF_HPP