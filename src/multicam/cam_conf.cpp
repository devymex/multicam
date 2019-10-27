#include "cam_conf.hpp"

#include <glog/logging.h>

template<typename _Pointer>
class FlirPtr {
};

template<>
class FlirPtr<int32_t> {
public:
	using type = flir::GenApi::CIntegerPtr;
};

template<>
class FlirPtr<float> {
public:
	using type = flir::GenApi::CFloatPtr;
};

template<>
class FlirPtr<bool> {
public:
	using type = flir::GenApi::CBooleanPtr;
};

template<typename _Ty>
_Ty GetNode(flir::GenApi::INodeMap &nodeMap, const std::string &strKey) {
	_Ty pNode = nodeMap.GetNode(strKey.c_str());
	CHECK(flir::GenApi::IsAvailable(pNode)) << strKey;
	CHECK(flir::GenApi::IsReadable(pNode)) << strKey;
	CHECK(flir::GenApi::IsWritable(pNode)) << strKey;
	return pNode;
}

template<typename _Ty>
void SetParam(flir::GenApi::INodeMap &nodeMap, const std::string &strKey,
		_Ty nValue) {
	auto pNode = GetNode<typename FlirPtr<_Ty>::type>(nodeMap, strKey);
	pNode->SetValue(nValue);
}

template<>
void SetParam<const char*>(flir::GenApi::INodeMap &nodeMap,
		const std::string &strKey, const char* pValue) {
	auto pNode = GetNode<flir::GenApi::CEnumerationPtr>(nodeMap, strKey);
	auto pEntry = pNode->GetEntryByName(pValue);
	CHECK(flir::GenApi::IsAvailable(pEntry)) << pValue;
	CHECK(flir::GenApi::IsReadable(pEntry)) << pValue;
	pNode->SetIntValue(pEntry->GetValue());
}

template<typename _Ty>
void GetParam(flir::GenApi::INodeMap &nodeMap, const std::string &strKey,
		_Ty &nValue) {
	auto pNode = GetNode<typename FlirPtr<_Ty>::type>(nodeMap, strKey);
	nValue = pNode->GetValue();
}

template<typename _Ty>
void GetParamMinMax(flir::GenApi::INodeMap &nodeMap, const std::string &strKey,
		_Ty &minVal, _Ty &maxVal) {
	auto pNode = GetNode<typename FlirPtr<_Ty>::type>(nodeMap, strKey);
	minVal = pNode->GetMin();
	maxVal = pNode->GetMax();
}

CameraConfig::CameraConfig(flir::GenApi::INodeMap &nodeMap)
		: m_NodeMap(nodeMap) {
}

void CameraConfig::SetPixelFormat(const std::string &strPixelFormat) {
	SetParam(m_NodeMap, "PixelFormat", strPixelFormat.c_str());
}

void CameraConfig::SetTriggerDevice(const std::string &strTriggerDevice) {
	SetParam(m_NodeMap, "AcquisitionMode", "Continuous");
	if (strTriggerDevice == "Off") {
		SetParam(m_NodeMap, "TriggerMode", "Off");
	} else if (strTriggerDevice == "Software") {
		SetParam(m_NodeMap, "TriggerMode", "On");
		SetParam(m_NodeMap, "TriggerSource", "Software");
	} else if (strTriggerDevice == "Line0") {
		SetParam(m_NodeMap, "TriggerMode", "On");
		SetParam(m_NodeMap, "TriggerSource", "Line0");
	}
	//SetParam(m_NodeMap, "TriggerActivation", "RisingEdge");
}

// Set to maximum framerate if less than 0
void CameraConfig::SetFrameRate(float fFrameRate) {
	SetParam(m_NodeMap, "TriggerMode", "Off");
	std::string strKey = "AcquisitionFrameRateEnabled";
	if (!flir::GenApi::IsAvailable(m_NodeMap.GetNode(strKey.c_str()))) {
		strKey = "AcquisitionFrameRateEnable";
		CHECK(flir::GenApi::IsAvailable(m_NodeMap.GetNode(strKey.c_str())));
	}
	if (fFrameRate < 0.f) {
		SetParam(m_NodeMap, strKey, false);
		SetParam(m_NodeMap, "AcquisitionFrameRateAuto", "Continuous");
	} else {
		SetParam(m_NodeMap, strKey, true);
		if (strKey == "AcquisitionFrameRateEnabled") {
			SetParam(m_NodeMap, "AcquisitionFrameRateAuto", "Off");
		}
		if (fFrameRate < 1.f) {
			float fMin;
			GetParamMinMax(m_NodeMap, "AcquisitionFrameRate", fMin, fFrameRate);
			fFrameRate -= 0.01;
		}
		SetParam(m_NodeMap, "AcquisitionFrameRate", fFrameRate);
	}
}

void CameraConfig::SetWhiteBalance(float fBlueRatio, float fRedRatio) {
	if (fBlueRatio < 1.f || fRedRatio < 1.f) {
		SetParam(m_NodeMap, "BalanceWhiteAuto", "Continuous");
	} else {
		SetParam(m_NodeMap, "BalanceWhiteAuto", "Off");
		SetParam(m_NodeMap, "BalanceRatio", fBlueRatio);
		SetParam(m_NodeMap, "BalanceRatioSelector", "Blue");
		SetParam(m_NodeMap, "BalanceRatio", fRedRatio);
	}
}

void CameraConfig::SetSaturation(float fSaturation) {
	if (fSaturation < 0.f) {
		SetParam(m_NodeMap, "SaturationAuto", "Continuous");
	} else {
		SetParam(m_NodeMap, "ExposureAuto", "Off");
		SetParam(m_NodeMap, "Saturation", fSaturation);
	}
}

void CameraConfig::SetExposure(float fMicroseconds) {
	if (fMicroseconds < 0.f) {
		SetParam(m_NodeMap, "ExposureAuto", "Continuous");
	} else {
		SetParam(m_NodeMap, "ExposureAuto", "Off");
		SetParam(m_NodeMap, "ExposureTime", fMicroseconds);
		//SetParam(m_NodeMap, "ExposureTimeAbs", fMicroseconds);
	}
}

void CameraConfig::SetGain(float fGain) {
	if (fGain < 0.f) {
		float fMin, fMax;
		SetParam(m_NodeMap, "GainAuto", "Off");
		GetParamMinMax(m_NodeMap, "Gain", fMin, fMax);

		SetParam(m_NodeMap, "GainAuto", "Continuous");
		SetParam(m_NodeMap, "AutoGainLowerLimit", fMin);
		SetParam(m_NodeMap, "AutoGainUpperLimit", fMax);
	} else {
		SetParam(m_NodeMap, "GainAuto", "Off");
		SetParam(m_NodeMap, "Gain", fGain);
	}
}

void CameraConfig::SetGamma(float fGamma) {
	if (fGamma < 0.f) {
		SetParam(m_NodeMap, "GammaEnabled", false);
	} else {
		SetParam(m_NodeMap, "GammaEnabled", true);
		SetParam(m_NodeMap, "Gamma", fGamma);
	}
}
