#include "cam_prop.hpp"

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

template<>
class FlirPtr<std::string> {
public:
	using type = flir::GenApi::CStringPtr;
};

template<typename _Ty>
_Ty GetNode(flir::GenApi::INodeMap &nodeMap, const std::string &strKey) {
	_Ty pNode = nodeMap.GetNode(strKey.c_str());
	CHECK(flir::GenApi::IsAvailable(pNode)) << strKey;
	CHECK(flir::GenApi::IsReadable(pNode)) << strKey;
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
		_Ty &value) {
	auto pNode = GetNode<typename FlirPtr<_Ty>::type>(nodeMap, strKey);
	value = pNode->GetValue();
}

template<>
void GetParam(flir::GenApi::INodeMap &nodeMap, const std::string &strKey,
		std::string &value) {
	auto pNode = GetNode<typename FlirPtr<std::string>::type>(nodeMap, strKey);
	value = pNode->GetValue().c_str();
}

template<typename _Ty>
void GetParamMinMax(flir::GenApi::INodeMap &nodeMap, const std::string &strKey,
		_Ty &minVal, _Ty &maxVal) {
	auto pNode = GetNode<typename FlirPtr<_Ty>::type>(nodeMap, strKey);
	minVal = pNode->GetMin();
	maxVal = pNode->GetMax();
}

CameraProperties::CameraProperties(flir::CameraPtr pCam)
		: m_pCam(pCam) {
}

std::string CameraProperties::GetModelType() {
	std::string strModelType;
	GetParam(__NodeMap(TLDEV), "DeviceModelName", strModelType);
	return strModelType;
}

std::string CameraProperties::GetDeviceSN() {
	std::string strDeviceSN;
	GetParam(__NodeMap(TLDEV), "DeviceSerialNumber", strDeviceSN);
	return strDeviceSN;
}

void CameraProperties::SetPixelFormat(const std::string &strPixelFormat) {
	SetParam(__NodeMap(NORMAL), "PixelFormat", strPixelFormat.c_str());
}

void CameraProperties::SetTriggerDevice(const std::string &strTriggerDevice) {
	auto &nodeMap = __NodeMap(NORMAL);
	SetParam(nodeMap, "AcquisitionMode", "Continuous");
	if (strTriggerDevice == "Off") {
		SetParam(nodeMap, "TriggerMode", "Off");
	} else if (strTriggerDevice == "Software") {
		SetParam(nodeMap, "TriggerMode", "On");
		SetParam(nodeMap, "TriggerSource", "Software");
	} else if (strTriggerDevice == "Line0") {
		SetParam(nodeMap, "TriggerMode", "On");
		SetParam(nodeMap, "TriggerSource", "Line0");
	}
	//SetParam(nodeMap, "TriggerActivation", "RisingEdge");
}

// Set to maximum framerate if less than 0
void CameraProperties::SetFrameRate(float fFrameRate) {
	auto &nodeMap = __NodeMap(NORMAL);
	SetParam(nodeMap, "TriggerMode", "Off");
	std::string strKey = "AcquisitionFrameRateEnabled";
	if (!flir::GenApi::IsAvailable(nodeMap.GetNode(strKey.c_str()))) {
		strKey = "AcquisitionFrameRateEnable";
		CHECK(flir::GenApi::IsAvailable(nodeMap.GetNode(strKey.c_str())));
	}
	if (fFrameRate < 0.f) {
		SetParam(nodeMap, strKey, false);
		SetParam(nodeMap, "AcquisitionFrameRateAuto", "Continuous");
	} else {
		SetParam(nodeMap, strKey, true);
		if (strKey == "AcquisitionFrameRateEnabled") {
			SetParam(nodeMap, "AcquisitionFrameRateAuto", "Off");
		}
		if (fFrameRate < 1.f) {
			float fMin;
			GetParamMinMax(nodeMap, "AcquisitionFrameRate", fMin, fFrameRate);
			fFrameRate -= 0.01;
		}
		SetParam(nodeMap, "AcquisitionFrameRate", fFrameRate);
	}
}

void CameraProperties::SetWhiteBalance(float fBlueRatio, float fRedRatio) {
	auto &nodeMap = __NodeMap(NORMAL);
	if (fBlueRatio < 1.f || fRedRatio < 1.f) {
		SetParam(nodeMap, "BalanceWhiteAuto", "Continuous");
	} else {
		SetParam(nodeMap, "BalanceWhiteAuto", "Off");
		SetParam(nodeMap, "BalanceRatio", fBlueRatio);
		SetParam(nodeMap, "BalanceRatioSelector", "Blue");
		SetParam(nodeMap, "BalanceRatio", fRedRatio);
	}
}

void CameraProperties::SetSaturation(float fSaturation) {
	auto &nodeMap = __NodeMap(NORMAL);
	if (fSaturation < 0.f) {
		SetParam(nodeMap, "SaturationAuto", "Continuous");
	} else {
		SetParam(nodeMap, "SaturationAuto", "Off");
		SetParam(nodeMap, "Saturation", fSaturation);
	}
}

void CameraProperties::SetExposure(float fMicroseconds) {
	auto &nodeMap = __NodeMap(NORMAL);
	if (fMicroseconds < 0.f) {
		SetParam(nodeMap, "ExposureAuto", "Continuous");
	} else {
		SetParam(nodeMap, "ExposureAuto", "Off");
		SetParam(nodeMap, "ExposureTime", fMicroseconds);
		//SetParam(nodeMap, "ExposureTimeAbs", fMicroseconds);
	}
}

void CameraProperties::SetGain(float fGain) {
	auto &nodeMap = __NodeMap(NORMAL);
	if (fGain < 0.f) {
		float fMin, fMax;
		SetParam(nodeMap, "GainAuto", "Off");
		GetParamMinMax(nodeMap, "Gain", fMin, fMax);

		SetParam(nodeMap, "GainAuto", "Continuous");
		SetParam(nodeMap, "AutoGainLowerLimit", fMin);
		SetParam(nodeMap, "AutoGainUpperLimit", fMax);
	} else {
		SetParam(nodeMap, "GainAuto", "Off");
		SetParam(nodeMap, "Gain", fGain);
	}
}

void CameraProperties::SetGamma(float fGamma) {
	auto &nodeMap = __NodeMap(NORMAL);
	if (fGamma < 0.f) {
		SetParam(nodeMap, "GammaEnabled", false);
	} else {
		SetParam(nodeMap, "GammaEnabled", true);
		SetParam(nodeMap, "Gamma", fGamma);
	}
}

void CameraProperties::SetResolution(int32_t nWidth, int32_t nHeight) {
	auto &nodeMap = __NodeMap(NORMAL);
	if (nWidth == 0 || nHeight == 0) {
		int32_t nMaxWidth, nMaxHeight, nTmp;
		GetParamMinMax(nodeMap, "Width", nTmp, nMaxWidth);
		GetParamMinMax(nodeMap, "Height", nTmp, nMaxHeight);
		SetParam(nodeMap, "Width", nMaxWidth);
		SetParam(nodeMap, "Height", nMaxHeight);
	} else {
		SetParam(nodeMap, "Width", nWidth);
		SetParam(nodeMap, "Height", nHeight);
	}
}

void CameraProperties::SetBufferSize(int32_t nSize) {
	auto &nodeMap = __NodeMap(STREAM);
	SetParam(nodeMap, "StreamBufferCountMode", "Manual");
	SetParam(nodeMap, "StreamBufferCountManual", nSize);
}

flir::GenApi::INodeMap& CameraProperties::__NodeMap(NodeMapType nmt) {
	switch (nmt) {
	case NORMAL: return m_pCam->GetNodeMap();
	case TLDEV: return m_pCam->GetTLDeviceNodeMap();
	case STREAM: return m_pCam->GetTLStreamNodeMap();
	}
}