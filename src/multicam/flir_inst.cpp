#include "flir_inst.hpp"
#include <glog/logging.h>

flir::CameraList* FlirInstance::GetCameraList() {
	static FlirInstance inst_;
		return &inst_.m_CamList;
}

FlirInstance::FlirInstance() {
	m_pSys = flir::System::GetInstance();
	m_CamList = m_pSys->GetCameras();
}

FlirInstance::~FlirInstance() {
	m_CamList.Clear();
	m_pSys->ReleaseInstance();
}

template<typename _Ty>
_Ty GetNode(flir::GenApi::INodeMap &nodeMap, const std::string &strKey) {
	_Ty pNode = nodeMap.GetNode(strKey.c_str());
	CHECK(flir::GenApi::IsAvailable(pNode)) << strKey;
	CHECK(flir::GenApi::IsReadable(pNode)) << strKey;
	CHECK(flir::GenApi::IsWritable(pNode)) << strKey;
	return pNode;
}

void SetNodeParam(flir::GenApi::INodeMap &nodeMap, const std::string &strKey,
		const char *pValue) {
	auto pNode = GetNode<flir::GenApi::CEnumerationPtr>(nodeMap, strKey);
	auto pEntry = pNode->GetEntryByName(pValue);
	CHECK(flir::GenApi::IsAvailable(pEntry)) << pValue;
	CHECK(flir::GenApi::IsReadable(pEntry)) << pValue;
	pNode->SetIntValue(pEntry->GetValue());
}

void SetNodeParam(flir::GenApi::INodeMap &nodeMap, const std::string &strKey,
		int32_t nValue) {
	auto pNode = GetNode<flir::GenApi::CIntegerPtr>(nodeMap, strKey);
	pNode->SetValue(nValue);
}

void SetNodeParam(flir::GenApi::INodeMap &nodeMap, const std::string &strKey,
		bool bValue) {
	auto pNode = GetNode<flir::GenApi::CBooleanPtr>(nodeMap, strKey);
	pNode->SetValue(bValue);
}

void SetNodeParam(flir::GenApi::INodeMap &nodeMap, const std::string &strKey,
		float fValue) {
	auto pNode = GetNode<flir::GenApi::CFloatPtr>(nodeMap, strKey);
	pNode->SetValue(fValue);
}
