#ifndef __FLIR_INST_HPP
#define __FLIR_INST_HPP

#include <Spinnaker.h>
#include <SpinGenApi/SpinnakerGenApi.h>

namespace flir = Spinnaker;

class FlirInstance {
public:
	static flir::CameraList* GetCameraList();

private:
	FlirInstance();

	~FlirInstance();

private:
	flir::CameraList m_CamList;
	flir::SystemPtr m_pSys;
};

template<typename _Ty>
_Ty GetNode(flir::GenApi::INodeMap &nodeMap, const std::string &strKey);

void SetNodeParam(flir::GenApi::INodeMap &nodeMap, const std::string &strKey,
		const char *pValue);

void SetNodeParam(flir::GenApi::INodeMap &nodeMap, const std::string &strKey,
		int32_t nValue);

void SetNodeParam(flir::GenApi::INodeMap &nodeMap, const std::string &strKey,
		bool bValue);

void SetNodeParam(flir::GenApi::INodeMap &nodeMap, const std::string &strKey,
		float fValue);

#endif