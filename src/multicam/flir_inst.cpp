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
