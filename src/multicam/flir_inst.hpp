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

#endif