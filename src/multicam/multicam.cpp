
#include "multicam/multicam.hpp"

#include <algorithm>
#include <fstream>
#include <mutex>
#include <thread>

#include <glog/logging.h>
#include <Spinnaker.h>
#include <SpinGenApi/SpinnakerGenApi.h>

#include "multicam/ctimer.hpp"
#include "json.hpp"
#include "trigger.hpp"
#include "cam_prop.hpp"
#include "flir_inst.hpp"
#include "post_proc.hpp"

class MultipleCameras::MultipleCamerasImpl {
public:
	MultipleCamerasImpl(const std::string &strTriggerDevice) {
		if (!strTriggerDevice.empty()) {
			m_pTrigger.reset(new Trigger(strTriggerDevice));
		}
	}

	~MultipleCamerasImpl() {
		__WaitingForWorker();
		__WaitingForTrigger();
		auto pCamList = FlirInstance::GetCameraList();
		for (uint32_t i = 0; i < pCamList->GetSize(); ++i) {
			auto pCam = pCamList->GetByIndex(i);
			if (pCam->IsInitialized()) {
				if (pCam->IsStreaming()) {
					pCam->EndAcquisition();
				}
				pCam->DeInit();
			}
		}
	}

	void Initialize(uint32_t nExpoMicroSec, const std::string &strConfRoot,
			std::vector<int> gpuIds) {
		uint32_t nCamCnt = GetCameraCount();
		CHECK_GT(nCamCnt, 0);
		if (gpuIds.empty()) {
			gpuIds.resize(nCamCnt, -1);
		} else {
			CHECK_EQ((int)gpuIds.size(), nCamCnt);
		}

		if (m_pTrigger != nullptr) {
			m_pTrigger->SetDelay(nExpoMicroSec);
		}

		m_PostProcs.clear();
		auto pCamList = FlirInstance::GetCameraList();
		for (uint32_t i = 0; i < nCamCnt; ++i) {
			__InitializeCamera(pCamList->GetByIndex(i),
					nExpoMicroSec, strConfRoot);
			m_PostProcs.emplace_back(gpuIds[i]);
		}
		(*m_pTrigger)(500 * 1000);
	}

	void GetImages(std::vector<cv::cuda::GpuMat> &images) {
		auto pCamList = FlirInstance::GetCameraList();
		for (uint32_t i = 0; i < pCamList->GetSize(); ++i) {
			CHECK(pCamList->GetByIndex(i)->IsInitialized())
					<< "Camera have not been initialized yet!";
		}
		__WaitingForWorker();
		if (m_ImgBuf.empty()) {
			__WorkerProc();
		}
		images.swap(m_ImgBuf);
		__CaptureAsync();
	}

	uint32_t GetCameraCount() const {
		return (uint32_t)FlirInstance::GetCameraList()->GetSize();
	}

	CAMERA_INFO GetCameraInfo(uint32_t iCam) const {
		auto pCamList = FlirInstance::GetCameraList();
		CHECK_LT(iCam, pCamList->GetSize());
		CameraProperties camProp(pCamList->GetByIndex(iCam));

		CAMERA_INFO camInfo;
		camInfo.strModelType = camProp.GetModelType();
		camInfo.strDeviceSN = camProp.GetDeviceSN();
		return camInfo;
	}

private:
	void __CaptureAsync() {
		m_WorkerThread = std::thread(&MultipleCamerasImpl::__WorkerProc, this);
	}

	void __WaitingForWorker() {
		if (m_WorkerThread.joinable()) {
			m_WorkerThread.join();
		}
	}

	void __WorkerProc() {
		auto nCaptureTimeOutMilliSec = 500;
		auto pCamList = FlirInstance::GetCameraList();
		auto nCamCnt = pCamList->GetSize();
		std::vector<flir::ImagePtr> rawImgPtrs;
		rawImgPtrs.resize(nCamCnt);

		__WaitingForTrigger();
		for (uint32_t i = 0; i < nCamCnt; ++i) {
			auto pCam = pCamList->GetByIndex(i);
			flir::ImagePtr pRawImg;
			try {
				rawImgPtrs[i] = pCam->GetNextImage(nCaptureTimeOutMilliSec);
			} catch (...) {
				rawImgPtrs.clear();
			}
		}
		__TriggerAsync();
		if (rawImgPtrs.size() == nCamCnt) {
			m_ImgBuf.resize(nCamCnt);
			//CTimer t;
			std::vector<std::thread> pool;
			for (uint32_t i = 0; i < nCamCnt; ++i) {
				pool.emplace_back([&](int idx) {
					if (rawImgPtrs[idx].IsValid()) {
						if (!rawImgPtrs[idx]->IsIncomplete()) {
							m_PostProcs[idx].Process(rawImgPtrs[idx], m_ImgBuf[idx]);
						}
						rawImgPtrs[idx]->Release();
					}
				}, i);
			}
			for (uint32_t i = 0; i < nCamCnt; ++i) {
				if (pool[i].joinable()) {
					pool[i].join();
				}
			}
			//LOG(INFO) << t.Reset();
		} else {
			LOG(INFO) << "Lost frame!";
		}
	}

	void __TriggerAsync() {
		m_TriggerThread = std::thread(&MultipleCamerasImpl::__TriggerProc, this);
	}

	void __WaitingForTrigger() {
		if (m_TriggerThread.joinable()) {
			m_TriggerThread.join();
		}
	}

	void __TriggerProc() {
		if (m_pTrigger != nullptr) {
			(*m_pTrigger)();
		}
	}

	void __InitializeCamera(flir::CameraPtr pCam, uint32_t nExpoMicroSec,
			const std::string &strConfRoot) {
		CHECK(!pCam->IsInitialized()) << "Double initialized!";
		pCam->Init();
		CameraProperties camConf(pCam);

		auto strDeviceModel = camConf.GetModelType();
		auto strDeviceSN = camConf.GetDeviceSN();
		LOG(INFO) << "Found camera: " << strDeviceModel
				  << '(' << strDeviceSN << ')';

		std::replace(strDeviceModel.begin(), strDeviceModel.end(), ' ', '_');
		std::ostringstream ossConfFileName;
		ossConfFileName << strConfRoot << "/" << strDeviceModel << ".json";
		nlohmann::json jConf;
		std::ifstream confFile(ossConfFileName.str());
		if(confFile.is_open()) {
			confFile >> jConf;
			LOG(INFO) << "Load configuration from \""
					  << ossConfFileName.str() << "\"";
		}

		for (auto jItem = jConf.begin(); jItem != jConf.end(); ++jItem) {
			if (jItem.key() == "Gamma") {
				camConf.SetGamma(jItem.value().get<float>());
			} else if (jItem.key() == "Saturation") {
				camConf.SetSaturation(jItem.value().get<float>());
			} else if (jItem.key() == "PixelFormat") {
				camConf.SetPixelFormat(jItem.value().get<std::string>());
			} else if (jItem.key() == "WhiteBalance") {
				std::vector<int32_t> values;
				for (auto v : jItem.value()) {
					values.push_back(v.get<int32_t>());
				}
				CHECK_EQ(values.size(), 2);
				camConf.SetWhiteBalance(values[0], values[1]);
			} else if (jItem.key() == "Gain") {
				camConf.SetGain(jItem.value().get<float>());
			} else if (jItem.key() == "Resolution") {
				std::vector<int32_t> values;
				for (auto v : jItem.value()) {
					values.push_back(v.get<int32_t>());
				}
				CHECK_EQ(values.size(), 2);
				camConf.SetResolution(values[0], values[1]);
			}
		}
		camConf.SetExposure(nExpoMicroSec);
		camConf.SetFrameRate(__GetConfigValue(jConf, "FrameRate", 0.f));
		if (m_pTrigger == nullptr) {
			camConf.SetTriggerDevice("Off");
		} else {
			if (m_pTrigger->GetDevice() == "Software") {
				camConf.SetTriggerDevice("Software");
			} else {
				camConf.SetTriggerDevice("Line0");
			}
		}
		camConf.SetBufferSize(1);
		pCam->BeginAcquisition();
	}

	template<typename _Ty>
	_Ty __GetConfigValue(nlohmann::json &jConf, const std::string &strKey,
			const _Ty &defaultValue) {
		auto jItem = jConf.find(strKey);
		if (jItem != jConf.end()) {
			return jItem->get<_Ty>();
		}
		return defaultValue;
	}

private:
	std::unique_ptr<Trigger> m_pTrigger;
	std::vector<cv::cuda::GpuMat> m_ImgBuf;
	std::thread m_WorkerThread;
	std::thread m_TriggerThread;
	std::vector<PostProcessor> m_PostProcs;
};

MultipleCameras::MultipleCameras(const std::string &strTriggerDevice) {
	m_pImpl = new MultipleCamerasImpl(strTriggerDevice);
}

MultipleCameras::~MultipleCameras() {
	delete m_pImpl;
}

void MultipleCameras::Initialize(uint32_t nExpoMicroSec,
		const std::string &strConfRoot, const std::vector<int> &gpuIds) {
	m_pImpl->Initialize(nExpoMicroSec, strConfRoot, gpuIds);
}

void MultipleCameras::GetImages(std::vector<cv::cuda::GpuMat> &images) {
	m_pImpl->GetImages(images);
}

uint32_t MultipleCameras::GetCameraCount() const {
	return m_pImpl->GetCameraCount();
}

CAMERA_INFO MultipleCameras::GetCameraInfo(uint32_t iCam) const {
	return m_pImpl->GetCameraInfo(iCam);
}
