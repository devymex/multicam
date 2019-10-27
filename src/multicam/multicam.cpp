
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
#include "cam_conf.hpp"
#include "flir_inst.hpp"


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

	void Initialize(uint32_t nExpoMicroSec, const std::string &strConfRoot) {
		if (m_pTrigger != nullptr) {
			m_pTrigger->SetDelay(nExpoMicroSec);
		}
		auto pCamList = FlirInstance::GetCameraList();
		CHECK_GT(pCamList->GetSize(), 0);
		for (uint32_t i = 0; i < pCamList->GetSize(); ++i) {
			__InitializeCamera(pCamList->GetByIndex(i),
					nExpoMicroSec, strConfRoot);
		}
		__Trigger();
	}

	void GetImages(std::vector<cv::Mat> &images) {
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
		__DoNextAsync();
	}

private:

	void __DoNextAsync() {
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
		m_ImgBuf.clear();
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
		__Trigger();
		if (rawImgPtrs.size() == nCamCnt) {
			m_ImgBuf.resize(nCamCnt);
			//pragma omp parallel for
			for (uint32_t i = 0; i < nCamCnt; ++i) {
				if (rawImgPtrs[i].IsValid()) {
					if (!rawImgPtrs[i]->IsIncomplete()) {
						auto pBgrImg = rawImgPtrs[i]->Convert(
								flir::PixelFormat_BGR8, flir::HQ_LINEAR);
						cv::Mat img(pBgrImg->GetHeight(), pBgrImg->GetWidth(),
									CV_8UC3, pBgrImg->GetData());
						m_ImgBuf[i] = img.clone();
					}
					rawImgPtrs[i]->Release();
				}
			}
		} else {
			LOG(INFO) << "Lost frame!";
		}
	}

	void __Trigger() {
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

		flir::GenApi::CStringPtr ptrDeviceModelName =
				pCam->GetTLDeviceNodeMap().GetNode("DeviceModelName");
		std::string strDeviceModel = ptrDeviceModelName->ToString().c_str();
		std::replace(strDeviceModel.begin(), strDeviceModel.end(), ' ', '_');
		std::string strConfFileName = strConfRoot + "/"
				+ strDeviceModel + ".json";
		nlohmann::json jConf;
		std::ifstream confFile(strConfFileName);
		if(confFile.is_open()) {
			confFile >> jConf;
		}
		pCam->Init();
		CameraConfig camConf(pCam->GetNodeMap());
		for (auto jItem = jConf.begin(); jItem != jConf.end(); ++jItem) {
			if (jItem.key() == "Gamma") {
				camConf.SetGamma(jItem.value().get<float>());
			} else if (jItem.key() == "Saturation") {
				camConf.SetSaturation(jItem.value().get<float>());
			} else if (jItem.key() == "PixelFormat") {
				camConf.SetPixelFormat(jItem.value().get<std::string>());
			} else if (jItem.key() == "WhiteBalance") {
				auto fWhiteBalance = jItem.value().get<float>();
				camConf.SetWhiteBalance(fWhiteBalance, fWhiteBalance);
			} else if (jItem.key() == "Gain") {
				camConf.SetGain(jItem.value().get<float>());
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
	std::vector<cv::Mat> m_ImgBuf;
	std::thread m_WorkerThread;
	std::thread m_TriggerThread;
};

MultipleCameras::MultipleCameras(const std::string &strTriggerDevice) {
	m_pImpl = new MultipleCamerasImpl(strTriggerDevice);
}

MultipleCameras::~MultipleCameras() {
	delete m_pImpl;
}

void MultipleCameras::Initialize(uint32_t nExpoMicroSec,
		const std::string &strConfRoot) {
	m_pImpl->Initialize(nExpoMicroSec, strConfRoot);
}

void MultipleCameras::GetImages(std::vector<cv::Mat> &images) {
	m_pImpl->GetImages(images);
}
