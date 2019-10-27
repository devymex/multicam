
#include "multicam/multicam.hpp"

#include <condition_variable>
#include <mutex>
#include <thread>

#include <glog/logging.h>
#include <Spinnaker.h>
#include <SpinGenApi/SpinnakerGenApi.h>

#include "multicam/ctimer.hpp"
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

	void Initialize(uint32_t nExpoMicroSec) {
		m_nExpoMicroSec = nExpoMicroSec;
		auto pCamList = FlirInstance::GetCameraList();
		CHECK_GT(pCamList->GetSize(), 0);
		for (uint32_t i = 0; i < pCamList->GetSize(); ++i) {
			auto pCam = pCamList->GetByIndex(i);
			CHECK(!pCam->IsInitialized()) << "Double initialized!";
			pCam->Init();
			auto &nodeMap = pCam->GetNodeMap();

			CameraConfig camConf(nodeMap);
			camConf.SetFrameRate(0.f);
			camConf.SetWhiteBalance(-1.f, -1.f);
			camConf.SetSaturation(100.f);
			camConf.SetExposure((float)nExpoMicroSec);
			camConf.SetGain(15.f);
			camConf.SetGamma(1.f);
			if (m_pTrigger == nullptr) {
				camConf.SetTriggerDevice("Off");
			} else {
				if (m_pTrigger->GetDevice() == "Software") {
					camConf.SetTriggerDevice("Software");
				} else {
					camConf.SetTriggerDevice("Line0");
				}
			}
			//SetParam(pCam->GetTLStreamNodeMap(), "StreamBufferCountMode", "Manual");
			//SetParam(pCam->GetTLStreamNodeMap(), "StreamBufferCountManual", 1);
			pCam->BeginAcquisition();
		}
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

		if (m_pTrigger != nullptr) {
			auto nRealExpoMicroSec = (int32_t)(m_ExpTimer.Now() * 1000 * 1000);
			usleep(std::max(0, (int32_t)m_nExpoMicroSec - nRealExpoMicroSec));
			(*m_pTrigger)();
		}
		for (uint32_t i = 0; i < nCamCnt; ++i) {
			auto pCam = pCamList->GetByIndex(i);
			flir::ImagePtr pRawImg;
			try {
				rawImgPtrs[i] = pCam->GetNextImage(nCaptureTimeOutMilliSec);
			} catch (...) {
				rawImgPtrs.clear();
			}
		}
		m_ExpTimer.Reset();
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

private:
	std::unique_ptr<Trigger> m_pTrigger;
	std::vector<cv::Mat> m_ImgBuf;
	std::thread m_WorkerThread;

	uint32_t m_nExpoMicroSec { 10000 };
	CTimer m_ExpTimer;
};

MultipleCameras::MultipleCameras(const std::string &strTriggerDevice) {
	m_pImpl = new MultipleCamerasImpl(strTriggerDevice);
}

MultipleCameras::~MultipleCameras() {
	delete m_pImpl;
}

void MultipleCameras::Initialize(uint32_t nExpoMicroSec) {
	m_pImpl->Initialize(nExpoMicroSec);
}

void MultipleCameras::GetImages(std::vector<cv::Mat> &images) {
	m_pImpl->GetImages(images);
}
