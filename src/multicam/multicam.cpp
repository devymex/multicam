
#include "multicam/multicam.hpp"

#include <condition_variable>
#include <mutex>
#include <thread>

#include <glog/logging.h>
#include <Spinnaker.h>
#include <SpinGenApi/SpinnakerGenApi.h>

#include "multicam/ctimer.hpp"
#include "flir_inst.hpp"

class MultipleCameras::MultipleCamerasImpl {
public:
	MultipleCamerasImpl(Trigger *pTrigger)
			: m_Trigger(*pTrigger) {
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

	void Initialize(uint32_t nExpoMicroSec, float fGain) {
		m_nExpoMicroSec = nExpoMicroSec;
		auto pCamList = FlirInstance::GetCameraList();
		CHECK_GT(pCamList->GetSize(), 0);
		for (uint32_t i = 0; i < pCamList->GetSize(); ++i) {
			auto pCam = pCamList->GetByIndex(i);
			CHECK(!pCam->IsInitialized()) << "Double initialized!";
			pCam->Init();
			auto &nodeMap = pCam->GetNodeMap();
			SetNodeParam(nodeMap, "TriggerMode", "Off");

			if (false) {
				SetNodeParam(nodeMap, "AcquisitionFrameRateEnable", true);
				SetNodeParam(nodeMap, "AcquisitionFrameRate", 30.f);
			} else {
				SetNodeParam(nodeMap, "PixelFormat", "YCbCr8_CbYCr");
				SetNodeParam(nodeMap, "VideoMode", "Mode0");
				SetNodeParam(nodeMap, "AcquisitionFrameRateEnabled", true);
				SetNodeParam(nodeMap, "AcquisitionFrameRateAuto", "Off");
				SetNodeParam(nodeMap, "AcquisitionFrameRate", 20.f);
				SetNodeParam(nodeMap, "pgrExposureCompensationAuto", "Off");
				SetNodeParam(nodeMap, "pgrExposureCompensation", 0.f);
			}
			SetNodeParam(nodeMap, "BalanceWhiteAuto", "Off");
			SetNodeParam(nodeMap, "BalanceRatioSelector", "Red");
			SetNodeParam(nodeMap, "BalanceRatio", 1.5f);
			SetNodeParam(nodeMap, "BalanceRatioSelector", "Blue");
			SetNodeParam(nodeMap, "BalanceRatio", 1.5f);
			SetNodeParam(nodeMap, "BalanceWhiteAuto", "Continuous");

			SetNodeParam(nodeMap, "SaturationEnabled", true);
			SetNodeParam(nodeMap, "Saturation", 100.f);

			SetNodeParam(nodeMap, "ExposureAuto", "Off");
			SetNodeParam(nodeMap, "ExposureTime", (float)m_nExpoMicroSec);
			SetNodeParam(nodeMap, "ExposureTimeAbs", (float)m_nExpoMicroSec);
			if (fGain >= 0.f) {
				SetNodeParam(nodeMap, "GainAuto", "Off");
				SetNodeParam(nodeMap, "Gain", 20.f); // [0.0, 24.0] gain 12.2 equal to gamma 3.0 but more smooth
			} else {
				SetNodeParam(nodeMap, "GainAuto", "Continuous");
				SetNodeParam(nodeMap, "AutoGainLowerLimit", 0.f);
				SetNodeParam(nodeMap, "AutoGainUpperLimit", 24.f);
			}
			SetNodeParam(nodeMap, "GammaEnabled", true);
			SetNodeParam(nodeMap, "Gamma", 1.0f); // [1.0, 3.0] gamma 3.0 equal to gain 12.2 but more smooth

			for (auto conf : m_Trigger.GetCamConfigs()) {
				SetNodeParam(nodeMap, conf.first, conf.second.c_str());
			}

			SetNodeParam(pCam->GetTLStreamNodeMap(), "StreamBufferCountMode", "Manual");
			SetNodeParam(pCam->GetTLStreamNodeMap(), "StreamBufferCountManual", 1);

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

		auto nRealExpoMicroSec = (int32_t)(m_ExpTimer.Now() * 1000. * 1000.);
		usleep(std::max(0, (int32_t)m_nExpoMicroSec - nRealExpoMicroSec));
		m_Trigger();
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
	Trigger &m_Trigger;
	std::vector<cv::Mat> m_ImgBuf;
	std::thread m_WorkerThread;

	uint32_t m_nExpoMicroSec { 10000 };
	CTimer m_ExpTimer;
};

MultipleCameras::MultipleCameras(Trigger *pTrigger) {
	m_pImpl = new MultipleCamerasImpl(pTrigger);
}

MultipleCameras::~MultipleCameras() {
	delete m_pImpl;
}

void MultipleCameras::Initialize(uint32_t nExpoMicroSec, float fGain) {
	m_pImpl->Initialize(nExpoMicroSec, fGain);
}

void MultipleCameras::GetImages(std::vector<cv::Mat> &images) {
	m_pImpl->GetImages(images);
}
