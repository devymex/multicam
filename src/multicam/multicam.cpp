
#include "multicam/multicam.hpp"

#include <condition_variable>
#include <mutex>
#include <thread>

#include <glog/logging.h>
#include <Spinnaker.h>
#include <SpinGenApi/SpinnakerGenApi.h>

#include "flir_inst.hpp"
#include "ctimer.hpp"

class Semaphore {
public:
	void notify() {
		std::lock_guard<decltype(mutex_)> lock(mutex_);
		m_nCnt = 1;
		m_CV.notify_one();
	}

	bool wait(uint32_t nMilliseconds) {
		bool br = true;
		std::unique_lock<decltype(mutex_)> lock(mutex_);
		if (!m_nCnt) {
			auto duration = std::chrono::milliseconds(nMilliseconds);
			if (std::cv_status::timeout == m_CV.wait_for(lock, duration)) {
				br = false;
			}
		}
		m_nCnt = 0;
		return br;
	}
private:
	std::mutex mutex_;
	std::condition_variable m_CV;
	unsigned long m_nCnt { 0 };
};

class MultipleCameras::MultipleCamerasImpl : public flir::DeviceEvent {
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
				//pCam->UnregisterEvent(*this);
				pCam->DeInit();
			}
		}
	}

	void Initialize() {
		auto pCamList = FlirInstance::GetCameraList();
		CHECK_GT(pCamList->GetSize(), 0);
		for (uint32_t i = 0; i < pCamList->GetSize(); ++i) {
			auto pCam = pCamList->GetByIndex(i);
			CHECK(!pCam->IsInitialized()) << "Double initialized!";
			pCam->Init();
			//SetNodeParam(pCam->GetNodeMap(), "PixelFormat", "YCbCr8_CbYCr");
			//SetNodeParam(pCam->GetNodeMap(), "VideoMode", "Mode0");
			SetNodeParam(pCam->GetNodeMap(), "BalanceWhiteAuto", "Continuous");
			SetNodeParam(pCam->GetNodeMap(), "TriggerMode", "Off");

			SetNodeParam(pCam->GetNodeMap(), "AcquisitionFrameRateEnable", true);

			//SetNodeParam(pCam->GetNodeMap(), "AcquisitionFrameRateAuto", "Off");
			//SetNodeParam(pCam->GetNodeMap(), "AcquisitionFrameRateEnabled", true);

			SetNodeParam(pCam->GetNodeMap(), "AcquisitionFrameRate", 30.f);
			//SetNodeParam(pCam->GetNodeMap(), "Width", 2080);
			//SetNodeParam(pCam->GetNodeMap(), "Height", 1552);
			SetNodeParam(pCam->GetTLStreamNodeMap(), "StreamBufferCountMode", "Manual");
			SetNodeParam(pCam->GetTLStreamNodeMap(), "StreamBufferCountManual", 1);
			for (auto conf : m_Trigger.GetCamConfigs()) {
				SetNodeParam(pCam->GetNodeMap(), conf.first, conf.second.c_str());
			}

			//SetParam("BalanceRatioSelector", "Blue");
			//flir::GenApi::CIntegerPtr width = pCam->GetNodeMap().GetNode("Width");
			//flir::GenApi::CIntegerPtr height = pCam->GetNodeMap().GetNode("Height");
			//SetParam("BalanceRatioSelector", "Red");
			//balanceRatio->SetValue(2);

			auto pEventSelctor = GetNode<flir::GenApi::CEnumerationPtr>(
					pCam->GetNodeMap(), "EventSelector");
			flir::GenApi::NodeList_t entries;
			pEventSelctor->GetEntries(entries);
			for (unsigned int i = 0; i < entries.size(); i++) {
				flir::GenApi::CEnumEntryPtr ptrEnumEntry = entries.at(i);
				pEventSelctor->SetIntValue(ptrEnumEntry->GetValue());
				SetNodeParam(pCam->GetNodeMap(), "EventNotification", "On");
				LOG(INFO) << ptrEnumEntry->GetDisplayName();
			}
			//pCam->RegisterEvent(*this);
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
	void OnDeviceEvent(flir::GenICam::gcstring eventName) override {
		LOG(INFO) << "event=" << GetDeviceEventName() << ", ID="
			<< GetDeviceEventId();
		if (GetDeviceEventId() == 40003) {
			m_TriggerSignal.notify();
		}
	}

	void __DoNextAsync() {
		m_WorkerThread = std::thread(&MultipleCamerasImpl::__WorkerProc, this);
	}

	void __WaitingForWorker() {
		if (m_WorkerThread.joinable()) {
			m_WorkerThread.join();
		}
	}

	void __WorkerProc() {
		usleep(70 * 1000);
		m_Trigger();
		LOG(INFO) << "Triggered";
		auto nTimeOutMS = 3000;
		auto pCamList = FlirInstance::GetCameraList();
		auto nCamCnt = pCamList->GetSize();
		m_ImgBuf.clear();

		std::vector<flir::ImagePtr> rawImgPtrs;
		rawImgPtrs.resize(nCamCnt);
		for (uint32_t i = 0; i < nCamCnt; ++i) {
			auto pCam = pCamList->GetByIndex(i);
			flir::ImagePtr pRawImg;
			try {
				rawImgPtrs[i] = pCam->GetNextImage(nTimeOutMS);
				LOG(INFO) << "Captured";
			} catch (...) {
				rawImgPtrs.clear();
			}
		}
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
	Semaphore m_TriggerSignal;
};

MultipleCameras::MultipleCameras(Trigger *pTrigger) {
	m_pImpl = new MultipleCamerasImpl(pTrigger);
}

MultipleCameras::~MultipleCameras() {
	delete m_pImpl;
}

void MultipleCameras::Initialize() {
	m_pImpl->Initialize();
}

void MultipleCameras::GetImages(std::vector<cv::Mat> &images) {
	m_pImpl->GetImages(images);
}
