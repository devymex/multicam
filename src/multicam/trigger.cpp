#include "trigger.hpp"

#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <string>

#include <glog/logging.h>

#include "flir_inst.hpp"

class Trigger::TriggerImpl {
public:

	virtual ~TriggerImpl() {
	}

	inline void SetDelay(uint32_t nMicroseconds) {
		m_nDelayMicroSec = nMicroseconds;
	}

	inline uint32_t GetDelay() const {
		return m_nDelayMicroSec;
	}

	virtual void operator()() = 0;

private:
	uint32_t m_nDelayMicroSec { 0 };
};

class SoftwareTrigger : public Trigger::TriggerImpl {
public:
	SoftwareTrigger() {
	}

	void operator()() override {
		if (m_Commanders.empty()) {
			auto pCamList = FlirInstance::GetCameraList();
			for (uint32_t i = 0; i < pCamList->GetSize(); ++i) {
				auto pCam = pCamList->GetByIndex(i);
				CHECK(pCam->IsInitialized());
				auto &nodeMap = pCam->GetNodeMap();
				flir::GenApi::CCommandPtr pTriggerCmd =
						nodeMap.GetNode("TriggerSoftware");
				CHECK(flir::GenApi::IsAvailable(pTriggerCmd));
				CHECK(flir::GenApi::IsWritable(pTriggerCmd));
				m_Commanders.push_back(pTriggerCmd);
			}
		}
		if (GetDelay() > 0) {
			usleep(GetDelay());
		}
		for (auto ptr : m_Commanders) {
			ptr->Execute();
		}
	}

private:
	std::vector<flir::GenApi::CCommandPtr> m_Commanders;
};

class HardwareTrigger : public Trigger::TriggerImpl {
public:
	HardwareTrigger(const std::string &strDevPort) {
		m_nDevHdl = open(strDevPort.c_str(), O_RDWR | O_NDELAY);
		CHECK_GE(m_nDevHdl, 0) << "Can't open device \"" << strDevPort << "\"";
		CHECK_GE(fcntl(m_nDevHdl, F_SETFL, 0), 0);
		termios options = {0};
		options.c_cflag = B9600 | CS8 | CREAD; // | HUPCL | CLOCAL;
		CHECK_GE(tcsetattr(m_nDevHdl, TCSANOW, &options), 0);
	}

	~HardwareTrigger() {
		close(m_nDevHdl);
	}

	void operator()() override {
		if (GetDelay() > 0) {
			usleep(GetDelay());
		}
		int16_t nData = 1;
		size_t nSentBytes = write(m_nDevHdl, &nData, sizeof(nData));
		CHECK_EQ(nSentBytes, sizeof(nData));
	}

private:
	int m_nDevHdl;
};

Trigger::Trigger(std::string strDevice)
		: m_strDevice(strDevice) {
	if (strDevice == "Software") {
		m_pImpl = new SoftwareTrigger();
	} else {
		m_pImpl = new HardwareTrigger(strDevice);
	}
}

Trigger::~Trigger() {
	delete m_pImpl;
}

void Trigger::operator()() {
	(*m_pImpl)();
}

void Trigger::SetDelay(uint32_t nMilliseconds) {
	m_pImpl->SetDelay(nMilliseconds);
}

uint32_t Trigger::GetDelay() const {
	return m_pImpl->GetDelay();
}

std::string Trigger::GetDevice() const {
	return m_strDevice;
}