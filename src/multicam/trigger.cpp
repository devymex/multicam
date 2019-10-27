#include "multicam/trigger.hpp"

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

	void SetTriggerDelay(uint32_t nMilliseconds) {
		m_nTriggerDelayMS = nMilliseconds;
	}

	uint32_t GetTriggerDelay() const {
		return m_nTriggerDelayMS;
	}

	std::vector<std::pair<std::string, std::string>> GetCamConfigs() const {
		return m_CamConfs;
	}

	virtual void operator()() = 0;

protected:
	std::vector<std::pair<std::string, std::string>> m_CamConfs;

private:
	uint32_t m_nTriggerDelayMS { 0 };
};

class SoftwareTrigger : public Trigger::TriggerImpl {
public:
	SoftwareTrigger() {
		m_CamConfs.emplace_back("AcquisitionMode", "Continuous");
		m_CamConfs.emplace_back("TriggerMode", "Off");
		m_CamConfs.emplace_back("TriggerSource", "Software");
		//m_CamConfs.emplace_back("TriggerActivation", "RisingEdge");
		m_CamConfs.emplace_back("TriggerMode", "On");
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
				CHECK(flir::GenApi::IsAvailable(pTriggerCmd)
					&& flir::GenApi::IsWritable(pTriggerCmd));
				m_Commanders.push_back(pTriggerCmd);
			}
		}
		for (auto ptr : m_Commanders) {
			ptr->Execute();
		}
	}
	std::vector<flir::GenApi::CCommandPtr> m_Commanders;
};

class HardwareTrigger : public Trigger::TriggerImpl {
public:
	HardwareTrigger(const std::string &strDevPort) {
		m_nDevHdl = open(strDevPort.c_str(), O_RDWR | O_NDELAY);
		CHECK_GE(m_nDevHdl, 0);
		CHECK_GE(fcntl(m_nDevHdl, F_SETFL, 0), 0);
		termios options = {0};
		options.c_cflag = B9600 | CS8 | CREAD; // | HUPCL | CLOCAL;
		CHECK_GE(tcsetattr(m_nDevHdl, TCSANOW, &options), 0);

		m_CamConfs.emplace_back("AcquisitionMode", "Continuous");
		m_CamConfs.emplace_back("TriggerMode", "Off");
		m_CamConfs.emplace_back("TriggerSource", "Line0");
		m_CamConfs.emplace_back("TriggerMode", "On");
	}

	~HardwareTrigger() {
		close(m_nDevHdl);
	}

	void operator()() override {
		auto nDelay = GetTriggerDelay();
		CHECK_LE(nDelay, 10000);
		int16_t nData = _GetTriggerData(nDelay);
		size_t nSentBytes = write(m_nDevHdl, &nData, sizeof(nData));
		CHECK_EQ(nSentBytes, sizeof(nData));
	}

protected:
	virtual int16_t _GetTriggerData(uint32_t nDelay) = 0;

private:
	int m_nDevHdl;
};

class HardwareTriggerLL : public HardwareTrigger {
public:
	HardwareTriggerLL(const std::string &strDevName)
			: HardwareTrigger(strDevName) {
	}
protected:
	int16_t _GetTriggerData(uint32_t nDelay) override {
		return -(uint16_t)nDelay;
	}
};

class HardwareTriggerHL : public HardwareTrigger {
public:
	HardwareTriggerHL(const std::string &strDevName)
			: HardwareTrigger(strDevName) {
	}
protected:
	int16_t _GetTriggerData(uint32_t nDelay) override {
		return nDelay;
	}
};

Trigger::Trigger(std::string strDevice) {
	std::string strType, strName;
	int nColonPos = strDevice.find(':');
	if (nColonPos != std::string::npos) {
		strType = strDevice.substr(0, nColonPos);
		strName = strDevice.substr(nColonPos + 1, strDevice.length());
	} else {
		strType = strDevice;
	}
	if (strType == "software") {
		m_pImpl = new SoftwareTrigger();
	} else if (strType == "hardhl") {
		m_pImpl = new HardwareTriggerHL(strName);
	} else if (strType == "hardll") {
		m_pImpl = new HardwareTriggerLL(strName);
	} else {
		LOG(FATAL) << "Unsupported trigger type \"" << strType << "\"";
	}
}

Trigger::~Trigger() {
	delete m_pImpl;
}

void Trigger::operator()() {
	(*m_pImpl)();
}

void Trigger::SetTriggerDelay(uint32_t nMilliseconds) {
	m_pImpl->SetTriggerDelay(nMilliseconds);
}

uint32_t Trigger::GetTriggerDelay() const {
	return m_pImpl->GetTriggerDelay();
}

std::vector<std::pair<std::string, std::string>> Trigger::GetCamConfigs() const {
	return m_pImpl->GetCamConfigs();
}
