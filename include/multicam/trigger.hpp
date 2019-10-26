#ifndef __TRIGGER_HPP
#define __TRIGGER_HPP

#include <cstdint>
#include <string>
#include <vector>

class Trigger {
public:
	class TriggerImpl;

public:
	Trigger(std::string strDevice);

	~Trigger();

	void SetTriggerDelay(uint32_t nMilliseconds);

	uint32_t GetTriggerDelay() const;

protected:
	void operator()();

	std::vector<std::pair<std::string, std::string>> GetCamConfigs() const;

protected:
	TriggerImpl *m_pImpl;

	friend class MultipleCameras;
};

#endif