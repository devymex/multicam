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

	void SetDelay(uint32_t nMilliseconds);

	uint32_t GetDelay() const;

	std::string GetDevice() const;

protected:
	void operator()();

protected:
	std::string m_strDevice;

	TriggerImpl *m_pImpl;

	friend class MultipleCameras;
};

#endif