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

	void SetDelay(uint32_t nMicroseconds);

	uint32_t GetDelay() const;

	std::string GetDevice() const;

protected:
	void operator()(int32_t nMicroseconds = -1);

protected:
	std::string m_strDevice;

	TriggerImpl *m_pImpl;

	friend class MultipleCameras;
};

#endif