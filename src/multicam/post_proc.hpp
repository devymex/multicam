#ifndef __POST_PROC_HPP
#define __POST_PROC_HPP

#include <opencv2/opencv.hpp>
#include "flir_inst.hpp"

class PostProcessor {
public:
	~PostProcessor();

	cv::Mat operator()(flir::ImagePtr pRaw);

private:
	cv::Mat __UYV2Mat(flir::ImagePtr pImg);
	uint8_t* __RequestBuffer(uint32_t nBytes);

private:
	void *m_pBuffer { nullptr };
	uint32_t m_nBufSize { 0 };
};

#endif //__POST_PROC_HPP
