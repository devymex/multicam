#ifndef __POST_PROC_HPP
#define __POST_PROC_HPP

#include <opencv2/opencv.hpp>
#include "flir_inst.hpp"

class PostProcessor {
public:
	PostProcessor();

	~PostProcessor();

	cv::Mat operator()(flir::ImagePtr pRaw);

private:
	cv::Mat __UYV2BGR(flir::ImagePtr pImg);
	cv::Mat __DeBayer(flir::ImagePtr pImg);
	uint8_t* __RequestBuffer(uint32_t nBytes);

private:
	void *m_pBuffer { nullptr };
	uint32_t m_nBufSize { 0 };
	cv::Size m_DstSize;
};

#endif //__POST_PROC_HPP
