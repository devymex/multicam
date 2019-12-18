#ifndef __POST_PROC_HPP
#define __POST_PROC_HPP

#include <opencv2/opencv.hpp>
#include "flir_inst.hpp"

class PostProcessor {
public:
	PostProcessor(int nDevID = -1);
	~PostProcessor();

	void Process(flir::ImagePtr pRaw, cv::cuda::GpuMat &dstImg);

private:
	uint8_t* __RequestBuffer(uint32_t nBytes);

private:
	uint32_t m_nBufSize;
	uint8_t *m_pBuffer;
	int m_nDevID;
};

#endif //__POST_PROC_HPP
