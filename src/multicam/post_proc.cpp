#include "post_proc.hpp"

#include <glog/logging.h>

#define WITH_CUDA

#ifdef WITH_CUDA
#include <cuda.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_color_conversion.h>
#include <nppi_geometry_transforms.h>
#endif

PostProcessor::~PostProcessor() {
#ifdef WITH_CUDA
	cudaFree(m_pBuffer);
#endif
}

cv::Mat PostProcessor::operator()(flir::ImagePtr pRaw) {
#ifdef WITH_CUDA
	if (pRaw->GetPixelFormat() == flir::PixelFormat_YUV8_UYV) {
		return __UYV2BGR(pRaw);
	} else if (pRaw->GetPixelFormat() == flir::PixelFormat_BGR8){
		cv::Mat img(pRaw->GetHeight(), pRaw->GetWidth(), CV_8UC3,
				pRaw->GetData(), pRaw->GetStride());
		cv::resize(img, img, img.size() / 4);
		return img.clone();
	} else {
		LOG(INFO) << pRaw->GetPixelFormatName();
		LOG(INFO) << pRaw->GetWidth() << "x" << pRaw->GetHeight()
				  << "x" << pRaw->GetNumChannels();
	}
	return cv::Mat();
#else
	auto pBgrImg = pImg->Convert(flir::PixelFormat_BGR8, flir::HQ_LINEAR);
	cv::Mat img(pBgrImg->GetHeight(), pBgrImg->GetWidth(),
			CV_8UC3, pBgrImg->GetData());
	return img.clone();
#endif
}

cv::Mat PostProcessor::__UYV2BGR(flir::ImagePtr pImg) {
	NppiSize srcSize = {(int)pImg->GetWidth(), (int)pImg->GetHeight()};
	NppiSize dstSize = {srcSize.width / 2, srcSize.height / 2};
	NppiRect srcROI = {0, 0, srcSize.width, srcSize.height};
	NppiRect dstROI = {0, 0, dstSize.width, dstSize.height};
	uint32_t nChannels = pImg->GetNumChannels();

	uint32_t nSrcStride = pImg->GetStride();
	uint32_t nSrcBytes = srcSize.height * nSrcStride;

	uint32_t nDstStride = srcSize.width * nChannels;
	uint32_t nDstBytes = srcSize.height * nDstStride;

	uint8_t *pSrcBuf = __RequestBuffer(nSrcBytes + nDstBytes);;
	uint8_t *pDstBuf = pSrcBuf + nSrcBytes;

	cudaMemcpy(pSrcBuf, pImg->GetData(), nSrcBytes, cudaMemcpyHostToDevice);

	int order[] = {1, 0, 2};
	nppiSwapChannels_8u_C3IR(pSrcBuf, nSrcStride, srcSize, order);
	nppiYUVToBGR_8u_C3R(pSrcBuf, nSrcStride, pDstBuf, nDstStride, srcSize);

	std::swap(pDstBuf, pSrcBuf);
	nDstStride = dstSize.width * nChannels;
	nDstBytes = nDstStride * dstSize.height;

	nppiResize_8u_C3R(pSrcBuf, nSrcStride, srcSize, srcROI,
			pDstBuf, nDstStride, dstSize, dstROI, NPPI_INTER_LINEAR);

	cv::Mat img(dstSize.height, dstSize.width, CV_8UC3);
	cudaMemcpy(img.data, pDstBuf, nDstBytes, cudaMemcpyDeviceToHost);

	return img;
}

uint8_t* PostProcessor::__RequestBuffer(uint32_t nBytes) {
	if (m_nBufSize < nBytes) {
		cudaFree(m_pBuffer);
		cudaMalloc(&m_pBuffer, nBytes);
	}
	return (uint8_t*)m_pBuffer;
}