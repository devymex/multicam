#include "post_proc.hpp"

#include <glog/logging.h>

#define WITH_CUDA

#ifdef WITH_CUDA
#include <cuda.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_color_conversion.h>
#endif

cv::Mat UYV2Mat(flir::ImagePtr pImg) {
#ifdef WITH_CUDA
	NppiSize imgSize = {(int)pImg->GetWidth(), (int)pImg->GetHeight()};

	uint32_t nSrcStride = pImg->GetStride();
	uint32_t nSrcBytes = imgSize.height * nSrcStride;

	uint32_t nDstStride = imgSize.width * pImg->GetNumChannels();
	uint32_t nDstBytes = imgSize.height * nDstStride;

	uint8_t *pGpuBuf = nullptr;
	cudaMalloc(&pGpuBuf, nSrcBytes + nDstBytes);
	cudaMemcpy(pGpuBuf, pImg->GetData(), nSrcBytes, cudaMemcpyHostToDevice);

	int order[] = {1, 0, 2};
	nppiSwapChannels_8u_C3IR(pGpuBuf, nSrcStride, imgSize, order);
	nppiYUVToBGR_8u_C3R(pGpuBuf, pImg->GetStride(), pGpuBuf + nSrcBytes,
			nDstStride, imgSize);

	cv::Mat img(pImg->GetHeight(), pImg->GetWidth(), CV_8UC3);
	cudaMemcpy(img.data, pGpuBuf + nSrcBytes, nDstBytes, cudaMemcpyDeviceToHost);

	cudaFree(pGpuBuf);
	return img;
#else
	auto pBgrImg = pImg->Convert(flir::PixelFormat_BGR8, flir::HQ_LINEAR);
	cv::Mat img(pBgrImg->GetHeight(), pBgrImg->GetWidth(),
			CV_8UC3, pBgrImg->GetData());
	return img.clone();
#endif
}

cv::Mat PostProcess(flir::ImagePtr pRaw) {
	if (pRaw->GetPixelFormat() == flir::PixelFormat_YUV8_UYV) {
		return UYV2Mat(pRaw);
	}
	return cv::Mat();
}