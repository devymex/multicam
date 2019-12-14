#include "post_proc.hpp"

#include <functional>
#include <vector>
#include <glog/logging.h>

#include <cuda.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_color_conversion.h>
#include <nppi_geometry_transforms.h>

#define CUDA_VERIFY(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define NPP_VERIFY(ans) { nppAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Failed: %s %s %d\n",
				cudaGetErrorString(code), file, line);
		exit(code);
	}
}

inline void nppAssert(NppStatus code, const char *file, int line) {
	if (code != 0) {
		fprintf(stderr, "NPP Failed: %d %s %d\n", (int)code, file, line);
		exit((int)code);
	}
}

typedef std::function<uint8_t*(size_t)> ALLOCATOR;

void ReallocGpuMat8U(NppiSize srcSize, int nChannels, cv::cuda::GpuMat &dstImg) {
	cv::Size imgSize(srcSize.width, srcSize.height);
	int nSrcType = CV_MAKETYPE(CV_8U, nChannels);
	if (dstImg.size() != imgSize || dstImg.type() != nSrcType) {
		dstImg = cv::cuda::GpuMat(imgSize, nSrcType);
	}
}

void FlirUYV2GpuMat(flir::ImagePtr pImg, cv::cuda::GpuMat &dstImg,
		ALLOCATOR Allocator) {
	CHECK_EQ(pImg->GetNumChannels(), 3);

	// Copy data from host to device
	uint32_t nSrcStride = pImg->GetStride();
	uint32_t nBufStride = pImg->GetWidth() * pImg->GetNumChannels();
	uint8_t *pBuffer = Allocator(nBufStride * pImg->GetHeight());
	CUDA_VERIFY(cudaMemcpy2D(pBuffer, nBufStride, pImg->GetData(), nSrcStride,
			pImg->GetWidth() * pImg->GetNumChannels(), pImg->GetHeight(),
			cudaMemcpyHostToDevice));

	// Swap channels from UYV TO YUV
	NppiSize srcImgSize = {(int)pImg->GetWidth(), (int)pImg->GetHeight()};
	const int channelOrder[] = {1, 0, 2};
	NPP_VERIFY(nppiSwapChannels_8u_C3IR(pBuffer, nBufStride,
			srcImgSize, channelOrder));

	// Realloc dstImg buffer and convert YUV TO RGB
	ReallocGpuMat8U(srcImgSize, 3, dstImg);
	NPP_VERIFY(nppiYUVToBGR_8u_C3R(pBuffer, nSrcStride, dstImg.data,
			dstImg.step, srcImgSize));
}

void FlirBGR2GpuMat(flir::ImagePtr pImg, cv::cuda::GpuMat &dstImg,
		ALLOCATOR Allocator) {
	CHECK_EQ(pImg->GetNumChannels(), 3);

	//Realloc dstImg buffer
	NppiSize srcImgSize = {(int)pImg->GetWidth(), (int)pImg->GetHeight()};
	ReallocGpuMat8U(srcImgSize, 3, dstImg);

	// Copy memory with stride
	CUDA_VERIFY(cudaMemcpy2D(dstImg.data, dstImg.step, // dst
			pImg->GetData(), pImg->GetStride(), // src
			pImg->GetWidth() * pImg->GetNumChannels(), pImg->GetHeight(), // size
			cudaMemcpyHostToDevice)); // direction
}

void FlirBayer2GpuMat(flir::ImagePtr pImg, cv::cuda::GpuMat &dstImg,
		ALLOCATOR Allocator) {
	CHECK_EQ(pImg->GetNumChannels(), 1);

	// Copy data from host to device
	uint32_t nSrcStride = pImg->GetStride();
	uint32_t nBufStride = pImg->GetWidth() * pImg->GetNumChannels();
	uint8_t *pBuffer = Allocator(nBufStride * pImg->GetHeight());
	CUDA_VERIFY(cudaMemcpy2D(pBuffer, nBufStride, pImg->GetData(), nSrcStride,
			pImg->GetWidth() * pImg->GetNumChannels(), pImg->GetHeight(),
			cudaMemcpyHostToDevice));

	// Debayer
	NppiSize srcImgSize = {(int)pImg->GetWidth(), (int)pImg->GetHeight()};
	NppiRect srcROI = {0, 0, srcImgSize.width, srcImgSize.height};
	ReallocGpuMat8U(srcImgSize, 3, dstImg);
	NPP_VERIFY(nppiCFAToRGB_8u_C1C3R(pBuffer, nBufStride,
			srcImgSize, srcROI, dstImg.data, dstImg.step, NPPI_BAYER_RGGB,
			NPPI_INTER_UNDEFINED));

	// RGB to BGR
	const int channelOrder[] = {2, 1, 0};
	NPP_VERIFY(nppiSwapChannels_8u_C3IR(dstImg.data, dstImg.step,
			srcImgSize, channelOrder));
}

PostProcessor::PostProcessor() : m_pBuffer(nullptr) {
}

PostProcessor::~PostProcessor() {
	if (m_pBuffer != nullptr) {
		CUDA_VERIFY(cudaFree(m_pBuffer));
	}
}

void PostProcessor::Process(flir::ImagePtr pRaw, cv::cuda::GpuMat &dstImg) {
	auto ReqBuf = std::bind(&PostProcessor::__RequestBuffer,
			this, std::placeholders::_1);
	if (pRaw->GetPixelFormat() == flir::PixelFormat_YUV8_UYV) {
		FlirUYV2GpuMat(pRaw, dstImg, ReqBuf);
	} else if (pRaw->GetPixelFormat() == flir::PixelFormat_BGR8){
		FlirBGR2GpuMat(pRaw, dstImg, ReqBuf);
	} else if (pRaw->GetPixelFormat() == flir::PixelFormat_BayerRG8) {
		return FlirBayer2GpuMat(pRaw, dstImg, ReqBuf);
	} else {
		LOG(INFO) << pRaw->GetPixelFormatName();
		LOG(INFO) << pRaw->GetWidth() << "x" << pRaw->GetHeight()
				  << "x" << pRaw->GetNumChannels();
		dstImg = cv::cuda::GpuMat();
	}
}

uint8_t* PostProcessor::__RequestBuffer(uint32_t nBytes) {
	if (m_nBufSize < nBytes) {
		CUDA_VERIFY(cudaFree(m_pBuffer));
		CUDA_VERIFY(cudaMalloc(&m_pBuffer, nBytes));
		m_nBufSize = nBytes;
	} else if (nBytes == 0) {
		return nullptr;
	}
	return (uint8_t*)m_pBuffer;
}