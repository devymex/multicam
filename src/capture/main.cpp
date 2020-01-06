
#include <numeric>
#include <vector>
#include <glog/logging.h>

#include "multicam/multicam.hpp"
#include "multicam/ctimer.hpp"

int main(int nArgCnt, char *ppArgs[]) {
	std::string strTriggerDevice;
	if (nArgCnt > 1) {
		strTriggerDevice = ppArgs[1];
	}
	MultipleCameras multiCam(strTriggerDevice);

	std::vector<int> gpuIds;
	for (uint32_t i = 0; i < multiCam.GetCameraCount(); ++i) {
		gpuIds.push_back(i % cv::cuda::getCudaEnabledDeviceCount());
	}
	multiCam.Initialize(16000, "./config", gpuIds);

	CTimer t;
	std::vector<double> cycleTimer;

	std::vector<cv::cuda::GpuMat> images;
	cv::cuda::GpuMat tmp1, tmp2, tmp3, tmp4, resizedImg;
	cv::Mat showImg;

	for (int iFrame = 1; ; ++iFrame) {
		double dProcTime = 0;
		multiCam.GetImages(images);
		for (int iCam = 0; iCam < (int)images.size(); ++iCam) {
			auto &img = images[iCam];
			if (!img.empty()) {
				cv::cuda::setDevice(gpuIds[iCam]);
				CTimer t1;
				cv::cuda::cvtColor(img, tmp1, cv::COLOR_BGR2BGRA);
				cv::cuda::transpose(tmp1, tmp2);
				cv::cuda::cvtColor(tmp2, tmp3, cv::COLOR_BGRA2BGR);
				cv::cuda::flip(tmp3, tmp4, 1);
				cv::cuda::resize(tmp4, resizedImg, tmp4.size() / 2);
				resizedImg.download(showImg);
				dProcTime += t1.Reset();
				cv::imshow("cam" + std::to_string(iCam), showImg);
			}
		}
		cycleTimer.push_back(t.Reset());
		while (cycleTimer.size() > 10) {
			cycleTimer.erase(cycleTimer.begin());
		}
		LOG(INFO) << "FPS: " << (double)cycleTimer.size() /
				std::accumulate(cycleTimer.begin(), cycleTimer.end(), 0.)
				<< ", proc time: " << dProcTime;
		int nKey = cv::waitKey(1);
		if ('s' == nKey) {
			for (int iCam = 0; iCam < (int)images.size(); ++iCam) {
				auto &img = images[iCam];
				if (!img.empty()) {
					auto camInfo = multiCam.GetCameraInfo(iCam);
					std::string strName = "frm" + std::to_string(iFrame)
							+ "_cam" + camInfo.strDeviceSN + ".png";
					cv::Mat saveImg(img);
					cv::imwrite(strName, saveImg);
				}
			}
		} else if (27 == nKey) {
			break;
		}
	}
	return 0;
}
