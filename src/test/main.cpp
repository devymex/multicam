
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
	multiCam.Initialize(18000, "./config");
	CTimer t;
	std::vector<double> cycleTimer;
	for (int iFrame = 1; ; ++iFrame) {
		std::vector<cv::Mat> images;
		multiCam.GetImages(images);
		for (int i = 0; i < (int)images.size(); ++i) {
			auto &img = images[i];
			if (!img.empty()) {
				cv::resize(img, img, img.size() / 2);
				std::string strName = "img" + std::to_string(i);
				cv::imshow(strName, img);
			}
		}
		cycleTimer.push_back(t.Reset());
		if (cycleTimer.size() > 10) {
			cycleTimer.erase(cycleTimer.begin());
		}
		//LOG(INFO) << "FPS: " << (double)cycleTimer.size() /
		//		std::accumulate(cycleTimer.begin(), cycleTimer.end(), 0.);
		if (27 == cv::waitKey(1)) {
			break;
		}
	}
	return 0;
}
