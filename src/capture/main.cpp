
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
	multiCam.Initialize(16000, "./config");
	CTimer t;
	std::vector<double> cycleTimer;
	for (int iFrame = 1; ; ++iFrame) {
		std::vector<cv::Mat> images;
		multiCam.GetImages(images);
		for (int iCam = 0; iCam < (int)images.size(); ++iCam) {
			auto &img = images[iCam];
			if (!img.empty()) {
				cv::resize(img, img, img.size() / 4);
				std::string strName = "cam" + std::to_string(iCam);
				cv::imshow(strName, img);
			}
		}
		cycleTimer.push_back(t.Reset());
		while (cycleTimer.size() > 10) {
			cycleTimer.erase(cycleTimer.begin());
		}
		//LOG(INFO) << "FPS: " << (double)cycleTimer.size() /
		//		std::accumulate(cycleTimer.begin(), cycleTimer.end(), 0.);
		int nKey = cv::waitKey(1);
		if ('s' == nKey) {
			for (int iCam = 0; iCam < (int)images.size(); ++iCam) {
				auto &img = images[iCam];
				if (!img.empty()) {
					std::string strName = "frm" + std::to_string(iFrame)
							+ "_cam" + std::to_string(iCam) + ".png";
					cv::imwrite(strName, img);
				}
			}
		} else if (27 == nKey) {
			break;
		}
	}
	return 0;
}
