
#include <glog/logging.h>

#include "multicam/multicam.hpp"
#include "multicam/trigger.hpp"
#include "multicam/ctimer.hpp"

int main(int nArgCnt, char *ppArgs[]) {
	Trigger trigger("software");
	//Trigger trigger("hardhl:/dev/ttyUSB0");
	trigger.SetTriggerDelay(1);
	MultipleCameras multiCam(&trigger);
	multiCam.Initialize(20000);
	CTimer t;
	for (int iFrame = 1; ; ++iFrame) {
		std::vector<cv::Mat> images;
		multiCam.GetImages(images);
		if (!images.empty() && !images[0].empty()) {
			cv::resize(images[0], images[0], images[0].size() / 2);
			cv::imshow("img", images[0]);
		}
		if (iFrame % 10 == 0) {
			LOG(INFO) << "fps: " << 10. / t.Now();
			t.Reset();
		}
		if (27 == cv::waitKey(1)) {
			break;
		}
	}
	return 0;
}

