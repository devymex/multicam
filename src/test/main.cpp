
#include "multicam/multicam.hpp"
#include "multicam/trigger.hpp"
#include <glog/logging.h>

int main(int nArgCnt, char *ppArgs[]) {
	Trigger trigger("software");
	//Trigger trigger("hardhl:/dev/ttyUSB0");
	trigger.SetTriggerDelay(1);
	MultipleCameras multiCam(&trigger);
	multiCam.Initialize();
	do {
		std::vector<cv::Mat> images;
		multiCam.GetImages(images);
		if (!images.empty() && !images[0].empty()) {
			cv::resize(images[0], images[0], images[0].size() / 4);
			cv::imshow("img", images[0]);
		}
	} while (27 != cv::waitKey(1));
	return 0;
}

