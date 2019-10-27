
#include <glog/logging.h>

#include "multicam/multicam.hpp"
#include "multicam/ctimer.hpp"

int main(int nArgCnt, char *ppArgs[]) {
	MultipleCameras multiCam("Software");
	multiCam.Initialize(18000, "config");
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

