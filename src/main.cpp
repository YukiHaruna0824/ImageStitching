#include <iostream>
#include <string>
#include <vector>

#include "ImageUtils.h"
#include "HarrisDetector.h"
#include "MSOP.h"

#include<opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char* argv[])
{
	/*if (argc < 2) {
		std::cout << "argument error !" << std::endl;
		return 0;
	}

	std::string filename = argv[1];*/

	ImageUtils imgUtils;
	std::string rootFolder = "./parrington";
	imgUtils.parseImageInfo(rootFolder);
	
	std::vector<cv::Mat> images = imgUtils.getImages();
	std::vector<MSOP> msops;

	for (int i = 0; i < images.size(); i++) {
		cv::Mat image = images[i].clone();

		/*HarrisDetector detector(image);
		detector.findHarrisResponse();
		std::vector<cv::Point> pts = detector.getFeaturePoints(0.5f, 50);
		detector.showFeaturePoints(pts, 3);*/

		MSOP msop(image, 1);
		msop.findHarrisResponse();
		std::cout << "Find HarrisResponse!" << std::endl;
		msop.getFeaturePoints(100);
		std::cout << "Find Feature Points" << std::endl;
		//msop.showFeaturePoints(3);
		msop.setFeatureDescription();
		std::cout << "Find Description end!" << std::endl;
		msops.push_back(msop);
	}
	
	for (int i = 0; i < msops.size() - 1; i++) {
		imgUtils.matchFeaturePoints(msops[i], msops[i + 1], 0.4f);
	}


	return 0;
}