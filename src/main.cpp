#include <iostream>
#include <string>
#include <vector>

#include "ImageUtils.h"
#include "HarrisDetector.h"

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

	std::string filename = "test.jpg";
	cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
	
	HarrisDetector detector(image);
	detector.findHarrisResponse();
	
	std::vector<pointData> pts = detector.getFeaturePoints(0.5f, 50);
	detector.showFeaturePoints(pts);


	return 0;
}