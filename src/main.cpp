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
	if (argc < 2) {
		printf("argument error !\n");
		return 0;
	}
	std::string rootFolder = argv[1];

	int sampleCount = 200;
	if (argc > 2) {
		sampleCount = std::stoi(argv[2]);
	}
	else {
		printf("Use default sample count : 200\n");
	}

	float threshold = 0.5f;
	if (argc > 3) {
		threshold = std::stof(argv[3]);
	}
	else {
		printf("Use default threshold : 0.5\n");
	}

	int pyramidDepth = 1;
	if (argc > 4) {
		pyramidDepth = std::stoi(argv[4]);
	}
	else {
		printf("Use default pyramid depth : 1\n");
	}

	bool showFeatureResult = false;
	if (argc > 5) {
		showFeatureResult = std::stoi(argv[5]);
	}
	else {
		printf("Not show feature result\n");
	}

	ImageUtils imgUtils;
	imgUtils.parseImageInfo(rootFolder);
	
	std::vector<cv::Mat> images = imgUtils.getImages();
	
	std::vector<HarrisDetector> harrises;
	std::vector<MSOP> msops;

	for (int i = 0; i < images.size(); i++) {
		cv::Mat image = images[i].clone();

		// 9-dims feature vector
		/*printf("Image %d:\n", i);
		HarrisDetector harris(image);
		harris.findHarrisResponse();
		printf("Find HarrisResponse!\n");
		harris.getFeaturePoints(0.5f, 50);
		printf("Find Feature Points!\n");
		if(showFeatureResult)
			harris.showFeaturePoints(3);
		harris.setFeatureDescription();
		printf("Find Description end!\n");
		harrises.push_back(harris);*/

		//64-dims feature vector
		printf("Image %d:\n", i);
		MSOP msop(image, pyramidDepth);
		msop.findHarrisResponse();
		printf("Find HarrisResponse!\n");
		msop.getFeaturePoints(sampleCount);
		printf("Find Feature Points!\n");
		if (showFeatureResult)
			msop.showFeaturePoints(3);
		msop.setFeatureDescription();
		printf("Find Description end!\n");
		msops.push_back(msop);
	}
	
	/*for (int i = 0; i < (int)harrises.size() - 1; i++) {
		std::vector<cv::Vec2i> match = imgUtils.getMatchFeaturePoints(harrises[i], harrises[i + 1], threshold);
		printf("Image %d and Image %d match counts: %d\n", i, i + 1, match.size());
		printf("----------------------------------------------------------------------\n");
		if(showFeatureResult)
			imgUtils.showMatchResult(harrises[i], harrises[i + 1], match);
	}*/

	
	for (int i = 0; i < (int)msops.size() - 1; i++) {
		std::vector<std::vector<cv::Vec2i>> matches = imgUtils.getMatchFeaturePoints(msops[i], msops[i + 1], threshold);
		for (int m = 0; m < matches.size(); m++) {
			printf("Pyramid %d\n", m);
			printf("Image %d and Image %d match counts: %d\n", i, i + 1, matches[m].size());
			if (matches[m].size() == 0) {
				std::cout << "Add more sample feature points or use large threshold!" << std::endl;
				return 0;
			}
		}
		printf("----------------------------------------------------------------------\n");
		if(showFeatureResult)
			imgUtils.showMatchResult(msops[i], msops[i + 1], matches);
	}
	

	return 0;
}