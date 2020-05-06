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

	std::string rootFolder = "./denny";
	
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
		//harris.showFeaturePoints(3);
		harris.setFeatureDescription();
		printf("Find Description end!\n");
		harrises.push_back(harris);*/

		//64-dims feature vector
		printf("Image %d:\n", i);
		MSOP msop(image, 1);
		msop.findHarrisResponse();
		printf("Find HarrisResponse!\n");
		msop.getFeaturePoints(300);
		printf("Find Feature Points!\n");
		//msop.showFeaturePoints(3);
		msop.setFeatureDescription();
		printf("Find Description end!\n");
		msops.push_back(msop);
	}
	
	/*for (int i = 0; i < (int)harrises.size() - 1; i++) {
		std::vector<cv::Vec2i> match = imgUtils.getMatchFeaturePoints(harrises[i], harrises[i + 1], 0.5f);
		printf("Image %d and Image %d match counts: %d\n", i, i + 1, match.size());
		printf("----------------------------------------------------------------------\n");
		imgUtils.showMatchResult(harrises[i], harrises[i + 1], match);
	}*/

	
	for (int i = 0; i < (int)msops.size() - 1; i++) {
		std::vector<std::vector<cv::Vec2i>> matches = imgUtils.getMatchFeaturePoints(msops[i], msops[i + 1], 0.5f);
		for (int m = 0; m < matches.size(); m++) {
			printf("Pyramid %d\n", m);
			printf("Image %d and Image %d match counts: %d\n", i, i + 1, matches[m].size());
		}
		printf("----------------------------------------------------------------------\n");
		imgUtils.showMatchResult(msops[i], msops[i + 1], matches);
	}
	

	return 0;
}