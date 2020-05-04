#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/features2d/features2d.hpp>

#include "MSOP.h"

class ImageUtils
{
public:
	void parseImageInfo(std::string rootFolder);
	void matchFeaturePoints(MSOP &lhs, MSOP &rhs, float ratio);
	std::vector<cv::Mat>& getImages();
	std::vector<float>& getFocals();

private:
	float computeFeaturePointDistance(cv::Mat &lhs, cv::Mat &rhs);

	std::vector<cv::Mat> images;
	std::vector<float> focalLengths;
};