#pragma once

#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "Utils.h"

class HarrisDetector
{
public:
	HarrisDetector(cv::Mat image);
	HarrisDetector(cv::Mat image, float k, int filterSize);
	void findHarrisResponse();

	void getFeaturePoints(float percentage, int localMaximumSize);
	void showFeaturePoints(int radius);

	void setFeatureDescription();

	//getter
	std::vector<FeaturePoint>& getFeatureDescription();
	cv::Mat& getImage();

private:
	Derivative computeDerivative();
	void computeResponse(Derivative &derivative);
	void applyGaussToDerivative(Derivative &derivative);
	
	int filterSize;
	float k;
	
	std::vector<cv::Point> featurePoints;
	std::vector<FeaturePoint> featureDescriptions;

	cv::Mat image;
	cv::Mat grayImage;
	cv::Mat harrisResponse;
};