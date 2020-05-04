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

	std::vector<cv::Point> getFeaturePoints(float percentage, int localMaximumSize);
	void showFeaturePoints(std::vector<cv::Point> &pts, int radius);
	
	

private:
	Derivative computeDerivative();
	void computeResponse(Derivative &derivative);
	void applyGaussToDerivative(Derivative &derivative);
	
	int filterSize;
	float k;
	
	cv::Mat image;
	cv::Mat grayImage;
	cv::Mat harrisResponse;
};