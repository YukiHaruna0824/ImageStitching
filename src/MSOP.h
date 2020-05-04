#pragma once

#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "Utils.h"

class MSOP
{
public:
	MSOP(cv::Mat image);
	MSOP(cv::Mat image, float sigmaP, float sigmaI, float sigmaD, int filterSize, int pyramidDepth);

	void findHarrisResponse();

	void getFeaturePoints(int selectNum);
	void showFeaturePoints(int radius);

	void setFeatureDescription();
	std::vector<std::vector<FeaturePoint>>& getFeatureDescription();

	int getPyramidDepth();
	cv::Mat& getImage();
	std::vector<cv::Mat>& getPyramidImages();

private:
	void constructPyramid();
	std::vector<Derivative> computeDerivative();
	void computeResponse(std::vector<Derivative> &derivative);

	void applyGaussToImages();
	void applyGaussToDerivative(std::vector<Derivative> &derivative);

	cv::Mat image;
	
	std::vector<cv::Mat> pyramidImages;
	std::vector<cv::Mat> pyramidGrayImages;
	std::vector<cv::Mat> harrisResponses;

	std::vector<std::vector<cv::Point>> featurePoints;

	std::vector<std::vector<FeaturePoint>> featureDescriptions;


	float sigmaP, sigmaI, sigmaD;
	int filterSize;
	int pyramidDepth;
};