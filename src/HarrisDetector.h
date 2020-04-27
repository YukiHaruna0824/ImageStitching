#pragma once

#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

struct pointData
{
	float energy;
	cv::Point pt;
	pointData() {
		energy = 0;
		pt = cv::Point(0, 0);
	};

	pointData(float _energy, cv::Point _pt) {
		energy = _energy;
		pt = _pt;
	};
};

class HarrisDetector
{
public:
	HarrisDetector(cv::Mat image);
	HarrisDetector(cv::Mat image, float k, int filterRange);
	void findHarrisResponse();

	std::vector<pointData> getFeaturePoints(float percentage, int localMaximumSize);
	void showFeaturePoints(std::vector<pointData> pts);
	
private:
	std::vector<cv::Mat> computeDerivative(cv::Mat grayImage);
	void computeResponse(std::vector<cv::Mat> derivative);
	void applyGuassToDerivative(std::vector<cv::Mat>& derivative);
	
	int filterSize;
	float k;
	
	cv::Mat image;
	cv::Mat harrisResponse;
};