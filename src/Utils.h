#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

struct pointData
{
	float energy;
	cv::Point pt;

	//for non maximal suppression (MSOP)
	int distance;

	pointData() {
		this->energy = 0;
		this->pt = cv::Point(0, 0);
		this->distance = 0;
	};

	pointData(float _energy, cv::Point _pt) {
		this->energy = _energy;
		this->pt = _pt;
		this->distance = 0;
	};
};

struct Derivative
{
	cv::Mat Ix;
	cv::Mat Iy;
	cv::Mat Ixy;

	Derivative() {
		Ix = cv::Mat();
		Iy = cv::Mat();
		Ixy = cv::Mat();
	};

	Derivative(cv::Mat _Ix, cv::Mat _Iy, cv::Mat _Ixy) {
		this->Ix = _Ix;
		this->Iy = _Iy;
		this->Ixy = _Ixy;
	};
};