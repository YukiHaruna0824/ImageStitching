#pragma once

#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class ImageUtils
{
public:
	void parseImageInfo(std::string rootFolder);

	std::vector<cv::Mat>& getImages();

private:
	std::vector<cv::Mat> images;

};