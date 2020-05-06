#include "HarrisDetector.h"

HarrisDetector::HarrisDetector(cv::Mat image)
{
	this->image = image.clone();
	this->k = 0.04;
	this->filterSize = 3;
}

HarrisDetector::HarrisDetector(cv::Mat image, float k, int filterSize)
{
	this->image = image.clone();
	this->k = k;
	this->filterSize = filterSize;
}

void HarrisDetector::findHarrisResponse()
{
	this->grayImage = cv::Mat(this->image.rows, this->image.cols, CV_32F);
	for (int c = 0; c < this->grayImage.cols; c++) {
		for (int r = 0; r < this->grayImage.rows; r++) {
			this->grayImage.at<float>(r, c) =
				0.114 * this->image.at<cv::Vec3b>(r, c)[0] +
				0.587 * this->image.at<cv::Vec3b>(r, c)[1] +
				0.299 * this->image.at<cv::Vec3b>(r, c)[2];
		}
	}

	cv::GaussianBlur(this->grayImage, this->grayImage, cv::Size(this->filterSize, this->filterSize), 0.707, 0.707);

	//get Ix^2, Iy^2, Ixy
	Derivative derivative = this->computeDerivative();
	this->applyGaussToDerivative(derivative);
	computeResponse(derivative);
}

void HarrisDetector::getFeaturePoints(float percentage, int localMaximumSize)
{
	//find threshold
	std::vector<pointData> points;
	
	for (int i = 0; i < this->harrisResponse.rows; i++) {
		for (int j = 0; j < this->harrisResponse.cols; j++) {
			pointData ptData;
			ptData.energy = this->harrisResponse.at<float>(i, j);

			ptData.pt = cv::Point(i, j);
			points.push_back(ptData);
		}
	}
	std::sort(points.begin(), points.end(), [](const pointData &lhs, const pointData &rhs) {
		return lhs.energy > rhs.energy;
	});
	
	//std::cout << points.size() << std::endl;
	//std::cout << points[0].energy << " " << points[1].energy << std::endl;
	
	int topSize = this->harrisResponse.rows * this->harrisResponse.cols * percentage;
	cv::Mat checkValid = cv::Mat::zeros(this->harrisResponse.rows, this->harrisResponse.cols, CV_8U);
	for (int i = 0; i < topSize; i++) {
		checkValid.at<uchar>(points[i].pt.x, points[i].pt.y) = 1;
	}

	//Find local maximum
	for (int i = 0; i < this->harrisResponse.rows; i += localMaximumSize) {
		for (int j = 0; j < this->harrisResponse.cols; j += localMaximumSize) {
			float maxE = -FLT_MAX;
			cv::Point maxPoint = cv::Point(0, 0);
			bool flag = false;

			for (int m = 0; m < localMaximumSize; m++) {
				if (i + m >= this->harrisResponse.rows)
					continue;
				for (int n = 0; n < localMaximumSize; n++) {
					
					if (j + n >= this->harrisResponse.cols)
						continue;

					//the pixel is not bigger than threshold energy
					if (checkValid.at<uchar>(i + m, j + n) == 0)
						continue;

					float e = this->harrisResponse.at<float>(i + m, j + n);
					if (e > maxE) {
						flag = true;
						maxE = e;
						maxPoint = cv::Point(i + m, j + n);
					}
				}
			}

			if (flag) {
				//convert to original coordinate system
				this->featurePoints.push_back(cv::Point(maxPoint.x + 1, maxPoint.y + 1));
			}
		}
	}
	
	//std::cout << this->featurePoints.size() << std::endl;
}

void HarrisDetector::showFeaturePoints(int radius)
{
	cv::Mat showImg = this->image.clone();	
	int newX, newY;
	for (cv::Point &point : this->featurePoints){
		for (int r = -radius; r < radius; r++) {
			newX = std::max(0, std::min(point.x + r, showImg.rows - 1));
			newY = std::max(0, std::min(point.y + radius, showImg.cols - 1));
			showImg.at<cv::Vec3b>(newX, newY) = cv::Vec3b(0, 0, 255);
		}
		for (int r = -radius; r < radius; r++) {
			newX = std::max(0, std::min(point.x + r, showImg.rows - 1));
			newY = std::max(0, std::min(point.y - radius, showImg.cols - 1));
			showImg.at<cv::Vec3b>(newX, newY) = cv::Vec3b(0, 0, 255);
		}
		for (int r = -radius; r < radius; r++){
			newX = std::max(0, std::min(point.x - radius, showImg.rows - 1));
			newY = std::max(0, std::min(point.y + r, showImg.cols - 1));
			showImg.at<cv::Vec3b>(newX, newY) = cv::Vec3b(0, 0, 255);
		}
		for (int r = -radius; r < radius; r++) {
			newX = std::max(0, std::min(point.x + radius, showImg.rows - 1));
			newY = std::max(0, std::min(point.y + r, showImg.cols - 1));
			showImg.at<cv::Vec3b>(newX, newY) = cv::Vec3b(0, 0, 255);
		}

		//showImg.at<cv::Vec3b>(point.x, point.y) = cv::Vec3b(0, 0, 255);
	}
	
	cv::namedWindow("ImgViewer", 1);
	cv::imshow("ImgViewer", showImg);
	cv::waitKey(0);
}

void HarrisDetector::setFeatureDescription()
{
	cv::Mat feature = cv::Mat(3, 3, CV_32F);
	for (int k = 0; k < this->featurePoints.size(); k++) {
		for (int i = -1; i < 2; i++) {
			for (int j = -1; j < 2; j++) {
				feature.at<float>(i + 1, j + 1) = this->grayImage.at<float>(this->featurePoints[k].x + i, this->featurePoints[k].y + j);
			}
		}
		this->featureDescriptions.push_back(FeaturePoint(this->featurePoints[k], feature.clone()));
	}
}

std::vector<FeaturePoint>& HarrisDetector::getFeatureDescription()
{
	return this->featureDescriptions;
}

cv::Mat& HarrisDetector::getImage()
{
	return this->image;
}

Derivative HarrisDetector::computeDerivative()
{
	int height = this->grayImage.rows;
	int width = this->grayImage.cols;
	
	cv::Mat sobel_h = cv::Mat(height - 2, width, CV_32F);

	float a1, a2, a3;
	for (int i = 1; i < height - 1; i++) {
		for (int j = 0; j < width; j++) {
			a1 = this->grayImage.at<float>(i - 1, j);
			a2 = this->grayImage.at<float>(i, j);
			a3 = this->grayImage.at<float>(i + 1, j);
			sobel_h.at<float>(i - 1, j) = a1 + 2 * a2 + a3;
		}
	}

	cv::Mat sobel_w = cv::Mat(height , width - 2, CV_32F);
	for (int i = 0; i < height; i++) {
		for (int j = 1; j < width - 1; j++) {
			a1 = this->grayImage.at<float>(i, j - 1);
			a2 = this->grayImage.at<float>(i, j);
			a3 = this->grayImage.at<float>(i, j + 1);
			sobel_w.at<float>(i, j - 1) = a1 + 2 * a2 + a3;
		}
	}

	cv::Mat Ix = cv::Mat(height - 2, width - 2, CV_32F);
	cv::Mat Iy = cv::Mat(height - 2, width - 2, CV_32F);
	cv::Mat Ixy = cv::Mat(height - 2, width - 2, CV_32F);

	for (int i = 0; i < height - 2; i++) {
		for (int j = 0; j < width - 2; j++) {
			Ix.at<float>(i, j) = -sobel_w.at<float>(i, j) + sobel_w.at<float>(i + 2, j);
			Iy.at<float>(i, j) = sobel_h.at<float>(i, j) - sobel_h.at<float>(i, j + 2);
			Ixy.at<float>(i, j) = Ix.at<float>(i, j) * Iy.at<float>(i, j);

			Ix.at<float>(i, j) = Ix.at<float>(i, j) * Ix.at<float>(i, j);
			Iy.at<float>(i, j) = Iy.at<float>(i, j) * Iy.at<float>(i, j);
		}
	}
	return Derivative(Ix, Iy, Ixy);
}

void HarrisDetector::computeResponse(Derivative &derivative)
{
	this->harrisResponse = cv::Mat(derivative.Iy.rows, derivative.Ix.cols, CV_32F);
	float a00, a01, a10, a11, det, trace;
	for (int i = 0; i < this->harrisResponse.rows; i++) {
		for (int j = 0; j < this->harrisResponse.cols; j++) {
			a00 = derivative.Ix.at<float>(i, j);
			a01 = derivative.Ixy.at<float>(i, j);
			a10 = derivative.Ixy.at<float>(i, j);
			a11 = derivative.Iy.at<float>(i, j);
			det = a00 * a11 - a01 * a10;
			trace = a00 + a11;
			
			this->harrisResponse.at<float>(i, j) = fabs(det - this->k * trace * trace);
			//std::cout << this->harrisResponse.at<float>(i,j) << std::endl;
		}
	}
}

void HarrisDetector::applyGaussToDerivative(Derivative &derivative)
{
	if (this->filterSize == 0)
		return;

	cv::GaussianBlur(derivative.Ix, derivative.Ix, cv::Size(this->filterSize, this->filterSize), 0.707, 0.707);
	cv::GaussianBlur(derivative.Iy, derivative.Iy, cv::Size(this->filterSize, this->filterSize), 0.707, 0.707);
	cv::GaussianBlur(derivative.Ix, derivative.Ixy, cv::Size(this->filterSize, this->filterSize), 0.707, 0.707);
}