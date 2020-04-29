#include "MSOP.h"


MSOP::MSOP(cv::Mat image)
{
	this->image = image;
	this->sigmaP = 1.0;
	this->sigmaI = 1.5;
	this->sigmaD = 1.0;
	this->filterSize = 3;
	this->pyramidDepth = 3;
}

MSOP::MSOP(cv::Mat image, float sigmaP, float sigmaI, float sigmaD, int filterSize, int pyramidDepth)
{
	this->image = image;
	this->sigmaP = sigmaP;
	this->sigmaI = sigmaI;
	this->sigmaD = sigmaD;
	this->filterSize = filterSize;
	this->pyramidDepth = pyramidDepth;
}

void MSOP::findHarrisResponse()
{
	constructPyramid();
	applyGaussToImages();

	std::vector<Derivative> derivatives = computeDerivative();
	applyGaussToDerivative(derivatives);

	computeResponse(derivatives);
}

std::vector<std::vector<cv::Point>> MSOP::getFeaturePoints(int selectNum)
{
	std::vector<std::vector<cv::Point>> pyramidPoints;
	for (int k = 0; k < this->harrisResponses.size(); k++) {
		int height = this->harrisResponses[k].rows;
		int width = this->harrisResponses[k].cols;

		//find local maxima 3*3 window larger than 10
		std::vector<pointData> points;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				float e = this->harrisResponses[k].at<float>(i, j);;
				bool flag = true;

				for (int m = -1; m < 2; m++) {
					if (i + m >= height || i + m < 0)
						continue;
					for (int n = -1; n < 2; n++) {
						if (j + n >= width || j + n < 0)
							continue;
						
						float e1 = this->harrisResponses[k].at<float>(i + m, j + n);
						if (e1 > e) {
							flag = false;
							break;
						}
					}
				}
				if(e >= 10 && flag)
					points.push_back(pointData(e, cv::Point(i, j)));
			}
		}

		//adaptive non maximal suppression
		for (int i = 0; i < points.size(); i++) {
			int minD = INT_MAX;
			for (int j = 0; j < points.size(); j++) {
				if (points[j].energy > points[i].energy) {
					int d = (points[j].pt.x - points[i].pt.x) * (points[j].pt.x - points[i].pt.x)
						+ (points[j].pt.y - points[i].pt.y) * (points[j].pt.y - points[i].pt.y);
					if (d < minD)
						minD = d;
				}
			}
			points[i].distance = minD;
		}

		std::sort(points.begin(), points.end(), [](const pointData &lhs, const pointData &rhs) {
			return lhs.distance > rhs.distance;
		});

		std::vector<cv::Point> result;
		for (int i = 0; i < selectNum; i++) {
			if (i >= points.size())
				break;
			//convert to original coordinate system
			result.push_back(cv::Point(points[i].pt.x + 1, points[i].pt.y + 1));
		}
		pyramidPoints.push_back(result);
	}
	return pyramidPoints;
}

void MSOP::showFeaturePoints(std::vector<std::vector<cv::Point>> &pts, int radius)
{
	int newX, newY;
	cv::namedWindow("ImgViewer", 1);
	for (int i = 0; i < this->pyramidImages.size(); i++) {
		cv::Mat showImg = this->pyramidImages[i].clone();
		
		for (cv::Point &point : pts[i]) {
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
			for (int r = -radius; r < radius; r++) {
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
		cv::imshow("ImgViewer", showImg);
		cv::waitKey(0);
	}
}

void MSOP::constructPyramid()
{
	cv::Mat copyImage = this->image.clone();
	cv::Mat grayImage = cv::Mat(this->image.rows, this->image.cols, CV_32F);
	for (int c = 0; c < grayImage.cols; c++) {
		for (int r = 0; r < grayImage.rows; r++) {
			grayImage.at<float>(r, c) =
				0.114 * this->image.at<cv::Vec3b>(r, c)[0] +
				0.587 * this->image.at<cv::Vec3b>(r, c)[1] +
				0.299 * this->image.at<cv::Vec3b>(r, c)[2];
		}
	}

	this->pyramidImages.push_back(copyImage.clone());
	this->pyramidGrayImages.push_back(grayImage.clone());

	for (int i = 1; i < this->pyramidDepth; i++) {
		cv::GaussianBlur(grayImage, grayImage, cv::Size(this->filterSize, this->filterSize), sigmaP, sigmaP);
		cv::resize(grayImage, grayImage, cv::Size(grayImage.cols / 2 , grayImage.rows / 2));
		cv::resize(copyImage, copyImage, cv::Size(copyImage.cols / 2, copyImage.rows / 2));
		this->pyramidImages.push_back(copyImage.clone());
		this->pyramidGrayImages.push_back(grayImage.clone());
	}
}

std::vector<Derivative> MSOP::computeDerivative()
{
	std::vector<Derivative> result;

	for (int k = 0; k < this->pyramidGrayImages.size(); k++) {
		int height = this->pyramidGrayImages[k].rows;
		int width = this->pyramidGrayImages[k].cols;

		cv::Mat sobel_h = cv::Mat(height - 2, width, CV_32F);
		float a1, a2, a3;
		for (int i = 1; i < height - 1; i++) {
			for (int j = 0; j < width; j++) {
				a1 = this->pyramidGrayImages[k].at<float>(i - 1, j);
				a2 = this->pyramidGrayImages[k].at<float>(i, j);
				a3 = this->pyramidGrayImages[k].at<float>(i + 1, j);
				sobel_h.at<float>(i - 1, j) = a1 + 2 * a2 + a3;
			}
		}

		cv::Mat sobel_w = cv::Mat(height, width - 2, CV_32F);
		for (int i = 0; i < height; i++) {
			for (int j = 1; j < width - 1; j++) {
				a1 = this->pyramidGrayImages[k].at<float>(i, j - 1);
				a2 = this->pyramidGrayImages[k].at<float>(i, j);
				a3 = this->pyramidGrayImages[k].at<float>(i, j + 1);
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
	
		result.push_back(Derivative(Ix, Iy, Ixy));
	}
	return result;
}

void MSOP::computeResponse(std::vector<Derivative> &derivative)
{
	for (int k = 0; k < derivative.size(); k++) {
		cv::Mat harrisResponse = cv::Mat(derivative[k].Iy.rows, derivative[k].Ix.cols, CV_32F);
		float a00, a01, a10, a11, det, trace;
		for (int i = 0; i < harrisResponse.rows; i++) {
			for (int j = 0; j < harrisResponse.cols; j++) {
				a00 = derivative[k].Ix.at<float>(i, j);
				a01 = derivative[k].Ixy.at<float>(i, j);
				a10 = derivative[k].Ixy.at<float>(i, j);
				a11 = derivative[k].Iy.at<float>(i, j);
				det = a00 * a11 - a01 * a10;
				trace = a00 + a11;
				harrisResponse.at<float>(i, j) = det / (trace + 0.000001);
			}
		}
		this->harrisResponses.push_back(harrisResponse);
	}
}

void MSOP::applyGaussToImages()
{
	for (int i = 0; i < this->pyramidGrayImages.size(); i++)
		cv::GaussianBlur(this->pyramidGrayImages[i], this->pyramidGrayImages[i], cv::Size(this->filterSize, this->filterSize), sigmaD, sigmaD);
}

void MSOP::applyGaussToDerivative(std::vector<Derivative> &derivative)
{
	if (this->filterSize == 0)
		return;

	for (int i = 0; i < derivative.size(); i++) {
		cv::GaussianBlur(derivative[i].Ix, derivative[i].Ix, cv::Size(this->filterSize, this->filterSize), sigmaI, sigmaI);
		cv::GaussianBlur(derivative[i].Iy, derivative[i].Iy, cv::Size(this->filterSize, this->filterSize), sigmaI, sigmaI);
		cv::GaussianBlur(derivative[i].Ixy, derivative[i].Ixy, cv::Size(this->filterSize, this->filterSize), sigmaI, sigmaI);
	}

}