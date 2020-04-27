#include "HarrisDetector.h"

HarrisDetector::HarrisDetector(cv::Mat image)
{
	this->image = image.clone();
	this->k = 0.05;
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
	cv::Mat grayImage = cv::Mat(this->image.rows, this->image.cols, CV_32F);
	for (int c = 0; c < grayImage.cols; c++) {
		for (int r = 0; r < grayImage.rows; r++) {
			grayImage.at<float>(r, c) =
				0.114 * this->image.at<cv::Vec3b>(r, c)[0] +
				0.587 * this->image.at<cv::Vec3b>(r, c)[1] +
				0.299 * this->image.at<cv::Vec3b>(r, c)[2];
		}
	}
	
	std::vector<cv::Mat> derivative = this->computeDerivative(grayImage);
	this->applyGuassToDerivative(derivative);
	computeResponse(derivative);
}

std::vector<pointData> HarrisDetector::getFeaturePoints(float percentage, int localMaximumSize)
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
	std::sort(points.begin(), points.end(), [](const pointData& pt1, const pointData& pt2) {
		return pt1.energy > pt2.energy;
	});
	
	//std::cout << points.size() << std::endl;
	//std::cout << points[0].energy << " " << points[1].energy << std::endl;
	
	int topSize = this->harrisResponse.rows * this->harrisResponse.cols * percentage;
	cv::Mat checkValid = cv::Mat::zeros(this->harrisResponse.rows, this->harrisResponse.cols, CV_8U);
	for (int i = 0; i < topSize; i++) {
		checkValid.at<uchar>(points[i].pt.x, points[i].pt.y) = 1;
	}

	//Find local maximum
	std::vector<pointData> result;
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
				pointData ptdata;
				ptdata.energy = this->harrisResponse.at<float>(maxPoint.x, maxPoint.y);
				//convert to original coordinate system
				ptdata.pt = cv::Point(maxPoint.x + 1, maxPoint.y + 1);
				result.push_back(ptdata);
			}
		}
	}
	
	//std::cout << result.size() << std::endl;
	return result;
}

void HarrisDetector::showFeaturePoints(std::vector<pointData> pts)
{
	cv::Mat showImg = this->image.clone();
	int radius = 3;
	
	int newX, newY;
	for (pointData &point : pts){
		for (int r = -radius; r < radius; r++) {
			newX = std::max(0, std::min(point.pt.x + r, showImg.rows));
			newY = std::max(0, std::min(point.pt.y + radius, showImg.cols));
			showImg.at<cv::Vec3b>(newX, newY) = cv::Vec3b(0, 0, 255);
		}
		for (int r = -radius; r < radius; r++) {
			newX = std::max(0, std::min(point.pt.x + r, showImg.rows));
			newY = std::max(0, std::min(point.pt.y - radius, showImg.cols));
			showImg.at<cv::Vec3b>(newX, newY) = cv::Vec3b(0, 0, 255);
		}
		for (int r = -radius; r < radius; r++){
			newX = std::max(0, std::min(point.pt.x - radius, showImg.rows));
			newY = std::max(0, std::min(point.pt.y + r, showImg.cols));
			showImg.at<cv::Vec3b>(newX, newY) = cv::Vec3b(0, 0, 255);
		}
		for (int r = -radius; r < radius; r++) {
			newX = std::max(0, std::min(point.pt.x + radius, showImg.rows));
			newY = std::max(0, std::min(point.pt.y + r, showImg.cols));
			showImg.at<cv::Vec3b>(newX, newY) = cv::Vec3b(0, 0, 255);
		}

		//showImg.at<cv::Vec3b>(point.pt.x, point.pt.y) = cv::Vec3b(0, 0, 255);
	}
	
	cv::namedWindow("ImgViewer", 1);
	cv::imshow("ImgViewer", showImg);
	cv::waitKey(0);
}

std::vector<cv::Mat> HarrisDetector::computeDerivative(cv::Mat grayImage)
{
	int height = grayImage.rows;
	int width = grayImage.cols;
	
	cv::Mat sobel_h = cv::Mat(height - 2, width, CV_32F);

	float a1, a2, a3;
	for (int i = 1; i < height - 1; i++) {
		for (int j = 0; j < width; j++) {
			a1 = grayImage.at<float>(i - 1, j);
			a2 = grayImage.at<float>(i, j);
			a3 = grayImage.at<float>(i + 1, j);
			sobel_h.at<float>(i - 1, j) = a1 + 2 * a2 + a3;
		}
	}

	cv::Mat sobel_w = cv::Mat(height , width - 2, CV_32F);
	for (int i = 0; i < height; i++) {
		for (int j = 1; j < width - 1; j++) {
			a1 = grayImage.at<float>(i, j - 1);
			a2 = grayImage.at<float>(i, j);
			a3 = grayImage.at<float>(i, j + 1);
			sobel_w.at<float>(i, j - 1) = a1 + 2 * a2 + a3;
		}
	}

	cv::Mat Ix = cv::Mat(height - 2, width - 2, CV_32F);
	cv::Mat Iy = cv::Mat(height - 2, width - 2, CV_32F);
	//cv::Mat Ixy = cv::Mat(height - 2, width - 2, CV_32F);

	for (int i = 0; i < height - 2; i++) {
		for (int j = 0; j < width - 2; j++) {
			Ix.at<float>(i, j) = -sobel_w.at<float>(i, j) + sobel_w.at<float>(i + 2, j);
			Iy.at<float>(i, j) = sobel_h.at<float>(i, j) - sobel_h.at<float>(i, j + 2);
			//Ixy.at<float>(i, j) = Ix.at<float>(i, j) * Iy.at<float>(i, j);
		}
	}
	
	std::vector<cv::Mat> result = {Ix, Iy};
	return result;
}

void HarrisDetector::computeResponse(std::vector<cv::Mat> derivative)
{
	this->harrisResponse = cv::Mat(derivative[1].rows, derivative[0].cols, CV_32F);
	float a00, a01, a10, a11, det, trace;
	for (int i = 0; i < this->harrisResponse.rows; i++) {
		for (int j = 0; j < this->harrisResponse.cols; j++) {
			a00 = derivative[0].at<float>(i, j) * derivative[0].at<float>(i, j);
			a01 = derivative[0].at<float>(i, j) * derivative[1].at<float>(i, j);
			a10 = derivative[0].at<float>(i, j) * derivative[1].at<float>(i, j);
			a11 = derivative[1].at<float>(i, j) * derivative[1].at<float>(i, j);
			det = a00 * a11 - a01 * a10;
			trace = a00 + a11;

			this->harrisResponse.at<float>(i, j) = fabs(det - this->k * trace * trace);
		}
	}
}

void HarrisDetector::applyGuassToDerivative(std::vector<cv::Mat>& derivative)
{
	if (this->filterSize == 0)
		return;

	for (int i = 0; i < derivative.size(); i++)
		cv::GaussianBlur(derivative[i], derivative[i], cv::Size(this->filterSize, this->filterSize), 0.707);

}