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

void MSOP::getFeaturePoints(int selectNum)
{
	//std::vector<std::vector<cv::Point>> pyramidPoints;
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
		this->featurePoints.push_back(result);
	}
}

void MSOP::setFeatureDescription()
{
	cv::Mat grayImage, filterImage, localWindow;
	for (int i = 0; i < this->pyramidGrayImages.size(); i++) {
		grayImage = this->pyramidGrayImages[i];
		std::vector<FeaturePoint> fp;
		for (int j = 0; j < this->featurePoints[i].size(); j++) {
			//get 11 * 11 local window of feature points
			localWindow = cv::Mat(11, 11, CV_32F);
			bool flag = true;
			for (int m = -5; m < 6; m++) {
				int newX = this->featurePoints[i][j].x + m;
				if (newX < 0 || newX >= grayImage.rows || !flag) {
					flag = false;
					break;
				}
				for (int n = -5; n < 6; n++) {
					int newY = this->featurePoints[i][j].y + n;
					if (newY < 0 || newY >= grayImage.cols) {
						flag = false;
						break;
					}
					localWindow.at<float>(m + 5, n + 5) = grayImage.at<float>(newX, newY);
				}
			}

			// ignore when the window is out of image boundary  
			if (!flag)
				continue;

			//apply gaussian filter
			cv::GaussianBlur(localWindow, filterImage, cv::Size(11, 11), 4.5f, 4.5f);

			//decide major orientaition
			cv::Mat votes = cv::Mat::zeros(11, 11, CV_32F);
			std::vector<float> thetas;

			//calculate votes and theta
			float voteBox[36] = { 0 };
			float dX, dY, length, rad, theta;
			for (int y = 1; y < 10; y++) {
				for (int x = 1; x < 10; x++) {
					dX = filterImage.at<float>(y, x + 1) - filterImage.at<float>(y, x - 1);
					dY = filterImage.at<float>(y + 1, x) - filterImage.at<float>(y - 1, x);
					rad = atan2(dY, dX);
					theta = rad * 180 / CV_PI >= 0 ? rad * 180 / CV_PI : rad * 180 / CV_PI + 360;
					thetas.push_back(theta);

					length = sqrt(dX * dX + dY * dY);
					votes.at<float>(y, x) = length;

					//std::cout << theta << " ";
					//std::cout << length << std::endl;
				}
			}

			//apply gaussian filter to votes
			cv::GaussianBlur(votes, votes, cv::Size(11, 11), 1.5f, 1.5f);
			int count = 0;
			for (int y = 1; y < 10; y++) {
				for (int x = 1; x < 10; x++) {
					voteBox[(int)(thetas[count] / 10)] += votes.at<float>(y, x);
					count++;
				}
			}

			//get major orientaion
			int maxIndex = 0;
			int maxVote = voteBox[0];
			for (int k = 1; k < 36; k++) {
				if (voteBox[k] > maxVote) {
					maxIndex = k;
					maxVote = voteBox[k];
				}
			}

			//rotate 41 * 41 window by major theta
			int major_theta = maxIndex * 10;
			cv::Mat majorWindow = cv::Mat(41, 41, CV_32F);
			flag = true;
			for (int r = -20; r < 21; r++) {
				for (int c = -20; c < 21; c++) {
					rad = major_theta / 180 * CV_PI;
					//implement subpixel
					int newC = cos(rad) * c - sin(rad) * r + this->featurePoints[i][j].y;
					int newR = sin(rad) * c + cos(rad) * r + this->featurePoints[i][j].x;
					if (newC < 0 || newC >= grayImage.cols || newR < 0 || newR >= grayImage.rows) {
						flag = false;
						break;
					}
					majorWindow.at<float>(r + 20, c + 20) = grayImage.at<float>(newR, newC);
				}
			}

			// ignore when the window is out of image boundary
			if (!flag)
				continue;

			//resize to 8 * 8 window
			cv::resize(majorWindow, majorWindow, cv::Size(8, 8));
			cv::Mat mean, stddev;
			cv::meanStdDev(majorWindow, mean, stddev);

			//normalize
			for (int r = 0; r < majorWindow.rows; r++) {
				for (int c = 0; c < majorWindow.cols; c++) {
					majorWindow.at<float>(r, c) = (majorWindow.at<float>(r, c) - mean.at<double>(0, 0)) / stddev.at<double>(0, 0);
					//std::cout << majorWindow.at<float>(r, c) << " ";
				}
				//std::cout << std::endl;
			}
			fp.push_back(FeaturePoint(this->featurePoints[i][j], major_theta, majorWindow.clone()));
		}
		this->featureDescriptions.push_back(fp);
		fp.clear();
	}
}

std::vector<std::vector<FeaturePoint>>& MSOP::getFeatureDescription()
{
	return this->featureDescriptions;
}

int MSOP::getPyramidDepth()
{
	return this->pyramidDepth;
}

cv::Mat& MSOP::getImage() 
{
	return this->image;
}

std::vector<cv::Mat>& MSOP::getPyramidImages()
{
	return this->pyramidImages;
}

void MSOP::showFeaturePoints(int radius)
{
	int newX, newY;
	cv::namedWindow("ImgViewer", 1);
	for (int i = 0; i < this->pyramidImages.size(); i++) {
		cv::Mat showImg = this->pyramidImages[i].clone();
		
		for (cv::Point &point : this->featurePoints[i]) {
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
				if (trace == 0)
					harrisResponse.at<float>(i, j) = 0;
				else
					harrisResponse.at<float>(i, j) = det / trace;
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