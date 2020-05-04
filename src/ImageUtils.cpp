#include "ImageUtils.h"

void ImageUtils::parseImageInfo(std::string rootFolder)
{
	std::string infoPath = rootFolder + "/pano.txt";
	std::fstream fs;
	fs.open(infoPath, std::ios::in);

	if (fs) 
	{
		std::string imageName;
		float focalLength = 0.0f;
		while (fs >> imageName >> focalLength)
		{
			std::string imagePath = rootFolder + "/" + imageName;
			cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
			this->images.push_back(image.clone());
			this->focalLengths.push_back(focalLength);
		}
	}
	else
		std::cout << "Open File Failed!" << std::endl;
}

float ImageUtils::computeFeaturePointDistance(cv::Mat &lhs, cv::Mat &rhs)
{
	// compute L2 distance
	float dis = 0;
	for (int r = 0; r < lhs.rows; r++) {
		for (int c = 0; c < rhs.cols; c++) {
			dis += (lhs.at<float>(r, c) - rhs.at<float>(r, c)) * (lhs.at<float>(r, c) - rhs.at<float>(r, c));
		}
	}
	return sqrt(dis);
}

void ImageUtils::matchFeaturePoints(MSOP &lhs, MSOP &rhs, float ratio)
{
	int pyramidDepth = lhs.getPyramidDepth();

	for (int k = 0; k < pyramidDepth; k++) {

		std::vector<FeaturePoint> lhsFeaturePoints = lhs.getFeatureDescription()[k];
		std::vector<FeaturePoint> rhsFeaturePoints = rhs.getFeatureDescription()[k];

		std::vector<cv::KeyPoint> lhsKp;
		std::vector<cv::KeyPoint> rhsKp;
		std::vector<cv::DMatch> dms;

		for (int i = 0; i < lhsFeaturePoints.size(); i++)
			lhsKp.push_back(cv::KeyPoint(cv::Point(lhsFeaturePoints[i].pt.y, lhsFeaturePoints[i].pt.x), 2));
		for (int i = 0; i < rhsFeaturePoints.size(); i++)
			rhsKp.push_back(cv::KeyPoint(cv::Point(rhsFeaturePoints[i].pt.y, rhsFeaturePoints[i].pt.x), 2));
		
		for (int i = 0; i < lhsFeaturePoints.size(); i++) {
			int lowIndex = 0;
			float lowDis = FLT_MAX;
			int secondIndex = 0;
			float secondDis = FLT_MAX;
			for (int j = 0; j < rhsFeaturePoints.size(); j++) {
				//get two closet feature match
				float d = computeFeaturePointDistance(lhsFeaturePoints[i].featureVector, rhsFeaturePoints[j].featureVector);
				if (d < lowDis) {
					secondIndex = lowIndex;
					secondDis = lowDis;
					lowIndex = j;
					lowDis = d;
				}
				else if (d < secondDis) {
					secondIndex = j;
					secondDis = d;
				}
			}
			//std::cout << lowDis / secondDis << std::endl;
			if (lowDis / secondDis < ratio) {
				dms.push_back(cv::DMatch(i, lowIndex, lowDis));
			}
		}

		cv::Mat out;
		cv::drawMatches(lhs.getPyramidImages()[k], lhsKp, rhs.getPyramidImages()[k], rhsKp, dms, out);
		cv::imshow("match", out);
		cv::waitKey();
	}
}

std::vector<cv::Mat>& ImageUtils::getImages()
{
	return this->images;
}

std::vector<float>& ImageUtils::getFocals()
{
	return this->focalLengths;
}