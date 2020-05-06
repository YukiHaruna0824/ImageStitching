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

void ImageUtils::showMatchResult(HarrisDetector &lhs, HarrisDetector &rhs, std::vector<cv::Vec2i> &match)
{
	std::vector<FeaturePoint> lhsFeaturePoints = lhs.getFeatureDescription();
	std::vector<FeaturePoint> rhsFeaturePoints = rhs.getFeatureDescription();
	std::vector<cv::KeyPoint> lhsKp;
	std::vector<cv::KeyPoint> rhsKp;
	std::vector<cv::DMatch> dms;
	for (int i = 0; i < lhsFeaturePoints.size(); i++)
		lhsKp.push_back(cv::KeyPoint(cv::Point(lhsFeaturePoints[i].pt.y, lhsFeaturePoints[i].pt.x), 2));
	for (int i = 0; i < rhsFeaturePoints.size(); i++)
		rhsKp.push_back(cv::KeyPoint(cv::Point(rhsFeaturePoints[i].pt.y, rhsFeaturePoints[i].pt.x), 2));
	for (int i = 0; i < match.size(); i++)
		dms.push_back(cv::DMatch(match[i][0], match[i][1], -1));

	cv::Mat out;
	cv::drawMatches(lhs.getImage(), lhsKp, rhs.getImage(), rhsKp, dms, out);
	cv::imshow("MatchResult", out);
	cv::waitKey();
}

void ImageUtils::showMatchResult(MSOP &lhs, MSOP &rhs, std::vector<std::vector<cv::Vec2i>> &matches)
{
	for (int k = 0; k < matches.size(); k++) {
		std::vector<FeaturePoint> lhsFeaturePoints = lhs.getFeatureDescription()[k];
		std::vector<FeaturePoint> rhsFeaturePoints = rhs.getFeatureDescription()[k];

		std::vector<cv::KeyPoint> lhsKp;
		std::vector<cv::KeyPoint> rhsKp;
		std::vector<cv::DMatch> dms;

		for (int i = 0; i < lhsFeaturePoints.size(); i++)
			lhsKp.push_back(cv::KeyPoint(cv::Point(lhsFeaturePoints[i].pt.y, lhsFeaturePoints[i].pt.x), 2));
		for (int i = 0; i < rhsFeaturePoints.size(); i++)
			rhsKp.push_back(cv::KeyPoint(cv::Point(rhsFeaturePoints[i].pt.y, rhsFeaturePoints[i].pt.x), 2));

		for (int i = 0; i < matches[k].size(); i++) {
			dms.push_back(cv::DMatch(matches[k][i][0], matches[k][i][1], -1));
		}

		cv::Mat out;
		cv::drawMatches(lhs.getPyramidImages()[k], lhsKp, rhs.getPyramidImages()[k], rhsKp, dms, out);
		cv::imshow("MatchResult", out);
		cv::waitKey();
	}
}

std::vector<cv::Vec2i> ImageUtils::getMatchFeaturePoints(HarrisDetector &lhs, HarrisDetector &rhs, float ratio)
{
	std::vector<FeaturePoint> lhsFeaturePoints = lhs.getFeatureDescription();
	std::vector<FeaturePoint> rhsFeaturePoints = rhs.getFeatureDescription();
	std::vector<cv::Vec2i> match;

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
			match.push_back(cv::Vec2i(i, lowIndex));
		}
	}
	return match;
}

std::vector<std::vector<cv::Vec2i>> ImageUtils::getMatchFeaturePoints(MSOP &lhs, MSOP &rhs, float ratio)
{
	std::vector<std::vector<cv::Vec2i>> matches;
	std::vector<cv::Vec2i> match;
	
	int pyramidDepth = lhs.getPyramidDepth();
	for (int k = 0; k < pyramidDepth; k++) {

		std::vector<FeaturePoint> lhsFeaturePoints = lhs.getFeatureDescription()[k];
		std::vector<FeaturePoint> rhsFeaturePoints = rhs.getFeatureDescription()[k];
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
				match.push_back(cv::Vec2i(i, lowIndex));
			}
		}
		matches.push_back(match);
		match.clear();
	}
	return matches;
}

std::vector<cv::Mat>& ImageUtils::getImages()
{
	return this->images;
}

std::vector<float>& ImageUtils::getFocals()
{
	return this->focalLengths;
}