#include "stdafx.h"

#include "XConfiguration.h"

#include "FrameMain.h"

#include "opencv2\highgui\highgui_c.h"



using namespace Pylon;
using namespace std;




extern bool g_bTerminated;
extern bool g_bCamerasAreOk;
extern bool g_bCalibrationExists; 


Size g_boardSize = Size(6/*points_per_row*/, 5/*points_per_colum*/); // describes matrix of centers that are on symmetric grid
Size g_boardQuadSize = Size(g_boardSize.width * 2/*points_per_row*/, g_boardSize.height * 2/*points_per_colum*/); // describes matrix of corners that can be found around each center (4 corners per center)
Size g_boardChessSize = Size(4/*points_per_row*/, 8/*points_per_colum*/); // describes matrix of centers that are on asymmetric grid
Size g_boardChessCornersSize = Size(g_boardChessSize.width * 2 - 1/*points_per_row*/, g_boardChessSize.height - 1/*points_per_colum*/); // describes matrix of corners on asymetric grid, i.e. where quads of same color connect. 
Size g_imageSize = Size(1280, 1024);

// Chess board:
// - 7 rows by 8 columns
// - using white squares
// - each row has 4 white squares
// - 28 centers = 4 * 7
// - outer hull of centers delimits the inner corners
// - 42 inner corners = (7 = 4*2 - 1) * (6 = 7 - 1)



double g_pattern_distance = 2.5; // 3 for grid of squares, 6.625 for chess board

size_t g_min_images = 12;
double g_aposteriory_minsdistance = 100;


vector<vector<Point2f>> imagePoints_left;
vector<vector<Point2f>> imagePoints_right;
vector<vector<Point2f>> stereoImagePoints_left;
vector<vector<Point2f>> stereoImagePoints_right;
vector<Mat> imageRaw_left;
vector<Mat> imageRaw_right;
vector<Mat> stereoImageRaw_left;
vector<Mat> stereoImageRaw_right;
bool g_images_are_collected;

vector<vector<Point3f> > g_objectPoints;

std::string cv_windows[5];




void DrawImageAndBoard(const std::string& aname, const std::string& window_name, Mat& cv_image, const vector<Point2f>& board) {
	Mat cv_image1;

	if (cv_image.type() != CV_8UC3) {
		cv::cvtColor(cv_image, cv_image1, COLOR_GRAY2RGB);
	}
	else {
		cv_image1 = cv_image.clone();
		//cv::cvtColor(cv_image, cv_image1, COLOR_GRAY2RGB);
	}

	cv::Size boardSize;
	if (board.size() == (g_boardSize.height * g_boardSize.width)) {
		boardSize = g_boardSize;
	}
	else
		if (board.size() == (g_boardQuadSize.height * g_boardQuadSize.width)) {
			boardSize = g_boardQuadSize;
		}
	if (board.size()) {
		int red = 0;
		int green = 255;
		for (auto& point : board) {
			circle(cv_image1, Point2i((int)point.x, (int)point.y), 3, Scalar(0, green * 255, red * 255), -1);
			red += 3;
			green -= 3;
		}
		//drawChessboardCorners(cv_image1, boardSize, Mat(board), true);
	}

	double fx = 161.0 / cv_image.rows; // Sep. 23

	HWND hwnd = (HWND)cvGetWindowHandle(window_name.c_str());
	RECT clrect;
	if (GetWindowRect(GetParent(hwnd), &clrect)) {
		fx = (double)(clrect.bottom - clrect.top) / (double)cv_image.rows;
	}


	cv::resize(cv_image1, cv_image1, cv::Size(0, 0), fx, fx, INTER_AREA);
	cv_image1 *= std::max(256 / (int)g_bytedepth_scalefactor, 1);

	int baseLine = 0;
	Size textSize = getTextSize(aname.c_str(), 1, 5, 5, &baseLine);
	Point textOrigin(cv_image1.cols - textSize.width - 50, cv_image1.rows - 2 * baseLine - 10);
	putText(cv_image1, aname.c_str(), textOrigin, FONT_HERSHEY_SCRIPT_SIMPLEX, 3, Scalar(0, 0, 255 * 255), 5);

	try {
		cv::imshow(window_name.c_str(), cv_image1);
	}
	catch (Exception& ex) {
		std::cout << "DrawImageAndBoard:" << ' ' << ex.msg << std::endl;
	}
}



bool CalibrationFileExists() {
	bool exists = false;
	std::string path_calibrate_file = std::string(g_path_calib_images_dir) + g_path_calibrate_file;
	if (MyGetFileSize(path_calibrate_file) > 0) {
		try {
			FileStorage fs(path_calibrate_file, FileStorage::READ);
			if (fs.isOpened()) {
				Mat cameraMatrix;
				fs["cameraMatrix_l"] >> cameraMatrix;
				exists = cameraMatrix.cols > 0;
				fs.release();
			}
		}
		catch (...) {
		}
	}
	return exists;
}




void ClassBlobDetector::findBlobs(const Mat& image, Mat& binaryImage, std::vector<Center>& centers, std::vector<vector<Point>>& contours) const {
	(void)image;
	centers.clear();
	contours.clear();

	vector<double> dists;

	g_max_boxsize_pixels = std::max(binaryImage.rows / 8, binaryImage.cols / 8);

	ushort params_blobColor = (ushort)params.blobColor * g_bytedepth_scalefactor;

	std::vector<ABox> boxes;
	std::vector<ClusteredPoint> points;



	int kmatN = binaryImage.cols / 50;
	if (kmatN <= 5) {
		kmatN = 5;
	}
	else {
		kmatN = 7;
	}
	Mat_<double> kmat = LoG(1.25 * (kmatN / 7.0), kmatN);


	
	unsigned int threshold_intensity = 250 * g_bytedepth_scalefactor;

	while (ProcessWinMessages());
	BlobCentersLoG(boxes, points, binaryImage, threshold_intensity, cv::Rect(), kmat);
	while (ProcessWinMessages());

	double desired_min_inertia = sqrt(_min_confidence);
	double ratio_threshold = desired_min_inertia * params.minCircularity;

	for (size_t boxIdx = 0; boxIdx < boxes.size(); boxIdx++) {
		auto& contour = boxes[boxIdx].contour;
		if (contour.size() == 0) {
			continue;
		}

		contours.push_back(contour);

		Center center;
		center.confidence = 1;
		Moments moms = moments(Mat(contour));
		if (params.filterByArea)
		{
			double area = moms.m00;
			if (area < params.minArea || area >= params.maxArea) {
				continue;
			}
		}



		double ratio = desired_min_inertia;
		if (params.filterByInertia)
		{
			double denominator = sqrt(pow(2 * moms.mu11, 2) + pow(moms.mu20 - moms.mu02, 2));
			const double eps = 1e-2;
			if (denominator > eps)
			{
				double cosmin = (moms.mu20 - moms.mu02) / denominator;
				double sinmin = 2 * moms.mu11 / denominator;
				double cosmax = -cosmin;
				double sinmax = -sinmin;

				double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
				double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
				ratio = imin / imax;
			}
			else
			{
				ratio = 1;
			}

			if (ratio < params.minInertiaRatio || ratio >= params.maxInertiaRatio) {
				continue;
			}

			center.confidence = ratio * ratio;
		}

		if (params.filterByCircularity)
		{
			double area = moms.m00;
			double perimeter = arcLength(Mat(contour), true);
			ratio *= 4 * CV_PI * area / (perimeter * perimeter);
		}
		else {
			ratio *= params.minCircularity;
		}

		if (ratio < ratio_threshold) {
			continue;
		}
	



		if (params.filterByConvexity)
		{
			std::vector<cv::Point> hull;
			cv::convexHull(Mat(contour), hull);

			double area = contourArea(Mat(contour));
			double hullArea = contourArea(Mat(hull));

			double convexityRatio = area / hullArea;
			if (convexityRatio < params.minConvexity || convexityRatio >= params.maxConvexity) {
				continue;
			}
		}



		center.location = Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

		if (params.filterByColor)
		{
			ushort color = binaryImage.at<ushort>(cvRound(center.location.y), cvRound(center.location.x));
			if (color != params_blobColor) {
				continue;
			}
		}

		dists.clear();
		for (size_t pointIdx = 0; pointIdx < contour.size(); pointIdx++)
		{
			Point2d pt = contour[pointIdx];
			dists.push_back(norm(center.location - pt));
		}
		std::sort(dists.begin(), dists.end());
		center.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.0;

		centers.push_back(center);
	}
}


void ClassBlobDetector::detectImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const cv::Mat&) { // from openCV blobdetector.cpp
	keypoints.clear();

	const Size boardSize = _chess_board? g_boardChessSize: g_boardSize;
	const int expectedCount = boardSize.height * boardSize.width;

	const double threshold_confidence = 0.3 * _min_confidence; 

	Mat grayscaleImage;

	cv::normalize(image, grayscaleImage, 0, 255 * g_bytedepth_scalefactor, NORM_MINMAX, CV_32FC1, Mat());

	vector < vector<Center> > centers;
	double thresh = params.minThreshold;
	for(size_t nstep = 0; nstep < 20 && thresh <= params.maxThreshold; thresh += params.thresholdStep, ++nstep) {
		Mat binImage;
		try {
			threshold(grayscaleImage, binImage, thresh, 255 * g_bytedepth_scalefactor, THRESH_BINARY);
		}
		catch(Exception& ex) {
			std::cout << "BlobDetector:" << ' ' << ex.msg << std::endl;
			return; 
		}

		Mat binarizedImage;
		binImage.convertTo(binarizedImage, CV_16UC1);

		if(_invert_binary) {
			uchar *val = (uchar*)binarizedImage.data;
			for(int j = 0, N = binarizedImage.cols * binarizedImage.rows; j < N; ++j, ++val) {
				*val = *val? 0: 255;
			}
		}

		vector < Center > curCenters;
		try {
			double scale = 480.0 / binarizedImage.rows;
			if (scale < 0.95) {
				cv::resize(binarizedImage, binarizedImage, cv::Size(0, 0), scale, scale, INTER_AREA);
				params.minArea *= scale * scale;
				params.maxArea *= scale * scale;
			}
			else {
				scale = 1;
			}

			std::vector<vector<Point>> contours;
			findBlobs(grayscaleImage, binarizedImage, curCenters, contours);

			Mat image0;
			cvtColor(binarizedImage, image0, COLOR_GRAY2RGB);
			for each(auto& points in contours)
			{
				int n = (int)points.size();
				const Point* p = &points[0];
				cv::polylines(image0, &p, &n, 1, true, Scalar(0, 255 * 256, 0), 1, LINE_AA);
			}
			for each (auto & center in curCenters) {
				circle(image0, center.location, 3, Scalar(0, 0, 255 * 256), -1);
			}
			double fx = 280.0 / binarizedImage.rows;
			Mat image1;
			cv::resize(image0, image1, cv::Size(0, 0), fx, fx, INTER_AREA);
			cv::imshow("IMAGECalibr3", image1);
			while (ProcessWinMessages());

			if (scale != 1) {
				std::for_each(curCenters.begin(), curCenters.end(), [scale, &image0](ClassBlobDetector::Center& center) {
					center.location.x /= scale;
					center.location.y /= scale;
					center.radius /= scale;
				});
				params.minArea /= scale * scale;
				params.maxArea /= scale * scale;
			}
		}
		catch(Exception& ex) {
			std::cout << "BlobDetector:" << ' ' << ex.msg << std::endl;
			return;
		}

		if(_min_confidence > 0.0) { 
			if(curCenters.size() < expectedCount) {
				continue; 
			}

			int count = 0; 
			for(auto& c : curCenters) {
				if(c.confidence >= _min_confidence) {
					++count;
				}
			}

			if(count < expectedCount && count >(expectedCount * 0.6)) {
				int count_threshold = 0;
				double sum = 0;
				for(auto& c : curCenters) {
					if(c.confidence >= threshold_confidence) {
						++count_threshold;
						sum += c.confidence;
					}
				}
				if(count_threshold < expectedCount) {
					continue; 
				}
				double mean = sum / count_threshold;
				if(count_threshold == expectedCount && mean > _min_confidence) {
					count = expectedCount;
					for(auto& c : curCenters) {
						if(c.confidence < _min_confidence && c.confidence >= threshold_confidence) {
							c.confidence = _min_confidence;
						}
					}
				}
				else 
				{
					double ee = 0;
					for(auto& c : curCenters) {
						if(c.confidence >= threshold_confidence) {
							ee += std::pow(c.confidence - mean, 2.0);
						}
					}
					double sigma = std::sqrt(ee / count_threshold); 
					double confidence = mean - 1.96 * sigma;
					if(confidence >= threshold_confidence) {
						count = 0;
						for(auto& c : curCenters) {
							if(c.confidence >= confidence) {
								++count;
								c.confidence = _min_confidence; 
							}
						}
					}
				}
			}

			if(count != expectedCount) {
				continue;
			}
		}

		try {
			vector < vector<Center> > newCenters;
			for(size_t i = 0; i < curCenters.size(); i++) {
				if(curCenters[i].confidence < _min_confidence) { 
					continue; 
				}
				bool isNew = true;
				for(size_t j = 0; j < centers.size(); j++) {
					double dist = norm(centers[j][centers[j].size() / 2].location - curCenters[i].location);
					isNew = dist >= params.minDistBetweenBlobs && dist >= centers[j][centers[j].size() / 2].radius && dist >= curCenters[i].radius;
					if(!isNew) {
						centers[j].push_back(curCenters[i]);

						size_t k = centers[j].size() - 1;
						while(k > 0 && centers[j][k].radius < centers[j][k - 1].radius) {
							centers[j][k] = centers[j][k - 1];
							k--;
						}
						centers[j][k] = curCenters[i];

						break;
					}
				}
				if(isNew) {
					newCenters.push_back(vector<Center>(1, curCenters[i]));
				}
			}
			std::copy(newCenters.begin(), newCenters.end(), std::back_inserter(centers));
		}
		catch(Exception& ex) {
			std::cout << "BlobDetector:" << ' ' << ex.msg << std::endl;
			return;
		}
	}

	for(size_t i = 0; i < centers.size(); i++) {
		if(centers[i].size() < params.minRepeatability) {
			continue;
		}
		Point2d sumPoint(0, 0);
		double normalizer = 0;
		for(size_t j = 0; j < centers[i].size(); j++) {
			sumPoint += centers[i][j].confidence * centers[i][j].location;
			normalizer += centers[i][j].confidence;
		}
		sumPoint *= (1. / normalizer);
		KeyPoint kpt(sumPoint, (float)(centers[i][centers[i].size() / 2].radius));
		keypoints.push_back(kpt);
	}
}

bool findCirclesGridEx(Mat& image, vector<Point2f>& pointBuf, const Ptr<FeatureDetector> &blobDetector) { // on symmetric grid
	bool found = false;
	try {
		found = findCirclesGrid(image, g_boardSize, pointBuf, CALIB_CB_SYMMETRIC_GRID | CALIB_CB_CLUSTERING, blobDetector);
		if(!found) {
			found = findCirclesGrid(image, g_boardSize, pointBuf, CALIB_CB_SYMMETRIC_GRID, blobDetector);
		}
		if(found) {
			if(pointBuf[0].y > pointBuf[pointBuf.size() - g_boardSize.width].y) {
				std::reverse(pointBuf.begin(), pointBuf.end());
			}
		}
		if(!found) {
			cv::imshow("IMAGECalibr3", imread(IMG_DELETEDOCUMENT_H, cv::IMREAD_ANYCOLOR));
			while(ProcessWinMessages());
		}
	}
	catch(Exception& ex) {
		std::cout << "buildPointsFromImage:" << ' ' << ex.msg << std::endl;
		return false;
	}
	return found;
}

// approximate contour with accuracy proportional to the contour perimeter
bool approximateContourWithQuadrilateral(const std::vector<Point>& contour, std::vector<Point>& approx, double minArea, double maxArea) {
	double epsilon = 0.025;
	approxPolyDP((contour), approx, arcLength((contour), false) * epsilon, false);
	while(approx.size() > 4) {
		epsilon += 0.01;
		approxPolyDP((approx), approx, arcLength((approx), true) * epsilon, true);
	}
	bool rc = false; 

	// square-like contours should have 
	// - 4 vertices after approximation
	// - relatively large area (to filter out noisy contours)
	// - and be convex.
	// Note: absolute value of an area is used because
	// area may be positive or negative - in accordance with the contour orientation
	if(approx.size() == 4) {
		double area = std::fabs(contourArea((approx)));
		if(area > (minArea) && area < (maxArea)) {
			if(isContourConvex((approx))) {
				rc = true;
			}
		}
	}
	else {
		rc = false; 
	}
	return rc; 
}


// returns sequence of squares detected on the image.
void findSquares(const Mat& image, vector<vector<Point> >& squares, double maxArea) {
	squares.clear();

	vector<vector<Point> > contours;

	Mat gray0;
	image.convertTo(gray0, CV_8UC1);

	//uchar *val = (uchar*)gray0.data;
	//for(int j = 0, N = gray0.cols * gray0.rows; j < N; ++j, ++val) {
	//	*val = 255 - *val;
	//}

	int thresh = 255;

	for(int k = 0; k < gray0.rows; ++k) {
		for(int j = 0; j < gray0.cols; ++j) {
			auto p = gray0.at<unsigned char>(k, j);
			if(p > 20 && p < thresh) {
				thresh = p;
			}
		}
	}

	Mat gray1;
	normalize(gray0, gray1, 0, 255, NORM_MINMAX, CV_8UC1);

	// try several threshold levels
	const int NL = 11;
	Mat gray;
	for(int l = (int)((1.5 * thresh * NL) / 255.0 + 1.5); l < NL; ++l) { 
		gray = gray1 >= (l - 1) * 255 / NL; 

		contours.resize(0); 
		findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE); 

		vector<Point> approx; 
		for(size_t i = 0; i < contours.size(); i++) {
			approx.resize(0); 
			if(approximateContourWithQuadrilateral(contours[i], approx, 50, maxArea)) {
				squares.push_back(approx); 
			}
		}
	}
}

double FindMinDistanceBetweenNeighbours(vector<Point2f>& pointBuf, vector<vector<double>>& distancesMap) { // on symmetric grid
	distancesMap.resize(g_boardSize.height, vector<double>(g_boardSize.width, std::numeric_limits<double>::max()));
	int k = 0;
	int j = 0;
	double dist = std::numeric_limits<double>::max();
	for(auto& point : pointBuf) {
		int kk[4] = {-1, 0, 0, 1};
		int jj[4] = {0, -1, 1, 0};
		for(int n = 0; n < 4; ++n) {
			int kn = k + kk[n];
			int jn = j + jj[n];
			if(kn < 0 || kn >= g_boardSize.width) {
				continue;
			}
			if(jn < 0 || jn >= g_boardSize.height) {
				continue;
			}
			Point2f& p = pointBuf[jn * g_boardSize.width + kn];
			double d = sqrt(pow(point.x - p.x, 2) + pow(point.y - p.y, 2));
			if(d < dist) {
				dist = d;
			}
			if(d < distancesMap[j][k]) {
				distancesMap[j][k] = d; 
			}
		}
		if(++k >= g_boardSize.width) {
			k = 0;
			++j;
		}
	}
	return dist;
}

bool ExtractCornersOfRectangles(Mat& image/*in*/, vector<Point2f>& pointBuf/*in*/, vector<Point2f>& cornersBuf/*out*/) {
	vector<vector<double>> distancesMap;
	double dist = FindMinDistanceBetweenNeighbours(pointBuf, distancesMap);

	const int rows = image.rows;
	const float unitinpx = (float)0.5 * rows / (float)(g_boardSize.height + 1);


	// do pre-build homography for sorting the corners. 
	cornersBuf.resize(0);

	std::vector<Point2f> idealPoints;
	for(int i = 0; i < g_boardSize.height; ++i) {
		for(int j = 0; j < g_boardSize.width; ++j) {
			idealPoints.push_back(Point2f((float(j*g_pattern_distance) + 1) * unitinpx, (float(i*g_pattern_distance) + 1) * unitinpx));
		}
	}

	Mat H;
	try {
		H = findHomography(pointBuf, idealPoints, 0/*LMEDS*//*RANSAC*/, 4);
	}
	catch(Exception& ex) {
		std::cout << "DetectRectangularGridOfRectangles:" << ' ' << ex.msg << std::endl;
		return false; 
	}


	cornersBuf.resize(pointBuf.size() * 4); // 4 corners per one center


	Mat grayscaleImage;
	normalize(image, grayscaleImage, 0, 255, NORM_MINMAX, CV_8UC1);


	int currentPt = 0;
	int currentRw = 0;
	int currentCl = 0;

	for(auto& point : pointBuf) {
		currentCl = currentPt % g_boardSize.width;
		currentRw = currentPt / g_boardSize.width;
		++currentPt;

		dist = distancesMap[currentRw][currentCl];

		Rect box((int)(point.x - dist * 3.0 / 8.0), (int)(point.y - dist * 3.0 / 8.0), (int)(dist * 7.0 / 8.0), (int)(dist * 7.0 / 8.0));
		if(box.x < 0) {
			box.x = 0; 
		}
		if(box.y < 0) {
			box.y = 0;
		}
		Mat crop(grayscaleImage, box);

		vector<vector<Point> > squares;

		findSquares(crop, squares, (box.height - 4) * (box.width - 4));

		Mat croprgb; 
		cvtColor(crop, croprgb, CV_GRAY2RGB);

		auto croprgbShow = [&croprgb]() {
			double fx = 280.0 / croprgb.rows;
			Mat image0 = croprgb.clone();
			cv::resize(image0, croprgb = Mat(), cv::Size(0, 0), fx, fx, INTER_AREA);

			cv::imshow("IMAGECalibr3", croprgb);
			while(ProcessWinMessages());
		};

		if(squares.size() > 0) {
			for(auto& square : squares) {
				for(int j = 0; j < 4; ++j) {
					cv::line(croprgb, square[j], square[(j + 1) % 4], Scalar(255, 0, 0));
				}
			}


			vector<Point> combinedpoints(squares[0]);
			for(int j = 1; j < squares.size(); ++j) {
				combinedpoints.insert(combinedpoints.end(), squares[j].begin(), squares[j].end());
			}


			for(int k = 0; k < combinedpoints.size(); ++k) {
				circle(croprgb, combinedpoints[k], 1, Scalar(0, 0, 255), -1);
			}


			//std::vector<int> ylevels(combinedpoints.size(), -1);
			std::vector<int> ylevels;
			partitionEx(combinedpoints, ylevels, [dist](const Point &one, const Point &another) -> bool {
				return (pow(one.y - another.y, 2) + pow(one.x - another.x, 2)) < std::max((double)17/*4 pixels*/, std::pow(dist / 60.0, 2)); 
			});

			std::vector<std::vector<Point>> clusters;

			std::vector<int>::iterator it_lbl = ylevels.begin();
			for(auto& point : combinedpoints) {
				int cluster = *(it_lbl++);
				if((int)clusters.size() < (cluster + 1)) {
					clusters.resize(cluster + 1);
				}
				clusters[cluster].push_back(point);
			}

			std::vector<Point2f> centers; 
			int min_size = 1; 
			do {
				++min_size;
				centers.resize(0);

				for(auto& cluster : clusters) {
					Point2f center(0, 0);
					for(auto &point : cluster) {
						center.x += point.x;
						center.y += point.y;
					}
					center.x /= (float)cluster.size();
					center.y /= (float)cluster.size();

					if(cluster.size() >= min_size) {
						centers.push_back(center);
					}
				}
			} while(centers.size() > 4);


			for(int k = 0; k < centers.size(); ++k) {
				circle(croprgb, centers[k], 1, Scalar(0, 255, 0), -1);
			}

			croprgbShow();

			if(centers.size() != 4) {
				SleepEx(2000, TRUE);
				return false; 
			}

			Size winSize = Size(5, 5);
			Size zeroZone = Size(-1, -1);
			TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);

			try {
				cornerSubPix(crop, centers, winSize, zeroZone, criteria);
			}
			catch(Exception& ex) {
				std::cout << "DetectRectangularGridOfRectangles:" << ' ' << ex.msg << std::endl;
			}
			for(auto &center : centers) {
				center.x += box.x;
				center.y += box.y;
			}

			std::vector<Point2f> hull;
			convexHull(centers, hull, false/*will be clockwise because Y is pointing down*/, true/*do return points*/);

			vector<Point2f> hullMapped;
			perspectiveTransform(hull, hullMapped, H);

			//ylevels.resize(hullMapped.size(), -1);
			ylevels.resize(0);
			partitionEx(hullMapped, ylevels, [dist](const Point2f &one, const Point2f &another) -> bool {
				return std::abs(one.y - another.y) < (dist / 6);
			});

			int firstPoint = -1;
			int ylevel = -1;
			double ylevel_min = std::numeric_limits<double>::max();
			double xlevel_min = std::numeric_limits<double>::max();
			for(int k = 0; k < hullMapped.size(); ++k) {
				if(ylevel == ylevels[k]) {
					if(hullMapped[k].x < xlevel_min) {
						xlevel_min = hullMapped[k].x;
						firstPoint = k;
					}
				}
				else
				if(hullMapped[k].y < ylevel_min) {
					ylevel = ylevels[k];
					ylevel_min = hullMapped[k].y;
					xlevel_min = hullMapped[k].x;
					firstPoint = k;
				}
			}
			if(firstPoint > 0) {
				std::rotate(hull.begin(), hull.begin() + firstPoint, hull.end());
			}

			if(hull.size() == 4) {
				int c = g_boardSize.width * 2;

				int k = currentRw * 2;
				int j = currentCl * 2;

				cornersBuf[k * c + j] = hull[0];
				cornersBuf[k * c + j + 1] = hull[1];
				cornersBuf[(k + 1) * c + j + 1] = hull[2];
				cornersBuf[(k + 1) * c + j] = hull[3];
			}
		}
		else {
			croprgbShow();
			SleepEx(2000, TRUE);
		}
	}

	// verify the order of corners. it has to be a reqular grid. 

	vector<Point2f> cornersMapped;
	perspectiveTransform(cornersBuf, cornersMapped, H);

	std::vector<int> ylevels(cornersMapped.size(), -1);
	partitionEx(cornersMapped, ylevels, [dist](const Point2f &one, const Point2f &another) -> bool {
		return std::abs(one.y - another.y) < (dist / 6);
	});

	std::vector<int>::iterator it_lbl = ylevels.begin();

	for(int k = 0; k < g_boardQuadSize.height; ++k) {
		for(int j = 0; j < g_boardQuadSize.width; ++j) {
			int cluster = *(it_lbl++);
			if(cluster != k) {
				return false; 
			}
		}
	}

	return true; 
}

bool ExtractCornersOfChessPattern(Mat& imageInp, vector<Point2f>& pointBuf, const Ptr<FeatureDetector>& detector) {
	vector<Point2f> edgesBuf;

	vector<Point2f> approx2fminQuad; // Qaudrilateral delimiting the area of the image that contains corners. 
	Mat H; // Homography that transforms quadrilateral to rectangle (not rotated), enabling therefore the sorting of corners. 
	vector<Point2f> approx2fminRectMapped(4); // Qadrilateral tranformed to rectangle.
	Rect approxBoundingRectMapped;

	Mat image = imageInp.clone();

	double fy = 1.0; 

	//if(image.rows > 1600) {
	//	double fy = 1600.0 / image.rows; 
	//	cv::resize(image, image, cv::Size(0, 0), fy, fy, INTER_AREA);
	//}

	auto image_type = image.type();

	if (image_type == CV_8UC3) {
		//double black_color[3] = { 117, 110, 86 };// { 109, 110, 87 };
		//StandardizeImage_HSV_Likeness(image, black_color);

		double mean_data[3] = { 116.83199, 110.26242, 86.00596 };// { 143.05750, 119.46569, 68.23818 };// { 158.8078, 147.9445, 125.1749 };
		double invCovar_data[9] = { 0.009066458, -0.002604613, -0.007678366, -0.002604613, 0.020680217, -0.014549229, -0.007678366, -0.014549229, 0.023863367 };// { 0.0102539613, -0.01036747, 0.0004458039, -0.0103674702, 0.03199960, -0.0149542107, 0.0004458039, -0.01495421, 0.0114805264 };// { 0.006623272, -0.005202919, -0.001526119, -0.005202919, 0.009449343, -0.003572090, -0.001526119, -0.003572090, 0.005443892 };
		double invCholesky_data[9] = { 0.04583349, 0.00000000, 0.0000000, -0.06704573, 0.10867251, 0.0000000, -0.04970533, -0.09418335, 0.1544777 };// { 0.050860724, 0.0000000, 0.0000000, -0.087463346, 0.1118958, 0.0000000, 0.004160667, -0.1395670, 0.1071472 };// { 0.02789284, 0.00000000, 0.00000000, -0.07360323, 0.08429391, 0.00000000, -0.02068396, -0.04841362, 0.07378274 };

		cv:Mat mean = cv::Mat(1, 3, CV_64F, mean_data);
		cv::Mat invCovar = cv::Mat(3, 3, CV_64F, invCovar_data);
		cv::Mat invCholesky = cv::Mat(3, 3, CV_64F, invCholesky_data);

		cv::Mat stdDev;
		cv::Mat factorLoadings;

		StandardizeImage_Likeness(image, mean, stdDev, factorLoadings, invCovar, invCholesky);

		image = mat_loginvert2word(image);
		image = mat_invert2word(image);
		cv::normalize(image.clone(), image, 0, (size_t)256 * g_bytedepth_scalefactor, NORM_MINMAX, CV_16UC1, Mat());
		image_type = image.type();
	}

	//bool found = findCirclesGrid(image, g_boardChessSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID/* | CALIB_CB_CLUSTERING*/, detector);

	std::vector<KeyPoint> keyPoints;
	detector->detect(image, keyPoints);

	pointBuf.reserve(keyPoints.size());
	pointBuf.resize(0);

	for (auto& keyPoint : keyPoints) {
		pointBuf.push_back(keyPoint.pt);
	}

	bool found = pointBuf.size() == g_boardChessSize.width * g_boardChessSize.height;

	// build min. enclosing quadrilateral
	if(found) {
		std::vector<Point> blobs(pointBuf.size());
		for(int k = 0; k < pointBuf.size(); ++k) {
			blobs[k] = Point2i((int)(pointBuf[k].x + 0.5), (int)(pointBuf[k].y + 0.5));
		}

		std::vector<Point> hull;
		convexHull(blobs, hull, true/*will be counter-clockwise because Y is pointing down*/, true/*do return points*/);

		vector<Point> approx;
		approxPolyDP(Mat(hull), approx, arcLength(Mat(hull), true)*0.02, true);

		if(approx.size() >= 6) {
			// Approximate the outer hull with min. enclosing quadrilateral. 

			// 1. Order the hull counterclockwise starting with the most remote point, because
			// that is how the pattern of ideal points is preset. 
			// 2. Use ideal points to build homography. 
			// 3. Use homography to build the min. enclosing rectangle. 
			// 4. Transform back to create the min. enclosing quadrilateral. 

			// 1. The first point must be the most remote point. 
			// Remove (in iterations) two most proximate points until two or one is left. 
			vector<int> approx_idx(approx.size());
			for(int j = 0; j < approx_idx.size(); ++j) {
				approx_idx[j] = j;
			}
			while(approx_idx.size() > 2) {
				double min_dist = std::numeric_limits<double>::max();
				int min_idx[2] = {0, 1};
				for(int j = 0, k = 1; j < approx_idx.size(); ++j, ++k) {
					if(k >= approx_idx.size()) {
						k = 0;
					}
					double dist = cv::norm(approx[approx_idx[j]] - approx[approx_idx[k]]);
					if(dist < min_dist) {
						min_dist = dist;
						min_idx[0] = approx_idx[j];
						min_idx[1] = approx_idx[k];
					}
				}
				approx_idx.erase(std::remove_if(approx_idx.begin(), approx_idx.end(), [&min_idx](const int idx) -> bool {
					return idx == min_idx[0] || idx == min_idx[1];
				}), approx_idx.end());
			}
			if(approx_idx.size() == 2) {
				if(approx[approx_idx[0]].y > approx[approx_idx[1]].y) {
					approx_idx[0] = approx_idx[1];
				}
			}

			int idx = approx_idx[0];

			// points need to be reversed if the next point is also a remote point. 

			if(cv::norm(approx[(idx + 1) % approx.size()] - approx[(idx + 2) % approx.size()]) > cv::norm(approx[(idx - 1) % approx.size()] - approx[(idx - 2) % approx.size()])) {
				std::reverse(approx.begin(), approx.end());
				idx = (int)approx.size() - 1 - idx;
			}

			if(idx > 0) {
				std::rotate(approx.begin(), approx.begin() + idx, approx.end());
			}




			vector<Point2f> approx2f(approx.size());
			for(int j = 0; j < 6; ++j) {
				approx2f[j] = Point2f((float)approx[j].x, (float)approx[j].y);
			}



			// 2. Use ideal points to build homography. 

			const int rows = image.rows;
			const float unitinpx = (float)0.4 * rows / (float)(g_boardChessSize.height);

			std::vector<Point2f> ideal_approx2f(6); // counterclockwise starting from top right.
			if (g_boardChessSize.height == 7) {
				ideal_approx2f[0] = Point2f((0 + 5) * unitinpx, (0 + 5) * unitinpx);
				ideal_approx2f[1] = Point2f((6 + 5) * unitinpx, (0 + 5) * unitinpx);
				ideal_approx2f[2] = Point2f((7 + 5) * unitinpx, (1 + 5) * unitinpx);
				ideal_approx2f[3] = Point2f((7 + 5) * unitinpx, (6 + 5) * unitinpx);
				ideal_approx2f[4] = Point2f((6 + 5) * unitinpx, (6 + 5) * unitinpx);
				ideal_approx2f[5] = Point2f((0 + 5) * unitinpx, (6 + 5) * unitinpx);
			}
			else {
				ideal_approx2f[0] = Point2f((0 + 5) * unitinpx, (0 + 5) * unitinpx);
				ideal_approx2f[1] = Point2f((6 + 5) * unitinpx, (0 + 5) * unitinpx);
				ideal_approx2f[2] = Point2f((7 + 5) * unitinpx, (1 + 5) * unitinpx);
				ideal_approx2f[3] = Point2f((7 + 5) * unitinpx, (7 + 5) * unitinpx);
				ideal_approx2f[4] = Point2f((1 + 5) * unitinpx, (7 + 5) * unitinpx);
				ideal_approx2f[5] = Point2f((0 + 5) * unitinpx, (6 + 5) * unitinpx);
			}
			if(approx2f[0].x > approx2f[1].x) {
				float max_x = 0;
				for_each(ideal_approx2f.begin(), ideal_approx2f.end(), [&max_x](Point2f& point) {
					if(point.x > max_x) {
						max_x = point.x;
					}
				});
				for_each(ideal_approx2f.begin(), ideal_approx2f.end(), [max_x](Point2f& point) {
					point.x = max_x + 5 - point.x;
				});
			}


			try {
				H = findHomography(approx2f, ideal_approx2f, 0/*LMEDS*//*RANSAC*/, 4);
			}
			catch(Exception& ex) {
				std::cout << "ExtractCornersOfChessPattern:" << ' ' << ex.msg << std::endl;
			}


			// 3. Use homography to build the min. enclosing rectangle. 

			vector<Point2f> approx2fMapped;
			perspectiveTransform(approx2f, approx2fMapped, H);

			RotatedRect minRect = minAreaRect(Mat(approx2fMapped));
			minRect.points(&approx2fminRectMapped[0]);

			approxBoundingRectMapped = boundingRect(Mat(approx2fMapped));


			// 4. Transform back to create the min. enclosing quadrilateral. 

			approx2fminQuad.resize(4);
			perspectiveTransform(approx2fminRectMapped, approx2fminQuad, H.inv());


			// 5. Visualize the centers and quadrilateral. 

			Mat image0;
			normalize(image, image0, 0, 255 * 256, NORM_MINMAX);
			Mat image1;
			cvtColor(image0, image1, COLOR_GRAY2RGB);

			for(int j = 0; j < 4; ++j) {
				cv::line(image1, approx2fminQuad[j], approx2fminQuad[(j + 1) % 4], Scalar(255 * 256, 0, 0));
			}
			for(int k = 0; k < blobs.size(); ++k) {
				circle(image1, blobs[k], 3, Scalar(0, 0, 255 * 256), -1);
			}
			const Point *p = &approx[0];
			int n = (int)approx.size();
			cv::polylines(image1, &p, &n, 1, true, Scalar(0, 255 * 256, 0), 1, LINE_AA);

			double fx = 280.0 / image1.rows;
			cv::resize(image1, image0 = Mat(), cv::Size(0, 0), fx, fx, INTER_AREA);

			cv::imshow("IMAGECalibr3", image0);
			while(ProcessWinMessages());

			vector<int> compression_params;
			compression_params.push_back(IMWRITE_PNG_COMPRESSION); // Mar.4 2015.
			compression_params.push_back(0);

			cv::imwrite(std::string(g_path_calib_images_dir) + "ST1-ChessPattern-enclose.png", image1, compression_params);
			while (ProcessWinMessages());
		}
	}
	else {
		cv::imshow("IMAGECalibr3", imread(IMG_DELETEDOCUMENT_H, cv::IMREAD_ANYCOLOR));
		while(ProcessWinMessages());
	}

	if(approx2fminQuad.size()) {
		//imageInp.convertTo(image, CV_16UC1);
		image = imageInp.clone();
		double color2gray[3] = { 0.299, 0.587, 0.114 };
		//double color2gray[3] = { 0.114, 0.587, 0.299 };
		ConvertColoredImage2Mono(image, color2gray, [](double ch) {
			return std::min(ch * 256, 256.0 * 256.0);
		});
		Mat imageMapped;
		warpPerspective(image, imageMapped, H, image.size()/*, INTER_CUBIC*/);
		Mat crop(imageMapped, approxBoundingRectMapped);
		Mat grayscale;
		normalize(crop, grayscale, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
		while (ProcessWinMessages());


		vector<vector<Point2f>> edgesMappedBuf;
		int ylevel_maxwidth = 0;

		for(int level = 120; level < 200; level += 10) {

			Mat binImage;
			threshold(grayscale, binImage, level, 255, THRESH_BINARY);
			Mat erodedImage;
			cv::erode(binImage, erodedImage, getStructuringElement(MORPH_ELLIPSE, Size(5 * 1 + 1, 5 * 1 + 1)));
			Mat corners;
			int blockSize = grayscale.cols / 50;
			cornerHarris(erodedImage/*grayscale*/, corners, std::max(blockSize, 3), 3/*5*/, 0.1/*0.17*/, BORDER_DEFAULT);
			Mat normalizedCorners;
			normalize(corners, normalizedCorners, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
			vector<Point2f> cornersBufMapped;
			for(int k = 0; k < normalizedCorners.rows; ++k) {
				for(int j = 0; j < normalizedCorners.cols; ++j) {
					auto p = normalizedCorners.at<unsigned char>(k, j);
					if(p > 180) {
						cornersBufMapped.push_back(Point2f((float)j, (float)k));
					}
				}
			}


			std::vector<int> dist_levels(cornersBufMapped.size(), -1);
			int	clusters_count = std::numeric_limits<int>::max();
			const int centroids_count = g_boardChessCornersSize.width * g_boardChessCornersSize.height; 
			int dist = grayscale.cols / (g_boardChessCornersSize.width * 2);
			while(clusters_count > centroids_count) {
				const int squared_dist = dist * dist + 1;
				partitionEx(cornersBufMapped, dist_levels, [squared_dist](const Point2f &one, const Point2f &another) -> bool {
					return (pow(one.y - another.y, 2) + pow(one.x - another.x, 2)) < squared_dist;
				});
				clusters_count = *(std::max_element(dist_levels.begin(), dist_levels.end())) + 1;
				dist += 2;
			}
			if(clusters_count != centroids_count) {
				continue;
			}
			if(ylevel_maxwidth < dist) {
				ylevel_maxwidth = dist;
			}


			vector<Point2f> edgesMapped;

			Mat kmeans_centers;
			Mat cornersMat;
			try {
				for(auto& corner : cornersBufMapped) {
					cornersMat.push_back(corner);
				}
				kmeans(cornersMat, centroids_count, dist_levels,
					TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 50, 0.1), 3/*random number generator state*/, KMEANS_USE_INITIAL_LABELS, kmeans_centers);

				for(int k = 0; k < kmeans_centers.rows; ++k) {
					edgesMapped.push_back(Point2f(kmeans_centers.at<float>(k, 0), kmeans_centers.at<float>(k, 1)));
				}

				dist_levels.resize(cornersBufMapped.size(), -1);
				const int squared_dist = dist * dist;
				partitionEx(edgesMapped, dist_levels, [squared_dist](const Point2f &one, const Point2f &another) -> bool {
					return (pow(one.y - another.y, 2) + pow(one.x - another.x, 2)) < squared_dist;
				});
				clusters_count = *(std::max_element(dist_levels.begin(), dist_levels.end())) + 1;

				if(clusters_count == centroids_count) {
					edgesMappedBuf.push_back(edgesMapped);
				}
			}
			catch(Exception& ex) {
				std::cout << "ExtractCornersOfChessPattern:" << ' ' << ex.msg << std::endl;
			}



			Mat image1;
			cvtColor(/*binImage*/grayscale, image1, COLOR_GRAY2RGB);
			for(auto& point : cornersBufMapped) {
				circle(image1, Point2i((int)point.x, (int)point.y), 1, Scalar(0, 0, 255), -1);
			}
			for(int k = 0; k < edgesMapped.size(); ++k) {
				circle(image1, Point2i((int)edgesMapped[k].x, (int)edgesMapped[k].y), 3, Scalar(0, 255, 0), -1);
			}

			cv::imshow("IMAGECalibr3", image1);
			while(ProcessWinMessages());

			vector<int> compression_params;
			compression_params.push_back(IMWRITE_PNG_COMPRESSION); // Mar.4 2015.
			compression_params.push_back(0);

			std::string name = std::string(g_path_calib_images_dir) + "ST2-ChessPattern-edgesbylevel" + std::to_string(level) + ".png"; 
			cv::imwrite(name, image1, compression_params);
		}

		vector<Point2f> edgesMapped(edgesMappedBuf.size()? g_boardChessCornersSize.width * g_boardChessCornersSize.height: 0);
		for(auto& edges : edgesMappedBuf) {
			std::sort(edges.begin(), edges.end(), [ylevel_maxwidth](const Point2f &one, const Point2f &another) -> bool {
				return one.y < (another.y - ylevel_maxwidth) || (std::abs(one.y - another.y) < ylevel_maxwidth && one.x < another.x);
			});
			vector<Point2f>::iterator it = edgesMapped.begin();
			for_each(edges.begin(), edges.end(), [&it](const Point2f &point) -> void {
				(*it).x += point.x;
				(*it).y += point.y;
				++it;
			});
		}
		for(auto& point : edgesMapped) {
			point *= 1.0 / (double)edgesMappedBuf.size();
		}

		try {
			if(edgesMapped.size() && ylevel_maxwidth) {
				cornerSubPix(grayscale, edgesMapped, Size(ylevel_maxwidth / 3, ylevel_maxwidth / 3)/*window size*/, Size(-1, -1)/*no zero zone*/, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001));
			}
		}
		catch(Exception& ex) {
			std::cout << "ExtractCornersOfChessPattern:" << ' ' << ex.msg << std::endl;
			edgesMapped.resize(0); 
		}


		Mat image1;
		cvtColor(/*binImage*/grayscale, image1, COLOR_GRAY2RGB);
		for(int k = 0; k < edgesMapped.size(); ++k) {
			circle(image1, Point2i((int)edgesMapped[k].x, (int)edgesMapped[k].y), 3, Scalar(0, 0, 255), -1);
		}

		cv::imshow("IMAGECalibr3", image1);
		while(ProcessWinMessages());

		vector<int> compression_params;
		compression_params.push_back(IMWRITE_PNG_COMPRESSION); // Mar.4 2015.
		compression_params.push_back(0);

		cv::imwrite(std::string(g_path_calib_images_dir) + "ST3-ChessPattern-edgesMapped.png", image1, compression_params);

		//double fx = 270.0 / image1.rows;
		//cv::resize(image1, image1, cv::Size(0, 0), fx, fx, INTER_AREA);

		//cv::imshow("IMAGECalibr3", image1);
		//while(ProcessWinMessages());



		if(edgesMapped.size() == (g_boardChessCornersSize.width * g_boardChessCornersSize.height)) {
			for(auto& point : edgesMapped) {
				point.x += approxBoundingRectMapped.x;
				point.y += approxBoundingRectMapped.y;
			}
			edgesBuf.resize(edgesMapped.size());
			perspectiveTransform(edgesMapped, edgesBuf, H.inv());
		}
		else {
			static int x = 0; 
			++x; 
		}
	}

	if(edgesBuf.size()) {
		Mat grayscale;
		normalize(image, grayscale, 0, 255, NORM_MINMAX, CV_8UC1, Mat());

		try {
			cornerSubPix(grayscale, edgesBuf, Size(5, 5)/*window size*/, Size(-1, -1)/*no zero zone*/, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001));
		}
		catch(Exception& ex) {
			std::cout << "ExtractCornersOfChessPattern:" << ' ' << ex.msg << std::endl;
			edgesBuf.resize(0); 
		}
	}

	if(fy > 1) {
		for(auto& point : edgesBuf) {
			point *= fy;
		}
	}

	pointBuf.swap(edgesBuf); 

	return pointBuf.size() == (g_boardChessCornersSize.width * g_boardChessCornersSize.height); 
}

bool DetectGrid(Mat& image, vector<Point2f>& pointBuf, const Ptr<FeatureDetector> &blobDetector, int gridType/*0 - circles, 1 - regular grid squares, 2 - chess grid of squares*/) {
	pointBuf.clear();
	bool found = false;
	if(image.data) {
		if(gridType == 2) {
			found = ExtractCornersOfChessPattern(image, pointBuf, blobDetector);
		}
		else {
			found = findCirclesGridEx(image, pointBuf, blobDetector);
		}
		if(gridType == 1) { // regular grid squares
			if(found) {
				vector<Point2f> cornersBuf;
				found = ExtractCornersOfRectangles(image/*in*/, pointBuf/*in*/, cornersBuf/*out*/);
				if(found) {
					pointBuf.swap(cornersBuf);
				}
			}
		}
	}
	return found;
}



void ProjectIdeal(vector<Point2f>& pointBuf, vector<Point2f>& idealMapped, const cv::Size &boardSize, bool byrow) { 
	int nrows = byrow? boardSize.height: boardSize.width; 
	int iters = byrow? boardSize.width: boardSize.height; 
	int kstep = byrow? iters: 1;
	int jstep = byrow? 1: nrows;

	vector<Point2f> P(idealMapped.size());

	Mat_<double> A(nrows, 4);
	Mat_<double> X(nrows, 1);
	Mat_<double> Y(nrows, 1);

	for(int j = 0; j < iters; ++j) { 
		for(int k = 0; k < nrows; ++k) {
			A(k, 0) = 1; 
			A(k, 1) = pointBuf[j * jstep + k * kstep].x;
			A(k, 2) = pointBuf[j * jstep + k * kstep].y;
			A(k, 3) = std::sqrt(A(k, 1) * A(k, 2)); // 2016-05-12
			X(k, 0) = idealMapped[j * jstep + k * kstep].x;
			Y(k, 0) = idealMapped[j * jstep + k * kstep].y;
		}

		Mat_<double> AT(4, nrows);
		transpose(A, AT);

		Mat_<double> AI(A.cols, A.rows);
		invert(AT * A, AI); 

		Mat_<double> X_Hat = A * AI * AT * X;
		Mat_<double> Y_Hat = A * AI * AT * Y;

		for(int k = 0; k < nrows; ++k) {
			idealMapped[j * jstep + k * kstep].x = (float)X_Hat(k, 0);
			idealMapped[j * jstep + k * kstep].y = (float)Y_Hat(k, 0);
		}
	}
} 

//findHomography
//transform the image to coord.system of the pattern. 
//evaluate the circl.centers in the transformed image and transform them back to the original system. 
bool optimizePointsFromImage(Mat& image, vector<Point2f>& pointBuf, const Ptr<FeatureDetector> &blobDetector, int warped_iterations = 0, bool use_idealMapped = true) {
	const static int warped_threshold = 140;

	int gridTtype = 0; /*0 - circles, 1 - regular grid squares, 2 - chess grid of squares*/

	double pattern_distance = g_pattern_distance;
	cv::Size boardSize; 
	if(pointBuf.size() == (g_boardSize.height * g_boardSize.width)) { 
		boardSize = g_boardSize; 
		gridTtype = 0; 
	}
	else
	if(pointBuf.size() == (g_boardQuadSize.height * g_boardQuadSize.width)) {
		boardSize = g_boardQuadSize;
		pattern_distance /= 2.0;
		gridTtype = 1; 
	}
	else 
	if(pointBuf.size() == (g_boardChessCornersSize.height * g_boardChessCornersSize.width)) {
		boardSize = g_boardChessCornersSize;
		pattern_distance = pattern_distance /= 2.0; //3.0 + 2.5 / 8.0;
		gridTtype = 2;
	}

	bool useWarped = warped_iterations > 0;
	size_t number_of_iterations = useWarped? warped_iterations: 1;

	std::vector<Point2f> idealPoints;
	const int rows = image.rows;
	const float unitinpx = (float)0.5 * rows / (float)(boardSize.height + 1);
	for(int i = 0; i < boardSize.height; ++i) {
		for(int j = 0; j < boardSize.width; ++j) {
			idealPoints.push_back(Point2f((float(j*pattern_distance) + 1) * unitinpx, (float(i*pattern_distance) + 1) * unitinpx));
		}
	}

	Mat warped_image;
	for(size_t j = 0; j < number_of_iterations; ++j) {
		Mat H;
		try {
			H = findHomography(pointBuf, idealPoints, 0/*LMEDS*//*RANSAC*/, 4);
		}
		catch(Exception& ex) {
			std::cout << "optimizePointsFromImage:" << ' ' << ex.msg << std::endl;
			return false;
		}
		try {
			warpPerspective(image, warped_image = Mat(), H, image.size()/*, INTER_CUBIC*/);
		}
		catch(Exception& ex) {
			std::cout << "optimizePointsFromImage:" << ' ' << ex.msg << std::endl;
			return false;
		}


		if(j == (number_of_iterations - 1)) { // last iteration
			vector<Point2f> idealMapped;
			perspectiveTransform(idealPoints, idealMapped, H.inv());
			double err_sum[2] = {0, 0};
			double err_sum_sqr[2] = {0, 0};
			double err_sum_cube[2] = {0, 0};
			double err_sum_quad[2] = {0, 0};
			vector<Point2f>::const_iterator it = pointBuf.begin();
			for(auto& idealPoint : idealMapped) {
				double err[2] = {idealPoint.x - (*it).x, idealPoint.y - (*it).y};
				for(int k = 0; k < 2; ++k) {
					err_sum[k] += err[k];
				}
				if(g_configuration._file_log == 2) {
					std::ostringstream ostr;
					if(it != pointBuf.begin()) {
						ostr << ',';
					}
					ostr << idealPoint.x << ',' << (*it).x << ',' << idealPoint.y << ',' << (*it).y;
					VS_FileLog(ostr.str());
				}
				++it;
			}
			double mean[2] = {0, 0};
			for(int k = 0; k < 2; ++k) {
				mean[k] = err_sum[k] / (int)idealMapped.size();
			}
			it = pointBuf.begin();
			for(auto& idealPoint : idealMapped) {
				double err[2] = {idealPoint.x - (*it).x, idealPoint.y - (*it).y};
				for(int k = 0; k < 2; ++k) {
					err[k] -= mean[k];
					err_sum_sqr[k] += pow(err[k], 2);
					err_sum_cube[k] += pow(err[k], 3);
					err_sum_quad[k] += pow(err[k], 4);
				}
				++it;
			}
			double sd[2] = {0, 0};
			double skewness[2] = {0, 0};
			double kurtosis[2] = {0, 0};
			for(int k = 0; k < 2; ++k) {
				int n = (int)idealMapped.size();
				sd[k] = sqrt(err_sum_sqr[k] / (n - 1));
				skewness[k] = (err_sum_cube[k] / pow(sd[k], 3)) * (n) / ((n - 1) * (n - 2));
				kurtosis[k] = (err_sum_quad[k] / pow(sd[k], 4)) * (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) - 3;
				if(g_configuration._file_log == 2) {
					std::ostringstream ostr;
					ostr << ',' << mean[k] << ',' << sd[k] << ',' << kurtosis[k] << ',' << skewness[k];
					VS_FileLog(ostr.str());
				}
			}
			bool is_normal = false;
			if(std::max(abs(kurtosis[0]), abs(kurtosis[1])) < 1) {
				if(std::max(abs(skewness[0]), abs(skewness[1])) < 1) {
					if(abs(mean[0]) < (0.7 * sd[0]) && abs(mean[1]) < (0.7 * sd[1])) {
						is_normal = true;
					}
				}
			}
			if(g_configuration._file_log == 2) {
				VS_FileLog("\r\n");
			}
			std::cout << std::fixed << std::setw(7) << std::setprecision(4) << (char)(is_normal? 'N': '?') << ' ' << "err {" << abs(mean[0]) << ',' << abs(mean[1]) << '}' << "; sd {" << sd[0] << ',' << sd[1] << '}' << "; kurt {" << abs(kurtosis[0]) << ',' << abs(kurtosis[1]) << '}' << "; skew {" << abs(skewness[0]) << ',' << abs(skewness[1]) << '}' << std::endl;
			if(j > 2 && use_idealMapped) {
				if(is_normal) {
					ProjectIdeal(pointBuf, idealMapped, boardSize, false);
					ProjectIdeal(pointBuf, idealMapped, boardSize, true);
					pointBuf.swap(idealMapped);
				}
			}
		}
		else 
		if(useWarped) {
			try {
				vector<Point2f> warped_points;
				if(DetectGrid(warped_image, warped_points, blobDetector, gridTtype)) {
					perspectiveTransform(warped_points, pointBuf, H.inv());
				}
				else {
					break;
				}
			}
			catch(Exception& ex) {
				std::cout << "optimizePointsFromImage:" << ' ' << ex.msg << std::endl;
				return false;
			}
		}
	}
	//
	// Previously there was an issue with displaying the image of the homography because of built-in height auto-alignment by KFrame::AlignChildRectangles().
	// It has been fixed by using isAbsolute=1 on the definition of Canvas, so KFrame::AlignChildRectangles() is not auto-aligning. 
	//
	if(useWarped) {
		unsigned short *val = (unsigned short*)warped_image.data;
		for(int j = 0, N = warped_image.cols * warped_image.rows; j < N; ++j, ++val) {
			if(*val < ((warped_threshold + 60) * g_bytedepth_scalefactor)) {
				*val = 0;
			}
		}
	}
	try {
		cv::imshow("IMAGECalibr3", warped_image *= g_bytedepth_scalefactor); // Mar.4 2015.
	}
	catch(Exception& ex) {
		std::cout << "optimizePointsFromImage:" << ' ' << ex.msg << std::endl;
	}
	return true; 
}

bool buildPointsFromImage(Mat& image, vector<Point2f>& pointBuf, SImageAcquisitionCtl& ctl, double min_confidence = 0.0, size_t min_repeatability = 3, int warped_iterations = 0, bool use_idealMapped = false) {
	//Ptr<FeatureDetector> blobDetector = ClassBlobDetector::create(ClassBlobDetector(min_confidence, min_repeatability, 120, ctl._pattern_is_whiteOnBlack, ctl._pattern_is_chessBoard).params);
	Ptr<FeatureDetector> blobDetector = new ClassBlobDetector(min_confidence, min_repeatability, 40, ctl._pattern_is_whiteOnBlack, ctl._pattern_is_chessBoard);
	pointBuf.clear();
	bool found = false;
	if(image.data) { 
		g_imageSize = image.size();

		found = DetectGrid(image, pointBuf, blobDetector, ctl._pattern_is_chessBoard? 2: ctl._pattern_is_gridOfSquares? 1: 0);
		if(found) {
			if(warped_iterations || use_idealMapped) {
				optimizePointsFromImage(image, pointBuf, blobDetector, ctl._pattern_is_gridOfSquares? 0: warped_iterations, use_idealMapped);
			}
		}
	}

	return found;
}


double calc_betweenimages_rmse(vector<Point2f>& image1, vector<Point2f>& image2) {
	double rmse = 0;
	size_t min_size = std::min(image1.size(), image2.size());
	bool strict = (imagePoints_left.size() + imagePoints_right.size()) < 5/*(g_min_images)*/;
	for(int j = 0; j < (int)min_size; ++j) {
		double e = sqrt(pow(image1[j].x - image2[j].x, 2) + pow(image1[j].y - image2[j].y, 2));
		if(strict) {
			rmse += e;
		}
		else
		if(e > rmse) {
			rmse = e; 
		} 
	}
	if(strict) {
		rmse /= min_size;
	}
	return rmse; 
}



bool EvaluateImagePoints(Mat& cv_image, vector<vector<Point2f>>& imagePoints, SImageAcquisitionCtl& ctl, double min_confidence = 0, size_t min_repeatability = 5) {
	bool is_ok = false;

	int x = (int)imagePoints.size() - 1;
	if(buildPointsFromImage(cv_image, imagePoints[x], ctl, min_confidence, min_repeatability)) {
		is_ok = true;
		if(x > 0) {
			for(int k = x - 1; k >= 0 && is_ok; --k) {
				double rmse = calc_betweenimages_rmse(imagePoints[k], imagePoints[x]);
				if(rmse < g_aposteriory_minsdistance) {
					is_ok = false;
				}
			}
		}
	}

	return is_ok;
}


double CalibrateSingleCamera(vector<vector<Point2f>>& imagePoints, Mat& cameraMatrix, Mat& distortionCoeffs, int flag = cv::CALIB_RATIONAL_MODEL, bool reprojectPoints = false) {
	double pattern_distance = g_pattern_distance;
	cv::Size boardSize;
	if(imagePoints[0].size() == (g_boardSize.height * g_boardSize.width)) {
		boardSize = g_boardSize;
	}
	else
	if(imagePoints[0].size() == (g_boardQuadSize.height * g_boardQuadSize.width)) {
		boardSize = g_boardQuadSize;
		pattern_distance /= 2.0;
	}
	else
	if(imagePoints[0].size() == (g_boardChessCornersSize.height * g_boardChessCornersSize.width)) {
		boardSize = g_boardChessCornersSize;
		pattern_distance = pattern_distance /= 2.0; //3.0 + 2.5 / 8.0;
	}

	vector<vector<Point3f> > objectPoints(1);
	for(int i = 0; i < boardSize.height; ++i) {
		for(int j = 0; j < boardSize.width; ++j) {
			objectPoints[0].push_back(Point3f(float(j*pattern_distance), float(i*pattern_distance), 0));
		}
	}
	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	cameraMatrix = Mat::eye(3, 3, CV_64F);
	cameraMatrix.at<double>(0, 0) = 1.0;

	distortionCoeffs = Mat::zeros(8, 1, CV_64F); // distortion coefficient matrix. Initialize with zero. 

	vector<Mat> rvecs;
	vector<Mat> tvecs;


	double rms = calibrateCamera(objectPoints, imagePoints, g_imageSize, cameraMatrix, distortionCoeffs, rvecs, tvecs, flag);

	//for(size_t j = 0; j < objectPoints.size(); ++j) {
	//	projectPoints(objectPoints[j], rvecs[j], tvecs[j], cameraMatrix, distortionCoeffs, imagePoints[j]); 
	//}

	g_objectPoints = objectPoints;

	return rms;
}


bool ReEvaluateUndistortImagePoints(vector<Mat>& imageRaw, vector<vector<Point2f>>& imagePoints, Mat& cameraMatrix, Mat& distortionCoeffs, SImageAcquisitionCtl& ctl, std::string& cv_window, vector<vector<Point2f>> *imagePoints_paired = 0, vector<Mat> *imageRaw_paired = 0) {
	Mat map[2];

	cv::initUndistortRectifyMap(cameraMatrix, distortionCoeffs, cv::Mat(), cv::getOptimalNewCameraMatrix(cameraMatrix, distortionCoeffs, g_imageSize, 0), g_imageSize, CV_32FC2, map[0], map[1]);

	for(auto& image : imageRaw) {
		Mat undistorted;
		remap(image, undistorted, map[0], map[1], INTER_CUBIC/*INTER_LINEAR*/, BORDER_CONSTANT);
		image = undistorted;

		while(ProcessWinMessages());
	}
	for(int j = 0; j < imagePoints.size(); ++j) {
		while(ProcessWinMessages());

		if(!g_configuration._calib_use_homography || !buildPointsFromImage(imageRaw[j], imagePoints[j], ctl, g_configuration._calib_min_confidence, 5, 2, true)) {
			while(ProcessWinMessages());

			if(!buildPointsFromImage(imageRaw[j], imagePoints[j], ctl, g_configuration._calib_min_confidence, 5)) {
				std::cout << "image number " << j << " has failed" << std::endl;
				imagePoints.erase(imagePoints.begin() + j);
				imageRaw.erase(imageRaw.begin() + j);
				if(imagePoints_paired) {
					imagePoints_paired->erase(imagePoints_paired->begin() + j);
				}
				if(imageRaw_paired) {
					imageRaw_paired->erase(imageRaw_paired->begin() + j);
				}
				--j;
			}
		}
		else {
			DrawImageAndBoard(std::to_string(j + 1), cv_window, imageRaw[j], imagePoints[j]);
		}
	}

	while(ProcessWinMessages());
	return imagePoints.size() >= g_min_images;
}

void SampleImagepoints(vector<vector<Point2f>>& imagePoints_src, vector<vector<Point2f>>& imagePoints_dst, vector<vector<Point2f>> *imagePoints_src2 = 0, vector<vector<Point2f>> *imagePoints_dst2 = 0) {
	std::vector<size_t> x;
	x.reserve(g_min_images);
	for(size_t j = 0; j < g_min_images;) {
		size_t pos = (size_t)(__int64)rand() % imagePoints_src.size();
		if(std::find(x.begin(), x.end(), pos) == x.end()) {
			x.push_back(pos);
			++j;
		}
	}
	imagePoints_dst.reserve(g_min_images);
	imagePoints_dst.resize(0);
	if(imagePoints_dst2) {
		imagePoints_dst2->reserve(g_min_images);
		imagePoints_dst2->resize(0);
	}
	for(auto pos : x) {
		imagePoints_dst.push_back(imagePoints_src[pos]);
		if(imagePoints_src2 && imagePoints_dst2) { 
			if(imagePoints_src2->size() > pos) {
				imagePoints_dst2->push_back((*imagePoints_src2)[pos]);
			}
		}
	}
}

void BoostCalibrate(size_t number_of_iterations, Mat& cameraMatrixL, Mat& cameraMatrixR, Mat& distortionCoeffsL, Mat& distortionCoeffsR, double rms[2]) {
	std::vector<Mat> CMVec2(number_of_iterations * 2), DCVec2(number_of_iterations * 2);
	if(number_of_iterations > 1) {
		vector<vector<Point2f>> imagePoints_left1(g_min_images);
		vector<vector<Point2f>> imagePoints_right1(g_min_images);

		double iteration_rms[256];
		memset(iteration_rms, 0, sizeof(iteration_rms));

		double average_rms = 0; 
		size_t rms_cnt = 0;

		double max_rms = 0; 

		size_t iter_num;
		for(iter_num = 0; !g_bTerminated && iter_num < number_of_iterations; ++iter_num) {
			SampleImagepoints(imagePoints_left, imagePoints_left1);
			SampleImagepoints(imagePoints_right, imagePoints_right1);

			try {
				rms[0] = CalibrateSingleCamera(imagePoints_left1, CMVec2[iter_num * 2], DCVec2[iter_num * 2]);
				rms[1] = CalibrateSingleCamera(imagePoints_right1, CMVec2[iter_num * 2 + 1], DCVec2[iter_num * 2 + 1]);

				iteration_rms[iter_num] = std::max(rms[0], rms[1]);
				if(max_rms < iteration_rms[iter_num]) {
					max_rms = iteration_rms[iter_num];
				}
				average_rms += iteration_rms[iter_num];
				++rms_cnt;
			}
			catch(...) {
				rms[0] = rms[1] = 100;
			}

			std::cout << iter_num << ' ' << "Pre-Re-projection error for left camera: " << rms[0] << std::endl;
			std::cout << iter_num << ' ' << "Pre-Re-projection error for right camera: " << rms[1] << std::endl;

			while(ProcessWinMessages());
		}
		average_rms /= rms_cnt;
		average_rms = (average_rms + max_rms) / 2; 

		const size_t max_iter = iter_num;

		for(iter_num = 0; iter_num < max_iter; ++iter_num) {
			if(iteration_rms[iter_num] <= average_rms) {
				cameraMatrixL = CMVec2[iter_num * 2]; cameraMatrixR = CMVec2[iter_num * 2 + 1]; distortionCoeffsL = DCVec2[iter_num * 2]; distortionCoeffsR = DCVec2[iter_num * 2 + 1];
				break; 
			}
			else { 
				--number_of_iterations; 
			}
		}

		for(++iter_num; iter_num < max_iter; ++iter_num) {
			if(iteration_rms[iter_num] <= average_rms) {
				cameraMatrixL = cameraMatrixL + CMVec2[iter_num * 2];
				cameraMatrixR = cameraMatrixR + CMVec2[iter_num * 2 + 1];
				distortionCoeffsL = distortionCoeffsL + DCVec2[iter_num * 2];
				distortionCoeffsR = distortionCoeffsR + DCVec2[iter_num * 2 + 1];
			}
			else {
				--number_of_iterations;
			}
		}
		if(number_of_iterations > 0) {
			cameraMatrixL = cameraMatrixL / (double)number_of_iterations;
			cameraMatrixR = cameraMatrixR / (double)number_of_iterations;
			distortionCoeffsL = distortionCoeffsL / (double)number_of_iterations;
			distortionCoeffsR = distortionCoeffsR / (double)number_of_iterations;
		}
	}

	if(number_of_iterations <= 1 && !g_bTerminated) {
		rms[0] = CalibrateSingleCamera(imagePoints_left, cameraMatrixL, distortionCoeffsL);
		rms[1] = CalibrateSingleCamera(imagePoints_right, cameraMatrixR, distortionCoeffsR);

		std::cout << "Pre-Re-projection error for left camera: " << rms[0] << std::endl;
		std::cout << "Pre-Re-projection error for right camera: " << rms[1] << std::endl;

		while(ProcessWinMessages());
	}
}

void Save_Images(Mat& image, vector<vector<Point2f>>& imagePoints, int points_idx, const std::string& name) {
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION); // Mar.4 2015.
	compression_params.push_back(0);

	std::string image_name = std::string(g_path_calib_images_dir) + name + ".png";

	std::cout << "Saving image" << ' ' << image_name << std::endl;
	cv::imwrite(image_name, image * std::max(256 / (int)g_bytedepth_scalefactor, 1), compression_params); // Mar.4 2015.

	if(points_idx > 0) {
		Mat color_image;

		if (image.type() != CV_8UC3) {
			normalize(image, color_image, 0, 255 * 256, NORM_MINMAX, CV_16UC1);
			cvtColor(color_image.clone(), color_image, COLOR_GRAY2RGB);
		}
		else {
			color_image = image.clone();
		}

		for(int k = 0; k < imagePoints[points_idx - 1].size(); ++k) {
			circle(color_image, Point2i((int)(imagePoints[points_idx - 1][k].x + 0.5), (int)(imagePoints[points_idx - 1][k].y + 0.5)), 5, Scalar(0, 0, 255 * 256), -1);
		}
		image_name = std::string(g_path_calib_images_dir) + name + "-points" + ".png";
		std::cout << "Saving image" << ' ' << image_name << std::endl;
		cv::imwrite(image_name, color_image, compression_params); // Mar.4 2015.
	}
}

return_t __stdcall AcquireImagepoints(LPVOID lp) {
	SImageAcquisitionCtl *ctl = (SImageAcquisitionCtl*)lp;

	size_t max_images = g_min_images * 3; 

	imagePoints_left.reserve(2 * max_images);
	imagePoints_right.reserve(2 * max_images);
	stereoImagePoints_left.reserve(max_images + 1);
	stereoImagePoints_right.reserve(max_images + 1);

	imagePoints_left.resize(1);
	imagePoints_right.resize(1);

	while (ProcessWinMessages());

	size_t current_N = 1; 
	size_t min_repeatability = 5; 

	ctl->_imagepoints_status = 1;

	_g_calibrationimages_frame->_toolbar->SetButtonStateByindex(TBSTATE_ENABLED, 0/*btn - save document*/, true/*remove enabled*/);
	_g_calibrationimages_frame->_stop_capturing = false; 

	g_aposteriory_minsdistance *= (ctl->_calib_images_from_files? 1: 1.01);

	while(!g_bTerminated && !ctl->_terminated && stereoImagePoints_left.size() < max_images && !_g_calibrationimages_frame->_stop_capturing) {
		if(ProcessWinMessages(10)) {
			continue;
		}

		__int64 time_now = OSDayTimeInMilliseconds();

		int64 image_localtime = 0;

		Mat left_image;
		Mat right_image;

		if(ctl->_calib_images_from_files) {
			if(!GetImagesFromFile(left_image, right_image, std::to_string(current_N))) {
				break;
			}
			min_repeatability = 2; 
			g_boardChessSize = Size(4/*points_per_row*/, 7/*points_per_colum*/); 
		}
		else 
		if(g_configuration._calib_auto_image_capture) {
			std::cout << "Getting images" << std::endl;
			if(!GetImages(left_image, right_image, &image_localtime, (int)g_rotating_buf_size - 1)) {
				continue;
			}
			std::cout << "Images Ok" << std::endl;
		}
		else
		if(!GetImagesEx(left_image, right_image, &image_localtime, (int)g_rotating_buf_size - 1)) {
			continue;
		}

		while(ProcessWinMessages());
		Mat cv_image[2] = { left_image.clone(), right_image.clone()};
		for (int c = 0; c < ARRAY_NUM_ELEMENTS(cv_image); ++c) {
			double fx = 700.0 / cv_image[c].cols;
			double fy = fx;
			HWND hwnd = (HWND)cvGetWindowHandle(cv_windows[c + 3].c_str());
			RECT clrect;
			if (GetWindowRect(GetParent(hwnd), &clrect)) {
				fx = (double)(clrect.right - clrect.left) / (double)cv_image[c].cols;
				fy = (double)(clrect.bottom - clrect.top) / (double)cv_image[c].rows;
				fx = fy = std::min(fx, fy);
			}
			cv::resize(cv_image[c], cv_image[c], cv::Size(0, 0), fx, fy, INTER_AREA);
			cv::imshow(cv_windows[c + 3], cv_image[c]);
		}


		int nl = -1;
		int nr = -1;

		// It is very likely that the pattern is recognizable in one image, but not in the other. 
		// The images need to be also of good quality. 
		// The strategy is to run first the detection with elevated constraint on quality, 
		// and then if one of the images has been recognized, but the other has been not, then run one more time with lowered constraint. 

		double min_confidence[2] = {g_configuration._calib_min_confidence, g_configuration._calib_min_confidence / 2}; // 2015-09-15 It is very difficult to capture images in certain environments.
		for(int j = 0; j < 1/*2*/; ++j) {
			while (ProcessWinMessages());
			std::cout << "EvaluateImagePoints(left)" << std::endl;
			if(nl == -1 && EvaluateImagePoints(left_image, imagePoints_left, *ctl, min_confidence[j], min_repeatability)) {
				nl = (int)imagePoints_left.size();
			}
			while(ProcessWinMessages());
			std::cout << "EvaluateImagePoints(right)" << std::endl;
			if(nr == -1 && EvaluateImagePoints(right_image, imagePoints_right, *ctl, min_confidence[j], min_repeatability)) {
				nr = (int)imagePoints_right.size();
			}
			while(ProcessWinMessages());

			if(nl > 0 && nr > 0) { 
				break; 
			}
			if(min_confidence[j] == 0) {
				break; 
			}
		}

		if(nl > 0 && nr > 0) {
			stereoImagePoints_left.push_back(imagePoints_left[nl - 1]);
			stereoImagePoints_right.push_back(imagePoints_right[nr - 1]);
		}
		if(nl > 0 && nl < (stereoImagePoints_left.size() + 5)) {
			imagePoints_left.resize(nl + 1);
			DrawImageAndBoard(std::to_string(nl) + '(' + std::to_string(stereoImagePoints_left.size()) + ')', cv_windows[0], left_image, imagePoints_left[nl - 1]);
		}
		if(nr > 0 && nr < (stereoImagePoints_right.size() + 5)) {
			imagePoints_right.resize(nr + 1);
			DrawImageAndBoard(std::to_string(nr) + '(' + std::to_string(stereoImagePoints_right.size()) + ')', cv_windows[1], right_image, imagePoints_right[nr - 1]);
		}


		auto lambda_Save_Images = [&left_image, &right_image, current_N](size_t N, int nl, int nr) {
			MyCreateDirectory(g_path_calib_images_dir, "AcquireImagepointsEx");
			while (ProcessWinMessages());
			if(current_N == 1) {
				Delete_FilesInDirectory(g_path_calib_images_dir);
				while (ProcessWinMessages());
			}
			Save_Images(left_image, imagePoints_left, nl, std::to_string(N) + 'l');
			while (ProcessWinMessages());
			Save_Images(right_image, imagePoints_right, nr, std::to_string(N) + 'r');
			while (ProcessWinMessages());

			std::string xml_name = std::string(g_path_calib_images_dir) + std::to_string(N) + ".xml";
			std::cout << "Saving xml" << ' ' << xml_name << std::endl;
			FileStorage fw(xml_name, FileStorage::WRITE);
			fw << "left_image" << left_image;
			fw << "right_image" << right_image;
			fw.release();
		};


		if(nl > 0 || nr > 0) {
			if(!ctl->_calib_images_from_files) {
				lambda_Save_Images(std::max(current_N, size_t(std::max(nl, nr))), nl, nr);
			}

			++current_N;

			if(nl > 0) {
				imageRaw_left.push_back(left_image);
			}
			if(nr > 0) {
				imageRaw_right.push_back(right_image);
			}
			if(nl > 0 && nr > 0) {
				stereoImageRaw_left.push_back(left_image);
				stereoImageRaw_right.push_back(right_image);

				if(stereoImagePoints_left.size() == g_min_images) {
					_g_calibrationimages_frame->_toolbar->SetButtonStateByindex(TBSTATE_ENABLED, 0/*btn - save document*/, false/*set enabled*/);
				}
			}
			while(ProcessWinMessages());
		}
		else
		if(ctl->_calib_images_from_files) {
			++current_N;
		}
		else
		if(ctl->_save_all_calibration_images) {
			lambda_Save_Images(current_N, 0, 0);

			++current_N;
		}
	}
	while(ProcessWinMessages());

	return 0;
}

return_t __stdcall ConductCalibration(LPVOID lp) {
	SImageAcquisitionCtl* ctl = (SImageAcquisitionCtl*)lp;


	g_aposteriory_minsdistance /= (ctl->_calib_images_from_files ? 1 : 1.01);

	imagePoints_left.resize(imagePoints_left.size() - 1);
	imagePoints_right.resize(imagePoints_right.size() - 1);

	srand((unsigned)time(NULL));

	if (g_configuration._file_log == 2) {
		VS_FileLog("", /*close*/true);
	}

	for (; stereoImagePoints_left.size() >= g_min_images && !g_bTerminated;) {
		Mat cameraMatrix[4];
		Mat distortionCoeffs[4];

		double rms[2];
		double rms_s = 0;

		Mat R, T, E, F;

		size_t number_of_iterations = (stereoImagePoints_left.size() - g_min_images) * 3 + 1;
		if (number_of_iterations > 64) {
			number_of_iterations = 64;
		}

		std::cout << "Calibrating cameras: iterations " << number_of_iterations << std::endl;

		if (ctl->_two_step_calibration) {
			std::cout << "Calibrating cameras: first undistort (because 2 step calibration is selected)" << std::endl;

			BoostCalibrate(number_of_iterations, cameraMatrix[2], cameraMatrix[3], distortionCoeffs[2], distortionCoeffs[3], rms);

			std::cout << "Undistorting images" << std::endl;

			vector<vector<Point2f>> imagePoints_left2 = imagePoints_left;
			vector<vector<Point2f>> imagePoints_right2 = imagePoints_right;
			vector<vector<Point2f>> stereoImagePoints_left2 = stereoImagePoints_left;
			vector<vector<Point2f>> stereoImagePoints_right2 = stereoImagePoints_right;

			bool ok = !g_bTerminated;

			ok = ok ? ReEvaluateUndistortImagePoints(imageRaw_left, imagePoints_left, cameraMatrix[2], distortionCoeffs[2], *ctl, cv_windows[0]) : false;
			ok = ok ? ReEvaluateUndistortImagePoints(imageRaw_right, imagePoints_right, cameraMatrix[3], distortionCoeffs[3], *ctl, cv_windows[1]) : false;
			ok = ok ? ReEvaluateUndistortImagePoints(stereoImageRaw_left, stereoImagePoints_left, cameraMatrix[2], distortionCoeffs[2], *ctl, cv_windows[0], &stereoImagePoints_right, &stereoImageRaw_right) : false;
			ok = ok ? ReEvaluateUndistortImagePoints(stereoImageRaw_right, stereoImagePoints_right, cameraMatrix[3], distortionCoeffs[3], *ctl, cv_windows[1], &stereoImagePoints_left, &stereoImageRaw_left) : false;

			if (ok) {
				if (stereoImagePoints_left.size() != stereoImagePoints_right.size() || stereoImagePoints_left.size() < g_min_images) {
					ok = false;
				}
			}

			if (!ok) {
				imagePoints_left = imagePoints_left2;
				imagePoints_right = imagePoints_right2;
				stereoImagePoints_left = stereoImagePoints_left2;
				stereoImagePoints_right = stereoImagePoints_right2;

				ctl->_two_step_calibration = false;
				std::cout << "two-step calibration has been canceled" << std::endl;
			}

			while (ProcessWinMessages());

			if (g_configuration._file_log == 2) {
				VS_FileLog("", /*close*/true);
			}
		}

		std::cout << "Calibrating cameras: single cameras" << std::endl;
		BoostCalibrate(number_of_iterations, cameraMatrix[0], cameraMatrix[1], distortionCoeffs[0], distortionCoeffs[1], rms);


		std::cout << "Calibrating stereo camera" << std::endl;

		std::vector<Mat> RVec(number_of_iterations), TVec(number_of_iterations), EVec(number_of_iterations), FVec(number_of_iterations);
		std::vector<Mat> CMVec(number_of_iterations * 2), DCVec(number_of_iterations * 2);

		g_objectPoints.resize(g_min_images, g_objectPoints[0]);

		double iteration_rms[256];
		memset(iteration_rms, 0, sizeof(iteration_rms));

		double average_rms = 0;
		size_t rms_cnt = 0;

		double max_rms = 0;

		size_t iter_num;
		for (iter_num = 0; !g_bTerminated && iter_num < number_of_iterations; ++iter_num) {
			vector<vector<Point2f>> stereoImagePoints_left1(g_min_images);
			vector<vector<Point2f>> stereoImagePoints_right1(g_min_images);

			SampleImagepoints(stereoImagePoints_left, stereoImagePoints_left1, &stereoImagePoints_right, &stereoImagePoints_right1);

			bool fix_intrinsic = ctl->_two_step_calibration && !g_configuration._calib_use_homography;
			fix_intrinsic = false;

			CMVec[2 * iter_num] = cameraMatrix[0].clone();
			CMVec[2 * iter_num + 1] = cameraMatrix[1].clone();
			DCVec[2 * iter_num] = distortionCoeffs[0].clone();
			DCVec[2 * iter_num + 1] = distortionCoeffs[1].clone();
			//rms_s = stereoCalibrate(g_objectPoints, stereoImagePoints_left1, stereoImagePoints_right1, CMVec[2 * iter_num], DCVec[2 * iter_num], CMVec[2 * iter_num + 1], DCVec[2 * iter_num + 1], g_imageSize, RVec[iter_num], TVec[iter_num], EVec[iter_num], FVec[iter_num], CALIB_USE_INTRINSIC_GUESS | CALIB_FIX_K1 | CALIB_FIX_K2 | CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 300, FLT_EPSILON/*DBL_EPSILON*/)); 
			//rms_s = stereoCalibrate(g_objectPoints, stereoImagePoints_left1, stereoImagePoints_right1, CMVec[2 * iter_num], DCVec[2 * iter_num], CMVec[2 * iter_num + 1], DCVec[2 * iter_num + 1], g_imageSize, RVec[iter_num], TVec[iter_num], EVec[iter_num], FVec[iter_num], CALIB_USE_INTRINSIC_GUESS, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 300, FLT_EPSILON/*DBL_EPSILON*/));
			rms_s = stereoCalibrate(g_objectPoints, stereoImagePoints_left1, stereoImagePoints_right1, CMVec[2 * iter_num], DCVec[2 * iter_num], CMVec[2 * iter_num + 1], DCVec[2 * iter_num + 1], g_imageSize, RVec[iter_num], TVec[iter_num], EVec[iter_num], FVec[iter_num], fix_intrinsic ? CALIB_FIX_INTRINSIC : CALIB_USE_INTRINSIC_GUESS, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));

			iteration_rms[iter_num] = rms_s;
			if (max_rms < rms_s) {
				max_rms = rms_s;
			}
			average_rms += rms_s;
			++rms_cnt;

			std::cout << iter_num << ' ' << "Re-projection error for stereo camera: " << rms_s << std::endl;

			while (ProcessWinMessages());

		}
		average_rms /= rms_cnt;
		average_rms = (average_rms + 2 * max_rms) / 3;

		const size_t max_iter = iter_num;

		R = RVec[0]; T = TVec[0]; E = EVec[0]; F = FVec[0];
		cameraMatrix[0] = CMVec[0]; cameraMatrix[1] = CMVec[1]; distortionCoeffs[0] = DCVec[0]; distortionCoeffs[1] = DCVec[1];

		for (iter_num = 0; iter_num < max_iter; ++iter_num) {
			if (iteration_rms[iter_num] <= average_rms) {
				R = RVec[iter_num]; T = TVec[iter_num]; E = EVec[iter_num]; F = FVec[iter_num];
				cameraMatrix[0] = CMVec[iter_num * 2]; cameraMatrix[1] = CMVec[iter_num * 2 + 1]; distortionCoeffs[0] = DCVec[iter_num * 2]; distortionCoeffs[1] = DCVec[iter_num * 2 + 1];
				break;
			}
			else {
				--number_of_iterations;
			}
		}

		for (++iter_num; iter_num < max_iter; ++iter_num) {
			if (iteration_rms[iter_num] <= average_rms) {
				R = R + RVec[iter_num];
				T = T + TVec[iter_num];
				E = E + EVec[iter_num];
				F = F + FVec[iter_num];
				cameraMatrix[0] = cameraMatrix[0] + CMVec[iter_num * 2];
				cameraMatrix[1] = cameraMatrix[1] + CMVec[iter_num * 2 + 1];
				distortionCoeffs[0] = distortionCoeffs[0] + DCVec[iter_num * 2];
				distortionCoeffs[1] = distortionCoeffs[1] + DCVec[iter_num * 2 + 1];
			}
			else {
				--number_of_iterations;
			}
		}
		if (number_of_iterations > 0) {
			R = R / (double)number_of_iterations;
			T = T / (double)number_of_iterations;
			E = E / (double)number_of_iterations;
			F = F / (double)number_of_iterations;
			cameraMatrix[0] = cameraMatrix[0] / (double)number_of_iterations;
			cameraMatrix[1] = cameraMatrix[1] / (double)number_of_iterations;
			distortionCoeffs[0] = distortionCoeffs[0] / (double)number_of_iterations;
			distortionCoeffs[1] = distortionCoeffs[1] / (double)number_of_iterations;
		}


		if (g_bTerminated) {
			break;
		}


		//if(ctl->_two_step_calibration) {
		//	rms[0] = CalibrateSingleCamera(imagePoints_left, cameraMatrix[2], distortionCoeffs[2]);
		//	rms[1] = CalibrateSingleCamera(imagePoints_right, cameraMatrix[3], distortionCoeffs[3]);

		//	while(ProcessWinMessages());

		//	vector<vector<Point2f>> imagePoints_left2 = imagePoints_left;
		//	vector<vector<Point2f>> imagePoints_right2 = imagePoints_right;
		//	vector<vector<Point2f>> stereoImagePoints_left2 = stereoImagePoints_left;
		//	vector<vector<Point2f>> stereoImagePoints_right2 = stereoImagePoints_right;

		//	bool ok = true; 

		//	ok = ok? ReEvaluateUndistortImagePoints(imageRaw_left, imagePoints_left, cameraMatrix[2], distortionCoeffs[2], *ctl, cv_windows[0]): false;
		//	ok = ok? ReEvaluateUndistortImagePoints(imageRaw_right, imagePoints_right, cameraMatrix[3], distortionCoeffs[3], *ctl, cv_windows[1]): false;
		//	ok = ok? ReEvaluateUndistortImagePoints(stereoImageRaw_left, stereoImagePoints_left, cameraMatrix[2], distortionCoeffs[2], *ctl, cv_windows[0]): false;
		//	ok = ok? ReEvaluateUndistortImagePoints(stereoImageRaw_right, stereoImagePoints_right, cameraMatrix[3], distortionCoeffs[3], *ctl, cv_windows[1]): false;

		//	if(!ok) {
		//		imagePoints_left = imagePoints_left2;
		//		imagePoints_right = imagePoints_right2;
		//		stereoImagePoints_left = stereoImagePoints_left2;
		//		stereoImagePoints_right = stereoImagePoints_right2;

		//		ctl->_two_step_calibration = false; 
		//		std::cout << "two-step calibration has been canceled" << std::endl;
		//	}
		//}

		//rms[0] = CalibrateSingleCamera(imagePoints_left, cameraMatrix[0], distortionCoeffs[0]);
		//rms[1] = CalibrateSingleCamera(imagePoints_right, cameraMatrix[1], distortionCoeffs[1]);

		//while(ProcessWinMessages());

		//g_objectPoints.resize(stereoImagePoints_left.size(), g_objectPoints[0]);

		////rms_s = stereoCalibrate(g_objectPoints, stereoImagePoints_left, stereoImagePoints_right, cameraMatrix[0], distortionCoeffs[0], cameraMatrix[1], distortionCoeffs[1], g_imageSize, R, T, E, F);
		//rms_s = stereoCalibrate(g_objectPoints, stereoImagePoints_left, stereoImagePoints_right, cameraMatrix[0], distortionCoeffs[0], cameraMatrix[1], distortionCoeffs[1], g_imageSize, R, T, E, F, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 300, FLT_EPSILON/*8DBL_EPSILON*/), CALIB_USE_INTRINSIC_GUESS | CALIB_FIX_K1 | CALIB_FIX_K2 | CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6);
		////rms_s = stereoCalibrate(g_objectPoints, stereoImagePoints_left, stereoImagePoints_right, cameraMatrix[0], distortionCoeffs[0], cameraMatrix[1], distortionCoeffs[1], g_imageSize, R, T, E, F, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 300, FLT_EPSILON/*8DBL_EPSILON*/), CALIB_USE_INTRINSIC_GUESS);

		//while(ProcessWinMessages());

		//std::cout << "Re-projection error for left camera: " << rms[0] << std::endl;
		//std::cout << "Re-projection error for right camera: " << rms[1] << std::endl;
		//std::cout << "Re-projection error for stereo camera: " << rms_s << std::endl;



		cv::Size rectified_image_size(g_imageSize.width * 6 / 4, g_imageSize.height * 6 / 4);
		rectified_image_size = g_imageSize;

		Mat Rl, Rr, Pl, Pr, Q;
		//cv::stereoRectify(cameraMatrix[0], distortionCoeffs[0], cameraMatrix[1], distortionCoeffs[1], g_imageSize, R, T, Rl, Rr, Pl, Pr, Q, 0);

		//rectified_image_size = g_imageSize; 
		//cv::stereoRectify(cameraMatrix[0], distortionCoeffs[0], cameraMatrix[1], distortionCoeffs[1], g_imageSize, R, T, Rl, Rr, Pl, Pr, Q, 0, 0);

		//cv::stereoRectify(cameraMatrix[0], distortionCoeffs[0], cameraMatrix[1], distortionCoeffs[1], g_imageSize, R, T, Rl, Rr, Pl, Pr, Q, 0, 0, rectified_image_size);

		cv::Rect Roi[2];
		cv::stereoRectify(cameraMatrix[0], distortionCoeffs[0], cameraMatrix[1], distortionCoeffs[1], g_imageSize, R, T, Rl, Rr, Pl, Pr, Q, 0, g_configuration._calib_rectify_alpha_param, rectified_image_size, &Roi[0], &Roi[1]);
		//if(Roi[0].width < (g_imageSize.width * 0.8) || Roi[1].width < (g_imageSize.width * 0.8)) {
		//	cv::stereoRectify(cameraMatrix[0], distortionCoeffs[0], cameraMatrix[1], distortionCoeffs[1], g_imageSize, R, T, Rl, Rr, Pl, Pr, Q, 0, 0, rectified_image_size, &Roi[0], &Roi[1]);
		//	std::cout << "The rectified image has been enlarged: " << std::endl;
		//}


		if (g_bTerminated) {
			break;
		}


		Mat map_l[4];
		Mat map_r[4];

		cv::initUndistortRectifyMap(cameraMatrix[0], distortionCoeffs[0], Rl, Pl, rectified_image_size, CV_16SC2/*CV_32F*/, map_l[0], map_l[1]);
		cv::initUndistortRectifyMap(cameraMatrix[1], distortionCoeffs[1], Rr, Pr, rectified_image_size, CV_16SC2/*CV_32F*/, map_r[0], map_r[1]);

		if (ctl->_two_step_calibration) {
			cv::initUndistortRectifyMap(cameraMatrix[2], distortionCoeffs[2], cv::Mat(), cv::getOptimalNewCameraMatrix(cameraMatrix[2], distortionCoeffs[2], g_imageSize, 0), g_imageSize, CV_16SC2/*CV_32F*/, map_l[2], map_l[3]);
			cv::initUndistortRectifyMap(cameraMatrix[3], distortionCoeffs[3], cv::Mat(), cv::getOptimalNewCameraMatrix(cameraMatrix[3], distortionCoeffs[3], g_imageSize, 0), g_imageSize, CV_16SC2/*CV_32F*/, map_r[2], map_r[3]);
		}

		FileStorage fs(".\\stereo_calibrate.xml", FileStorage::WRITE);

		fs << "cameraMatrix_l" << cameraMatrix[0];
		fs << "cameraMatrix_r" << cameraMatrix[1];
		fs << "distCoeffs_l" << distortionCoeffs[0];
		fs << "distCoeffs_r" << distortionCoeffs[1];
		fs << "R" << R;
		fs << "T" << T;
		fs << "E" << E;
		fs << "F" << F;
		fs << "R_l" << Rl;
		fs << "R_r" << Rr;
		fs << "P_l" << Pl;
		fs << "P_r" << Pr;
		fs << "Q" << Q;
		fs << "map_l1" << map_l[0];
		fs << "map_l2" << map_l[1];
		fs << "map_r1" << map_r[0];
		fs << "map_r2" << map_r[1];

		if (ctl->_two_step_calibration) {
			fs << "cameraMatrix_l_first" << cameraMatrix[2];
			fs << "cameraMatrix_r_first" << cameraMatrix[3];
			fs << "distCoeffs_l_first" << distortionCoeffs[2];
			fs << "distCoeffs_r_first" << distortionCoeffs[3];
			fs << "map_l1_first" << map_l[2];
			fs << "map_l2_first" << map_l[3];
			fs << "map_r1_first" << map_r[2];
			fs << "map_r2_first" << map_r[3];
		}

		fs << "roi_l" << Roi[0];
		fs << "roi_r" << Roi[1];

		fs.release();

		g_images_are_collected = true;
		break;
	}
	while (ProcessWinMessages()) {}

	ctl->_imagepoints_status = 0;

	return 0;
}


return_t __stdcall AcquireImagepointsWorkItem(LPVOID lp) {
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
	try {
		AcquireImagepoints(lp);
	}
	catch(Exception& ex) {
		std::cout << "AcquireImagepoints:" << ' ' << ex.msg << std::endl;
		((SImageAcquisitionCtl*)lp)->_imagepoints_status = 0;
	}
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
	return 0;
}


void launch_AcquireImages_calibration(SImageAcquisitionCtl& ctl, SPointsReconstructionCtl*) {
	int exposure_times[2] = {ctl._exposure_times[0], ctl._exposure_times[1]};

	if(g_bCamerasAreOk && !ctl._calib_images_from_files && !g_bTerminated) {
		OpenCameras(ctl);
	}

	ctl._exposure_times[0] = exposure_times[0];
	ctl._exposure_times[1] = exposure_times[1];

	if((g_bCamerasAreOk || ctl._calib_images_from_files) && !g_bTerminated) {
		ctl._imagepoints_status = -1;
		ctl._terminated = 0;
		if(g_bCamerasAreOk && !ctl._calib_images_from_files) {
			ctl._status = -1;
			QueueWorkItem(AcquireImages, &ctl);
		}
	}
}


void CalibrateCameras(StereoConfiguration& configuration, SImageAcquisitionCtl& image_acquisition_ctl) {
	g_images_are_collected = false;

	g_configuration._visual_diagnostics = false;

	Mat cv_image[2];

	while (ProcessWinMessages());
	_g_descriptorsLOCKER.lock();
	_g_calibrationimages_frame->SetTopMost();
	_g_main_frame->InsertHistory(_g_calibrationimages_frame);
	_g_descriptorsLOCKER.unlock();

	while (ProcessWinMessages());
	IPCSetLogHandler(_g_calibrationimages_frame->_hwnd);

	while (ProcessWinMessages());
	rootCVWindows(_g_calibrationimages_frame, 3, 0, cv_windows);
	rootCVWindows(_g_calibrationimages_frame, 2, 3, &cv_windows[3]);




	//launch_AcquireImages_calibration(image_acquisition_ctl, NULL);



	AcquireImagepoints(&image_acquisition_ctl);



	ConductCalibration(&image_acquisition_ctl);



	image_acquisition_ctl._terminated = 1;
	while(image_acquisition_ctl._status != 0) {
		ProcessWinMessages(10);
	}
	while(image_acquisition_ctl._imagepoints_status != 0) {
		ProcessWinMessages(10);
	}
	image_acquisition_ctl._terminated = 0;

	for(int x = 0; x < ARRAY_NUM_ELEMENTS(cv_windows); ++x) {
		try {
			if(cv_windows[x].size())
				destroyWindow(cv_windows[x]);
			cv_windows[x].resize(0);
		}
		catch(...) {
		}
	}

	if(g_images_are_collected) {
		g_bCalibrationExists = true;
	}

	while(_g_main_frame->StepBackHistory());
}


