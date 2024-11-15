#include "stdafx.h"

#include "XConfiguration.h"
#include "XAndroidCamera.h"

#include "FrameMain.h"

#include "opencv2\highgui\highgui_c.h"

#include "LoGSeedPoint.h"



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



double g_pattern_distance = 2.5; // 3 for grid of squares, 3.875 for chess board

size_t g_min_images = 16;
double g_aposteriory_minsdistance = 100;


vector<vector<Point2d>> imagePoints_left;
vector<vector<Point2d>> imagePoints_right;
vector<vector<Point2d>> stereoImagePoints_left;
vector<vector<Point2d>> stereoImagePoints_right;
vector<Mat> imageRaw_left;
vector<Mat> imageRaw_right;
vector<Mat> stereoImageRaw_left;
vector<Mat> stereoImageRaw_right;


std::string cv_windows[6];

extern HANDLE g_event_SeedPointIsAvailable;


double g_greenSquare_mean_data[3] = { 108.59519, 106.87839, 93.29372 };// { 117.41603, 108.29612, 89.85426 };




void DrawImageAndBoard(const std::string& aName, const std::string& window_name, Mat& cv_image, const vector<Point2d>& board) {
	Mat cv_image1;

	while (ProcessWinMessages());

	if (cv_image.type() != CV_8UC3) {
		cv::cvtColor(cv_image, cv_image1, COLOR_GRAY2RGB);
	}
	else {
		cv_image1 = cv_image.clone();
	}

	if (board.size()) {
		int red = 255;
		int green = 0;
		for (auto& point : board) {
			circle(cv_image1, Point2i((int)point.x, (int)point.y), 7, Scalar(0, green * 255, red * 255), -1);
		}
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
	Size textSize = getTextSize(aName.c_str(), 1, 5, 5, &baseLine);
	Point textOrigin(cv_image1.cols - textSize.width - 50, cv_image1.rows - 2 * baseLine - 10);
	putText(cv_image1, aName.c_str(), textOrigin, FONT_HERSHEY_SCRIPT_SIMPLEX, 3, Scalar(0, 0, 255 * 255), 5);

	try {
		cv::imshow(window_name.c_str(), cv_image1);
	}
	catch (Exception& ex) {
		std::cout << "DrawImageAndBoard:" << ' ' << ex.msg << std::endl;
	}

	while (ProcessWinMessages());
}


std::string CalibrationDirName() {
	if (g_configuration._frames_acquisition_mode > 1) {
		return std::string(g_path_calib_external_images_dir);
	}
	return std::string(g_path_calib_images_dir);
}


std::string CalibrationFileName() {
	return CalibrationDirName() + g_path_calibrate_file;
}



bool CalibrationFileExists() {
	bool exists = false;
	std::string path_calibrate_file = CalibrationFileName();
	if (MyGetFileSize(path_calibrate_file) > 0) {
		exists = true;
		//try {
		//	FileStorage fs(path_calibrate_file, FileStorage::READ);
		//	if (fs.isOpened()) {
		//		Mat cameraMatrix;
		//		fs["cameraMatrix_l"] >> cameraMatrix;
		//		exists = cameraMatrix.cols > 0;
		//		fs.release();
		//	}
		//}
		//catch (...) {
		//}
	}
	return exists;
}





// approximate contour with accuracy proportional to the contour perimeter
bool approximateContourWithQuadrilateral(const std::vector<Point>& contour, std::vector<Point>& approx, double minArea, double maxArea) {
	double epsilon = 0.025;
	approxPolyDP((contour), approx, arcLength((contour), false) * epsilon, false);
	while (approx.size() > 4) {
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
	if (approx.size() >= 4) {
		double area = std::fabs(contourArea((approx)));
		if (area > (minArea) && area < (maxArea)) {
			if (isContourConvex((approx))) {
				rc = true;
			}
		}
	}
	else {
		rc = false;
	}
	return rc;
}




void ClassBlobDetector::findBlobs(const Mat& image, Mat& binaryImage, std::vector<Center>& centers, std::vector<std::vector<cv::Point>>& contours) const {
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


	
	unsigned int threshold_intensity = 255 * g_bytedepth_scalefactor;

	/*
	* LoG generates too many boxes on binary images.
	* Use Guassian blur.
	* Use thershold 255 to start the generator from the inside of white area.
	* Increase the max_LoG so the generator goes closer to the black/white boundary.
	*/
	cv::GaussianBlur(binaryImage, binaryImage, Size(kmatN, kmatN), 0.9, 0.9);
	BlobCentersLoG(boxes, points, binaryImage, threshold_intensity, cv::Rect(), kmat, /*arff_file_requested*/false, /*intensity_avg_ptr*/nullptr, /*max_LoG_factor*/31.0);

	double desired_min_inertia = sqrt(_min_confidence);
	double ratio_threshold = desired_min_inertia * params.minCircularity * 0.8;

	for (size_t boxIdx = 0; boxIdx < boxes.size(); boxIdx++) {
		auto& contourOrig = boxes[boxIdx].contour;
		if (contourOrig.size() == 0) {
			continue;
		}

		double area = contourArea(Mat(contourOrig));

		std::vector<cv::Point> contour;
		if (!approximateContourWithQuadrilateral(contourOrig, contour, area / 2, area * 2)) {
			continue;
		}

		contour = contourOrig;

		contours.push_back(contour);

		Moments moms = moments(Mat(contour));
		area = moms.m00;


		// invariant moments; Mark Nixon, 7.3.2.2., p.318
		// built from normalized central moments.

		double M1 = moms.nu20 + moms.nu02;
		double M2 = std::pow(moms.nu20 - moms.nu02, 2) + 4 * std::pow(moms.nu11, 2);
		double M3 = std::pow(moms.nu30 - 3 * moms.nu12, 2) + std::pow(3 * moms.nu21 - moms.nu03, 2);

		//std::cout.precision(17);
		//std::cout << "M1=" << M1 << ' ' << "M2=" << M2 << ' ' << "M3=" << M3 << ' ' << std::endl;

		Center center;
		center.confidence = 1;
		if (params.filterByArea)
		{
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
			double perimeter = arcLength(Mat(contour), true);
			ratio *= 4 * CV_PI * area / (perimeter * perimeter);
		}
		else {
			ratio *= params.minCircularity;
		}


		if (ratio < ratio_threshold && (M1 > 0.18 || M2 > 0.004 || M3 > 0.0003)) {
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

	Mat aux = image.clone();
	//cv::medianBlur(aux, aux, 3);
	cv::GaussianBlur(aux, aux, Size(5, 5), 0.9, 0.9);
	cv::normalize(aux, grayscaleImage, 0, 255 * g_bytedepth_scalefactor, NORM_MINMAX, CV_32FC1, Mat());

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
			double scale = 1100.0 / binarizedImage.rows;
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
				cv::polylines(image0, &p, &n, 1, true, Scalar(0, 255 * 256, 0), 2, LINE_AA);
			}
			for each (auto & center in curCenters) {
				circle(image0, center.location, 3, Scalar(0, 0, 255 * 256), -1);
			}

			_ctl->_gate.lock();
			_ctl->_image2visualize = image0.clone();
			_ctl->_image_isvalid = true;
			_ctl->_last_image_timestamp = OSDayTimeInMilliseconds();
			_ctl->_gate.unlock();


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

			//if(count != expectedCount) {
			//	continue;
			//}
		}

		try {
			vector < vector<Center> > newCenters;
			for(size_t i = 0; i < curCenters.size(); i++) {
				if(curCenters[i].confidence < _min_confidence) { 
					continue; 
				}
				bool isNew = true;
				for(size_t j = 0; j < centers.size(); j++) {
					double minDist = std::max((double)params.minDistBetweenBlobs, curCenters[i].radius / 10.0);

					double dist = norm(centers[j][centers[j].size() / 2].location - curCenters[i].location);

					isNew = dist >= minDist && dist >= centers[j][centers[j].size() / 2].radius && dist >= curCenters[i].radius;
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




return_t __stdcall DetectBlackSquaresVia_HSV_Likeness(LPVOID lp) {
	SFeatureDetectorCtl* ctl = (SFeatureDetectorCtl*)lp;
	cv::Mat& image = ctl->_image;

	std::vector<KeyPoint> keyPoints;

	ctl->_status = 2;
	try {
		if (image.type() == CV_8UC3) {
			double mean_data[3] = { 108.59519, 106.87839, 93.29372 };// { 117.41603, 108.29612, 89.85426 };// { 117, 110, 86 };
			if (StandardizeImage_HSV_Likeness(image, mean_data)) {

				image = mat_loginvert2word(image);
				image = mat_invert2word(image);
				cv::normalize(image.clone(), image, 0, (size_t)256 * g_bytedepth_scalefactor, NORM_MINMAX, CV_16UC1, Mat());

				ctl->_detector->detect(image, ctl->_keyPoints);
			}
		}
	}
	catch (...) {
	}

	ctl->_gate.lock();
	ctl->_keyPoints.swap(keyPoints);
	ctl->_gate.unlock();

	ctl->_status = 0;

	return 0;
}

return_t __stdcall DetectBlackSquaresVia_ColorDistribution(LPVOID lp) {
	SFeatureDetectorCtl* ctl = (SFeatureDetectorCtl*)lp;
	cv::Mat image = ctl->_image.clone();

	std::vector<KeyPoint> keyPoints;
	vector<Point2f> pointBuf;

	ctl->_status = 2;
	try {
		if (image.type() == CV_8UC3) {
			double mean_data[3] = {g_greenSquare_mean_data[0], g_greenSquare_mean_data[1], g_greenSquare_mean_data[2]};

			double invCovar_data[9] = { 0.008706637, 0.002279566, -0.01101841, 0.002279566, 0.043274569, -0.03528071, -0.011018411, -0.035280710, 0.04288963 };// { 0.003871324, 0.001966916, -0.001440268, 0.001966916, 0.036550065, -0.030285284, -0.001440268, -0.030285284, 0.035661410 };
			double invCholesky_data[9] = { 0.05144795, 0.0000000, 0.0000000, -0.05682516, 0.1193855, 0.0000000, -0.05320382, -0.1703575, 0.2070981 };// { 0.061335769, 0.0000000, 0.0000000, 0.007146916, 0.1040693, 0.0000000, -0.007626832, -0.1603734, 0.1888423 };

			cv:Mat mean = cv::Mat(1, 3, CV_64F, mean_data);
			cv::Mat invCovar = cv::Mat(3, 3, CV_64F, invCovar_data);
			cv::Mat invCholesky = cv::Mat(3, 3, CV_64F, invCholesky_data);

			cv::Mat stdDev;
			cv::Mat factorLoadings;

			if (ctl->_saturationFactor != 0.0) {
				for (auto& mean : mean_data) {
					mean *= ctl->_saturationFactor; 
				}
			}

			StandardizeImage_Likeness(image, mean, stdDev, factorLoadings, invCovar, invCholesky);

			image = mat_loginvert2word(image);
			image = mat_invert2word(image);
			cv::normalize(image.clone(), image, 0, (size_t)256 * g_bytedepth_scalefactor, NORM_MINMAX, CV_16UC1, Mat());
		}

		ctl->_detector->detect(image, keyPoints);
	}
	catch (...) {
	}

	const int targetNumberOfKeyPoints = g_boardChessSize.width * g_boardChessSize.height;

	pointBuf.reserve(keyPoints.size());
	pointBuf.resize(0);

	for (auto& keyPoint : keyPoints) {
		pointBuf.push_back(keyPoint.pt);
	}

	ctl->_gate.lock();
	ctl->_keyPoints.swap(keyPoints);
	ctl->_pointBuf.swap(pointBuf);
	ctl->_gate.unlock();

	ctl->_status = 0;

	return 0;
}

return_t __stdcall BuildChessMinEnclosingQuadrilateral(LPVOID lp) {
	SFeatureDetectorCtl* ctl = (SFeatureDetectorCtl*)lp;
	cv::Mat& imageInp = ctl->_image;

	std::vector<cv::Point2f> pointBuf = ctl->_pointBuf;

	std::vector<cv::Point2f> approx2fminQuad; // Qaudrilateral delimiting the area of the image that contains corners. 
	Rect approxBoundingRectMapped;
	cv::Mat H; // Homography that transforms quadrilateral to rectangle (not rotated), enabling therefore the sorting of corners. 

	ctl->_status = 2;
	try {
		vector<Point2f> approx2fminRectMapped(4); // Qadrilateral tranformed to rectangle.


		const int targetNumberOfKeyPoints = g_boardChessSize.width * g_boardChessSize.height;


		bool found = pointBuf.size() == targetNumberOfKeyPoints;

		// build min. enclosing quadrilateral
		if (found) {
			Mat image = imageInp.clone();

			std::vector<Point> blobs(pointBuf.size());
			for (int k = 0; k < pointBuf.size(); ++k) {
				blobs[k] = Point2i((int)(pointBuf[k].x + 0.5), (int)(pointBuf[k].y + 0.5));
			}

			std::vector<Point> hull;
			cv::convexHull(blobs, hull, true/*will be counter-clockwise because Y is pointing down*/, true/*do return points*/);

			// minimize the number of points
			vector<Point> approx;
			cv::approxPolyDP(Mat(hull), approx, arcLength(Mat(hull), true) * 0.02, true);

			if (approx.size() >= 6) {
				// Approximate the outer hull with min. enclosing quadrilateral. 

				// 1. Order the hull counterclockwise starting with the most remote point, because
				// that is how the pattern of ideal points is preset. 
				// 2. Use ideal points to build homography. 
				// 3. Use homography to build the min. enclosing rectangle. 
				// 4. Transform back to create the min. enclosing quadrilateral. 

				// 1. The first point must be the most remote point, the one that corresponds to top right. 

				// Remove (in iterations) two closest points until two or one is left. 
				vector<int> approx_idx(approx.size());
				for (int j = 0; j < approx_idx.size(); ++j) {
					approx_idx[j] = j;
				}
				while (approx_idx.size() > 2) {
					double min_dist = std::numeric_limits<double>::max();
					int min_idx[2] = { 0, 1 };
					for (int j = 0, k = 1; j < approx_idx.size(); ++j, ++k) {
						if (k >= approx_idx.size()) {
							k = 0;
						}
						double dist = cv::norm(approx[approx_idx[j]] - approx[approx_idx[k]]);
						if (dist < min_dist) {
							min_dist = dist;
							min_idx[0] = approx_idx[j];
							min_idx[1] = approx_idx[k];
						}
					}
					approx_idx.erase(std::remove_if(approx_idx.begin(), approx_idx.end(), [&min_idx](const int idx) -> bool {
						return idx == min_idx[0] || idx == min_idx[1];
					}), approx_idx.end());
				}
				if (approx_idx.size() == 2) {
					if (approx[approx_idx[0]].y > approx[approx_idx[1]].y) {
						approx_idx[0] = approx_idx[1];
					}
				}

				int idx = approx_idx[0];

				//int idx = 0;
				//// Find a pair of most distant points. Take the one that is higher, i.e. smaller y. 
				//{
				//	int max_idx[2] = { 0, 1 };
				//	double max_dist = std::numeric_limits<double>::min();
				//	const int N = (int)approx.size();
				//	for (int j = 0; j < N; ++j) {
				//		for (int k = 0; k < N; ++k) {
				//			double dist = cv::norm(approx[j] - approx[k]);
				//			if (dist > max_dist) {
				//				max_dist = dist;
				//				max_idx[0] = j;
				//				max_idx[1] = k;
				//			}
				//		}
				//	}
				//	if (approx[max_idx[0]].y > approx[max_idx[1]].y) {
				//		max_idx[0] = max_idx[1];
				//	}

				//	idx = max_idx[0];
				//}

				if (idx > 0) {
					std::rotate(approx.begin(), approx.begin() + idx, approx.end());
				}




				vector<Point2f> approx2f(approx.size());
				for (int j = 0; j < 6; ++j) {
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
				if (approx2f[0].x > approx2f[1].x) {
					float max_x = 0;
					for_each(ideal_approx2f.begin(), ideal_approx2f.end(), [&max_x](Point2f& point) {
						if (point.x > max_x) {
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
				catch (Exception& ex) {
					std::cout << "ExtractCornersOfChessPattern:" << ' ' << ex.msg << std::endl;
				}


				// 3. Use homography to build the min. enclosing rectangle. 

				vector<Point2f> approx2fMapped;
				cv::perspectiveTransform(approx2f, approx2fMapped, H);

				RotatedRect minRect = minAreaRect(Mat(approx2fMapped));
				minRect.points(&approx2fminRectMapped[0]);

				approxBoundingRectMapped = cv::boundingRect(Mat(approx2fMapped));


				// 4. Transform back to create the min. enclosing quadrilateral. 

				approx2fminQuad.resize(4);
				perspectiveTransform(approx2fminRectMapped, approx2fminQuad, H.inv());


				// 5. Visualize the centers and quadrilateral. 

				Mat image0;
				Mat image1;
				if (image.type() == CV_16UC1) {
					normalize(image, image0, 0, 255 * 256, NORM_MINMAX);
					cvtColor(image0, image1, COLOR_GRAY2RGB);
				}
				else {
					image1 = image.clone();
				}

				for (int j = 0; j < 4; ++j) {
					cv::line(image1, approx2fminQuad[j], approx2fminQuad[(j + 1) % 4], Scalar(255 * 256, 0, 0));
				}
				for (int k = 0; k < blobs.size(); ++k) {
					circle(image1, blobs[k], 3, Scalar(0, 0, 255 * 256), -1);
				}
				const Point* p = &approx[0];
				int n = (int)approx.size();
				cv::polylines(image1, &p, &n, 1, true, Scalar(0, 255 * 256, 0), 1, LINE_AA);

				double fx = 280.0 / image1.rows;
				cv::resize(image1, image0 = Mat(), cv::Size(0, 0), fx, fx, INTER_AREA);


				ctl->_gate.lock();
				ctl->_image2visualize = image0.clone();
				ctl->_image_isvalid = true;
				ctl->_last_image_timestamp = OSDayTimeInMilliseconds();
				ctl->_gate.unlock();


				vector<int> compression_params;
				compression_params.push_back(IMWRITE_PNG_COMPRESSION); // Mar.4 2015.
				compression_params.push_back(0);


				cv::imwrite(std::string(CalibrationDirName()) + "ST1-" + ctl->_outputWindow + "-ChessPattern-enclose.png", image1, compression_params);
			}
		}
		else {
			ctl->_gate.lock();
			ctl->_image2visualize = imread(IMG_DELETEDOCUMENT_H, cv::IMREAD_ANYCOLOR);
			ctl->_image_isvalid = true;
			ctl->_last_image_timestamp = OSDayTimeInMilliseconds();
			ctl->_gate.unlock();
		}
	}
	catch (...) {
	}


	ctl->_gate.lock();
	ctl->_approx2fminQuad.swap(approx2fminQuad);
	ctl->_H = H;
	ctl->_approxBoundingRectMapped = approxBoundingRectMapped;
	ctl->_gate.unlock();

	ctl->_status = 0;

	return 0;
}




return_t __stdcall BuildChessGridCorners(LPVOID lp) {
	SFeatureDetectorCtl* ctl = (SFeatureDetectorCtl*)lp;
	cv::Mat& imageInp = ctl->_image;

	vector<Point2f> approx2fminQuad = ctl->_approx2fminQuad;
	Rect approxBoundingRectMapped = ctl->_approxBoundingRectMapped;

	vector<Point2f> edgesBuf;

	ctl->_status = 2;
	try {
		Mat& H = ctl->_H; // Homography that transforms quadrilateral to rectangle (not rotated). 

		if (approx2fminQuad.size()) {
			Mat image = imageInp.clone();

			double color2gray[3] = { 0.114, 0.587, 0.299 }; // opencv has BGR
			ConvertColoredImage2Mono(image, color2gray, [](double ch) {
				return std::min(ch * 256, 256.0 * 256.0);
			});

			Mat imageMapped;
			warpPerspective(image, imageMapped, H, image.size()/*, INTER_CUBIC*/);
			Mat crop(imageMapped, approxBoundingRectMapped);
			Mat grayscale;
			normalize(crop, grayscale, 0, 255, NORM_MINMAX, CV_8UC1, Mat());

			float windowRatio = (float)imageMapped.cols / (float)grayscale.cols;


			vector<vector<Point2f>> edgesMappedBuf;

			for (int level = 90; level < 200; level += 10) {

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
				for (int k = 0; k < normalizedCorners.rows; ++k) {
					for (int j = 0; j < normalizedCorners.cols; ++j) {
						auto p = normalizedCorners.at<unsigned char>(k, j);
						if (p > 180) {
							cornersBufMapped.push_back(Point2f((float)j, (float)k));
						}
					}
				}


				std::vector<int> dist_levels(cornersBufMapped.size(), -1);
				int	clusters_count = std::numeric_limits<int>::max();
				const int centroids_count = g_boardChessCornersSize.width * g_boardChessCornersSize.height;
				int dist = grayscale.cols / ((g_boardChessCornersSize.width + 1) * 2);
				std::sort(cornersBufMapped.begin(), cornersBufMapped.end(), [dist](const Point2f& one, const Point2f& another) -> bool {
					if (one.y < (another.y - dist)) return true;
					if (one.y > (another.y + dist)) return false;
					if (one.x < another.x) return true;
					return false;
				});
				//partitionEx(cornersBufMapped, dist_levels, [dist](const Point2f& one, const Point2f& another) -> bool {
				//	return std::max(abs(one.y - another.y), abs(one.x - another.x)) < dist;
				//});
				//clusters_count = *(std::max_element(dist_levels.begin(), dist_levels.end())) + 1;
				//if (clusters_count != centroids_count) {
				//	continue;
				//}

				if (cornersBufMapped.size() < centroids_count) {
					continue;
				}

				vector<Point2f> edgesMapped;

				Mat kmeans_centers;
				Mat cornersMat;
				try {
					for (auto& corner : cornersBufMapped) {
						cornersMat.push_back(corner);
					}
					kmeans(cornersMat, centroids_count, dist_levels,
						TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 50, 0.1), 3/*random number generator state*/, KMEANS_PP_CENTERS/*KMEANS_USE_INITIAL_LABELS*/, kmeans_centers);

					for (int k = 0; k < kmeans_centers.rows; ++k) {
						edgesMapped.push_back(Point2f(kmeans_centers.at<float>(k, 0), kmeans_centers.at<float>(k, 1)));
					}

					dist_levels.resize(0);
					dist_levels.resize(edgesMapped.size(), -1);
					partitionEx(edgesMapped, dist_levels, [dist](const Point2f& one, const Point2f& another) -> bool {
						return std::max(abs(one.y - another.y), abs(one.x - another.x)) < dist;
					});
					clusters_count = *(std::max_element(dist_levels.begin(), dist_levels.end())) + 1;

					if (clusters_count == centroids_count) {
						std::sort(edgesMapped.begin(), edgesMapped.end(), [dist](const Point2f& one, const Point2f& another) -> bool {
							if (one.y < (another.y - dist)) return true;
							if (one.y > (another.y + dist)) return false;
							if (one.x < another.x) return true;
							return false;
						});

						edgesMappedBuf.push_back(edgesMapped);
					}
				}
				catch (Exception& ex) {
					std::cout << "ExtractCornersOfChessPattern:" << ' ' << ex.msg << std::endl;
				}



				Mat image1;
				cvtColor(/*binImage*/grayscale, image1, COLOR_GRAY2RGB);
				for (auto& point : cornersBufMapped) {
					circle(image1, Point2i((int)point.x, (int)point.y), 1, Scalar(0, 0, 255), -1);
				}
				for (int k = 0; k < edgesMapped.size(); ++k) {
					circle(image1, Point2i((int)edgesMapped[k].x, (int)edgesMapped[k].y), 3, Scalar(0, 255, 0), -1);
				}



				ctl->_gate.lock();
				ctl->_image2visualize = image1.clone();
				ctl->_image_isvalid = true;
				ctl->_last_image_timestamp = OSDayTimeInMilliseconds();
				ctl->_gate.unlock();



				vector<int> compression_params;
				compression_params.push_back(IMWRITE_PNG_COMPRESSION); // Mar.4 2015.
				compression_params.push_back(0);

				for (int k = 0; k < edgesMapped.size(); ++k) {
					putText(image1, std::to_string(k).c_str(), edgesMapped[k], FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
				}
				std::string name = std::string(CalibrationDirName()) + "ST2-" + ctl->_outputWindow + "-ChessPattern-level" + std::to_string(level) + ".png";
				cv::imwrite(name, image1, compression_params);
			}


			vector<Point2f> edgesMax(edgesMappedBuf.size() ? g_boardChessCornersSize.width * g_boardChessCornersSize.height : 0);
			vector<Point2f> edgesMin(edgesMappedBuf.size() ? g_boardChessCornersSize.width * g_boardChessCornersSize.height : 0, Point2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::max()));
			vector<Point2f> edgesDif(edgesMappedBuf.size() ? g_boardChessCornersSize.width * g_boardChessCornersSize.height : 0);


			vector<Point2f> edgesMapped(edgesMappedBuf.size() ? g_boardChessCornersSize.width * g_boardChessCornersSize.height : 0);
			for (auto& edges : edgesMappedBuf) {
				for (size_t j = 0; j < edges.size(); ++j) {
					auto& p = edges[j];
					edgesMapped[j].x += p.x;
					edgesMapped[j].y += p.y;
					if (p.x > edgesMax[j].x) edgesMax[j].x = p.x;
					if (p.y > edgesMax[j].y) edgesMax[j].y = p.y;
					if (p.x < edgesMin[j].x) edgesMin[j].x = p.x;
					if (p.y < edgesMin[j].y) edgesMin[j].y = p.y;
				}
			}
			for (auto& point : edgesMapped) {
				point *= 1.0 / (double)edgesMappedBuf.size();
			}

			
			std::transform(edgesMax.begin(), edgesMax.end(), edgesMin.begin(), edgesDif.begin(), [](Point2f& a, Point2f& b) ->Point2f { return a - b; });
			float maxDiff_x = std::max_element(edgesDif.begin(), edgesDif.end(), [](Point2f& a, Point2f& b) ->bool { return a.x < b.x; })->x * 2;
			float maxDiff_y = std::max_element(edgesDif.begin(), edgesDif.end(), [](Point2f& a, Point2f& b) ->bool { return a.y < b.y; })->y * 2;

			if (maxDiff_x < 4) maxDiff_x = 4;
			if (maxDiff_y < 4) maxDiff_y = 4;


			try {
				if (edgesMapped.size()) {
					cv::cornerSubPix(grayscale, edgesMapped, Size(maxDiff_x, maxDiff_y)/*window size*/, Size(-1, -1)/*no zero zone*/, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001));
				}
			}
			catch (Exception& ex) {
				std::cout << "ExtractCornersOfChessPattern:" << ' ' << ex.msg << std::endl;
				edgesMapped.resize(0);
			}


			Mat image1;
			cvtColor(/*binImage*/grayscale, image1, COLOR_GRAY2RGB);
			for (int k = 0; k < edgesMapped.size(); ++k) {
				cv::circle(image1, Point2i((int)edgesMapped[k].x, (int)edgesMapped[k].y), 3, Scalar(0, 0, 255), -1);
			}
			for (int k = 0; k < edgesMapped.size(); ++k) {
				putText(image1, std::to_string(k).c_str(), edgesMapped[k], FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
			}



			ctl->_gate.lock();
			ctl->_image2visualize = image1.clone();
			ctl->_image_isvalid = true;
			ctl->_last_image_timestamp = OSDayTimeInMilliseconds();
			ctl->_gate.unlock();



			vector<int> compression_params;
			compression_params.push_back(IMWRITE_PNG_COMPRESSION); // Mar.4 2015.
			compression_params.push_back(0);

			cv::imwrite(CalibrationDirName() + "ST3-" + ctl->_outputWindow + "-ChessPattern-Mapped.png", image1, compression_params);


			if (edgesMapped.size() == (g_boardChessCornersSize.width * g_boardChessCornersSize.height)) {
				for (auto& point : edgesMapped) {
					point.x += approxBoundingRectMapped.x;
					point.y += approxBoundingRectMapped.y;
				}

				edgesBuf.resize(edgesMapped.size());
				cv::perspectiveTransform(edgesMapped, edgesBuf, H.inv());

				Mat grayscale;
				normalize(image, grayscale, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
				cv::cornerSubPix(grayscale, edgesBuf, Size(maxDiff_x * windowRatio, maxDiff_y * windowRatio)/*window size*/, Size(-1, -1)/*no zero zone*/, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100, 0.001));

				for (int k = 0; k < edgesBuf.size(); ++k) {
					putText(grayscale, std::to_string(k).c_str(), edgesBuf[k], FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(0, 0, 255), 2);
				}
				cv::imwrite(std::string(CalibrationDirName()) + "ST3-" + ctl->_outputWindow + "-ChessPattern-Labeled.png", grayscale, compression_params);
				ctl->_gate.lock();
				ctl->_image2visualize = grayscale;
				ctl->_image_isvalid = true;
				ctl->_last_image_timestamp = OSDayTimeInMilliseconds();
				ctl->_gate.unlock();
			}
			else {
				static int x = 0;
				++x;
			}
		}
	}
	catch (Exception& ex) {
		std::cout << "ExtractCornersOfChessPattern:" << ' ' << ex.msg << std::endl;
	}


	ctl->_gate.lock();
	ctl->_edgesBuf.reserve(edgesBuf.size());
	ctl->_edgesBuf.resize(0);
	for (auto& point : edgesBuf) {
		ctl->_edgesBuf.push_back(point);
	}
	ctl->_gate.unlock();

	ctl->_status = 0;

	return 0;
}




bool ExtractCornersOfChessPattern(Mat *imagesInp, vector<Point2d> *pointBufs, const size_t N, double saturationFactor, ClassBlobDetector& blobDetector) {

	SFeatureDetectorCtl controls[3] = { SFeatureDetectorCtl(imagesInp[0].clone()), SFeatureDetectorCtl(N > 1? imagesInp[1].clone(): Mat()), SFeatureDetectorCtl(N > 2 ? imagesInp[2].clone() : Mat()) };
	for (auto& ctl : controls) {
		ctl._saturationFactor = saturationFactor;
		ctl._detector = new ClassBlobDetector(blobDetector, &ctl);
		ctl._status = 0;
	}



	std::function<void()> runMessagePipe = [&]() {
		int status = 0;
		do {
			while (ProcessWinMessages(10));

			for (auto& ctl : controls) {
				if (ctl._image_isvalid) {

					ctl._gate.lock();
					cv::Mat image0 = ctl._image2visualize;
					ctl._image2visualize = Mat();
					int64_t image_timestamp = ctl._last_image_timestamp;
					ctl._image_isvalid = false;
					ctl._gate.unlock();

					double fx = 280.0 / image0.rows;

					HWND hwnd = (HWND)cvGetWindowHandle(ctl._outputWindow.c_str());
					RECT clrect;
					if (GetWindowRect(GetParent(hwnd), &clrect)) {
						fx = (double)(clrect.bottom - clrect.top) / (double)image0.rows;
					}

					cv::Mat image1;
					cv::resize(image0, image1, cv::Size(0, 0), fx, fx, INTER_LINEAR);

					cv::imshow(ctl._outputWindow, image1);
				}
			}

			status = 0;
			for (auto& ctl : controls) status += ctl._status;

		} while (status != 0);
	};



	controls[0]._outputWindow = "IMAGECalibr1";
	controls[1]._outputWindow = "IMAGECalibr2";
	controls[2]._outputWindow = "IMAGECalibr3";

	for (int j = 0; j < N; ++j) {
		controls[j]._status = 1;
		QueueWorkItem(DetectBlackSquaresVia_ColorDistribution, &controls[j]);
	}


	runMessagePipe();


	controls[0]._outputWindow = "IMAGECalibr4";
	controls[1]._outputWindow = "IMAGECalibr5";
	controls[2]._outputWindow = "IMAGECalibr6";

	for (int j = 0; j < N; ++j) {
		controls[j]._status = 1;
		QueueWorkItem(BuildChessMinEnclosingQuadrilateral, &controls[j]);
	}


	runMessagePipe();


	controls[0]._outputWindow = "Calibr1";
	controls[1]._outputWindow = "Calibr2";
	controls[2]._outputWindow = "Calibr3";

	for (int j = 0; j < N; ++j) { 
		controls[j]._status = 1;
		QueueWorkItem(BuildChessGridCorners, &controls[j]);
	}



	runMessagePipe();



	for (int j = 0; j < ARRAY_NUM_ELEMENTS(controls); ++j) {
		if (!controls[j]._edgesBuf.empty()) {
			pointBufs[j] = controls[j]._edgesBuf;
		}
	}



	const int tartgetNumberOfCorners = g_boardChessCornersSize.width * g_boardChessCornersSize.height;
	int okCount = 0;
	for (int j = 0; j < ARRAY_NUM_ELEMENTS(controls); ++j) 
		if(pointBufs[j].size() == tartgetNumberOfCorners) ++okCount;

	return okCount == N;
}

bool DetectChessGrid(Mat *images, vector<Point2d> *pointBufs, const size_t N, double saturationFactor, ClassBlobDetector& blobDetector) {
	for (size_t x = 0; x < N; ++x) pointBufs[x].clear();
	bool found = ExtractCornersOfChessPattern(images, pointBufs, N, saturationFactor, blobDetector);
	return found;
}


bool buildPointsFromImages(Mat* images, vector<Point2d>* pointBufs, const size_t N, SImageAcquisitionCtl& ctl, double min_confidence, size_t min_repeatability) {
	const int tartgetNumberOfCorners = g_boardChessCornersSize.width * g_boardChessCornersSize.height;
	ClassBlobDetector blobDetector = ClassBlobDetector(min_confidence, min_repeatability, 40, ctl._pattern_is_whiteOnBlack, ctl._pattern_is_chessBoard);
	bool found = false;
	for (size_t x = 0; x < N; ++x) pointBufs[x].clear();
	if (images[0].data) {
		g_imageSize = images[0].size();
		double saturationFactors[6] = { 1.0, 0.9, 1.1, 1.2, 0.8, 1.3 };
		std::vector<vector<Point2d>> foundBufs(N);
		for (auto saturationFactor : saturationFactors) {
			found = DetectChessGrid(images, foundBufs.data(), N, saturationFactor, blobDetector);
			for (size_t j = 0; j < N; ++j) {
				if (foundBufs[j].size() == tartgetNumberOfCorners) {
					pointBufs[j] = foundBufs[j];
				}
			}
			if (found) {
				break;
			}

			int okCount = 0;
			for (size_t j = 0; j < N; ++j) {
				if (pointBufs[j].size() == tartgetNumberOfCorners) {
					++okCount;
				}
			}
			found = okCount == N;
			if (found) {
				break;
			}
		}
	}

	return found;
}


double calc_betweenimages_rmse(vector<Point2d>& image1, vector<Point2d>& image2) {
	double rmse = 0;
	size_t min_size = std::min(image1.size(), image2.size());
	bool strict = true;
	for(int j = 0; j < (int)min_size; ++j) {
		double e = pow(image1[j].x - image2[j].x, 2) + pow(image1[j].y - image2[j].y, 2);
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
	return sqrt(rmse); 
}



std::vector<bool> EvaluateImagePoints(cv::Mat cv_images[2], std::vector<std::vector<cv::Point2d>> imagePoints[2], SImageAcquisitionCtl& ctl, double aposteriory_minsdistance, double min_confidence = 0.4, size_t min_repeatability = 2) {
	constexpr int N = 2;

	static double min_rmse = std::numeric_limits<double>::max();

	std::vector<bool> is_ok(N, false);
	std::vector<double> min_image_rmse(N, std::numeric_limits<double>::max());

	std::vector<cv::Point2d> points[N];

	if(buildPointsFromImages(cv_images, points, N, ctl, min_confidence, min_repeatability)) {
		for (int j = 0; j < N; ++j) {
			is_ok[j] = points[j].size() == g_boardChessCornersSize.width * g_boardChessCornersSize.height;

			int x = (int)imagePoints[j].size() - 1;

			imagePoints[j][x].swap(points[j]);

			if (x > 0) {
				for (int k = x - 1; k >= 0 && is_ok[j]; --k) {
					double rmse = calc_betweenimages_rmse(imagePoints[j][k], imagePoints[j][x]);
					if (min_image_rmse[j] > rmse) {
						min_image_rmse[j] = rmse;
					}
					if (min_rmse > rmse && rmse > aposteriory_minsdistance) {
						min_rmse = rmse;
					}
				}
			}
		}
		for (int j = 0; j < N; ++j) {
			if (min_image_rmse[j] < aposteriory_minsdistance) {
				for (int k = j + 1; k < N; ++k) {
					if (min_image_rmse[k] < aposteriory_minsdistance) {
						is_ok[j] = false;
						is_ok[k] = false;
						std::cout << "rejected by min_rmse=" << min_rmse << " vs. aposteriory_minsdistance=" << aposteriory_minsdistance << std::endl;
					}
				}
			}
		}
	}

	return is_ok;
}


void Save_Images(Mat& image, vector<vector<Point2d>>& imagePoints, int points_idx, const std::string& name) {
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION); // Mar.4 2015.
	compression_params.push_back(0);

	std::string image_name = std::string(CalibrationDirName()) + name + ".png";

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
		image_name = std::string(CalibrationDirName()) + name + "-points" + ".png";
		std::cout << "Saving image" << ' ' << image_name << std::endl;
		cv::imwrite(image_name, color_image, compression_params); // Mar.4 2015.
	}
}

/*
Read the frame's buffer to the end and pick
a frame that has images captured at smallest time-difference.
The N controlls a minimum amount of frames to consider.
*/
void VisualizeCapturedImages(cv::Mat& left_image, cv::Mat& right_image) {
	static bool onMouse_isActive[6] = { 0, 0, 0, 0, 0 };
	static MouseCallbackParameters mouse_callbackParams[6];

	while (ProcessWinMessages());
	Mat cv_image[2] = { left_image.clone(), right_image.clone() };
	for (int c = 0; c < ARRAY_NUM_ELEMENTS(cv_image); ++c) {
		double fx = 700.0 / cv_image[c].cols;
		double fy = fx;

		std::string& imagewin_name = cv_windows[c + 4];

		HWND hwnd = (HWND)cvGetWindowHandle(imagewin_name.c_str());
		RECT clrect;
		if (GetWindowRect(GetParent(hwnd), &clrect)) {
			fx = (double)(clrect.right - clrect.left) / (double)cv_image[c].cols;
			fy = (double)(clrect.bottom - clrect.top) / (double)cv_image[c].rows;
			fx = fy = std::min(fx, fy);
		}
		cv::resize(cv_image[c], cv_image[c], cv::Size(0, 0), fx, fy, INTER_AREA);
		cv::imshow(imagewin_name, cv_image[c]);

		if (!onMouse_isActive[c]) {
			void OnMouseCallback(int event, int x, int y, int flags, void* userdata); 

			mouse_callbackParams[c].scaleFactors.fx = fx;
			mouse_callbackParams[c].scaleFactors.fy = fy;
			mouse_callbackParams[c].windowNumber = c + 1;
			cv::setMouseCallback(imagewin_name, OnMouseCallback, (void*)&mouse_callbackParams[c]); 
			onMouse_isActive[c] = true;
		}
	}
}

int g_sync_button_pressed;
return_t __stdcall ShowSynchronizationButton(LPVOID lp) {
	ProcessWinMessages(10);
	MessageBoxA(NULL, "Press when ready", "Synchronization", MB_OK | MB_TOPMOST | MB_SETFOREGROUND);
	g_sync_button_pressed = true;
	return 0;
}
std::function<void()> g_imageGetter = nullptr;
return_t __stdcall GetImagesWorkItem(LPVOID lp) {
	g_imageGetter();
	return 0;
}
bool GetImagesEx(Mat& left, Mat& right, int64_t* time_spread, const int N/*min_frames_2_consider*/, bool& new_images) {
	g_sync_button_pressed = false;
	QueueWorkItem(ShowSynchronizationButton);

	bool imagesOk = false;

	Mat image[2];
	bool imageReady = false;
	bool imageGetterFinished = true; 
	bool click_has_occured = false;
	while (!(g_sync_button_pressed || click_has_occured) || !imageGetterFinished) {
		if (imageReady) {
			VisualizeCapturedImages(image[0], image[1]);
			imageReady = false;
			imagesOk = true;
		}
		if(imageGetterFinished && !(g_sync_button_pressed || click_has_occured)) {
			g_imageGetter = [&]() {
				imageReady = GetImages(image[0], image[1], time_spread, N, 500);
				g_imageGetter = nullptr;
				imageGetterFinished = true;
			};
			imageGetterFinished = false;
			QueueWorkItem(GetImagesWorkItem);
		}

		DWORD anEvent = WaitForSingleObjectEx(g_event_SeedPointIsAvailable, 0, FALSE);
		if (anEvent == WAIT_OBJECT_0) {
			click_has_occured = true;
		}

		while (ProcessWinMessages(10));
	}
	if (imagesOk) {
		left = image[0];
		right = image[1];
	}
	if (click_has_occured) {
		SetEvent(g_event_SeedPointIsAvailable);
	}

	new_images = imagesOk;

	if (!g_sync_button_pressed) {
		HWND msgbox = FindWindowA(NULL, "Synchronization");
		if (msgbox != NULL) {
			SendMessageA(msgbox, WM_CLOSE, 0, 0);
		}

		imagesOk = false;
	}
	else {
		imagesOk = left.rows > 0 && right.rows > 0;
	}

	return imagesOk;
}



std::function<void()> g_imageSaveLambda = nullptr;
return_t __stdcall SaveImagesWorkItem(LPVOID lp) {
	g_imageSaveLambda();
	return 0;
}

std::function<void()> g_imageGetFromFilesLambda = nullptr;
return_t __stdcall GetImagesImagesWorkItem(LPVOID lp) {
	g_imageGetFromFilesLambda();
	return 0;
}


template<typename T>
void WhiteBalance(Mat& image, double whiteFactor[3]) {
	typedef cv::Vec<T, 3> Vec3t;
	for (int r = 0; r < image.rows; ++r) {
		for (int c = 0; c < image.cols; ++c) {
			Vec3t pixImage(image.at<T>(r, c));
			for (int j = 0; j < 3; ++j) {
				pixImage[j] = cv::saturate_cast<T>(pixImage[j] * whiteFactor[j]);
			}
		}
	}
}

void WhiteBalance(Mat_<float>& image, double whiteFactor[3]) {
	WhiteBalance<float>(image, whiteFactor);
}

void WhiteBalance(Mat_<uchar>& image, double whiteFactor[3]) {
	WhiteBalance<uchar>(image, whiteFactor);
}


return_t __stdcall AcquireImagepoints(LPVOID lp) {
	SImageAcquisitionCtl *ctl = (SImageAcquisitionCtl*)lp;

	size_t max_images = g_min_images * 3; 

	stereoImagePoints_left.reserve(max_images + 1);
	stereoImagePoints_right.reserve(max_images + 1);

	vector<vector<Point2d>> imagePoints[2];
	for (auto& p : imagePoints) {
		p.reserve(2 * max_images);
		p.resize(1);
	}


	while (ProcessWinMessages());

	size_t current_N = 1; 

	ctl->_imagepoints_status = 1;

	_g_calibrationimages_frame->_toolbar->SetButtonStateByindex(TBSTATE_ENABLED, 0/*btn - save document*/, true/*remove enabled*/);
	_g_calibrationimages_frame->_stop_capturing = false; 

	g_aposteriory_minsdistance *= (ctl->_calib_images_from_files? 1: 1.01);

	Mat images[2];

	bool new_unprocessed_images_inprogress = false;

	while(!g_bTerminated && !ctl->_terminated && stereoImagePoints_left.size() < max_images && !_g_calibrationimages_frame->_stop_capturing) {
		if(ProcessWinMessages(10)) {
			continue;
		}

		__int64 time_now = OSDayTimeInMilliseconds();


		int64 image_localtime = 0;

		Mat& left_image = images[0];
		Mat& right_image = images[1];


		DWORD anEvent = WaitForSingleObjectEx(g_event_SeedPointIsAvailable, 0, FALSE);
		if (anEvent == WAIT_OBJECT_0) {
			ImageScaleFactors sf = g_LoG_seedPoint.params.scaleFactors;

			Point pt;
			pt.x = (int)(g_LoG_seedPoint.x / sf.fx + 0.5);
			pt.y = (int)(g_LoG_seedPoint.y / sf.fy + 0.5);

			int windowNumber = g_LoG_seedPoint.params.windowNumber - 1;

			Mat &aux = images[windowNumber];

			if (aux.rows > pt.y) {
				double seedReference[3];

				BuildIdealChannels_Likeness(aux, pt, seedReference, 7);
				double grayWorldMean = 0;
				for (int j = 0; j < 3; ++j) {
					if (seedReference[j] == 0) {
						seedReference[j] = 1;
					}
					grayWorldMean += seedReference[j];
				}
				grayWorldMean /= 3.0;

				std::cout << "Reference RGB values: " << seedReference[0] << ',' << seedReference[1] << ',' << seedReference[2] << ';' << " Mean value: " << grayWorldMean << std::endl;

				Mat image;
				aux.convertTo(image, CV_32FC3);

				double whiteFactor[3] = { 1, 1, 1 };
				for (int j = 0; j < 3; ++j) {
					switch (g_LoG_seedPoint.eventValue) {
					case 1:
					case 4:
						whiteFactor[j] = 255.0 / seedReference[j]; // ground truth 
						//whiteFactor[j] = grayWorldMean / seedReference[j]; // gray world 
						break;
					case 2:
					case 5:
						whiteFactor[j] = g_greenSquare_mean_data[j] / seedReference[j]; // green world 
						break;
					}
				}

				WhiteBalance<float>(image, whiteFactor);

				image.convertTo(aux, CV_8UC3);

				VisualizeCapturedImages(left_image, right_image);
			}
		}



		int nl = static_cast<int>(imagePoints[0].size()) - 1;
		int nr = static_cast<int>(imagePoints[1].size()) - 1;


		bool imagesFromFilesAreOk = false;

		g_imageGetFromFilesLambda = [&]() {
			Mat im[2];
			imagesFromFilesAreOk = GetImagesFromFile(im[0], im[1], imagePoints[0][nl], imagePoints[1][nr], std::to_string(current_N));
			if (imagesFromFilesAreOk) {
				left_image = im[0];
				right_image = im[1];
			}
			g_imageGetFromFilesLambda = nullptr;
		};

		QueueWorkItem(GetImagesImagesWorkItem);
		while (g_imageGetFromFilesLambda != nullptr) {
			ProcessWinMessages(10);
		}

		if (ctl->_calib_images_from_files) {
			if (!imagesFromFilesAreOk) {
				break;
			}

		}
		else 
		if(g_configuration._calib_auto_image_capture) {
			if (!imagesFromFilesAreOk) {
				std::cout << "Getting images" << std::endl;
				if (!GetImages(left_image, right_image, &image_localtime, (int)g_rotating_buf_size - 1)) {
					continue;
				}
				std::cout << "Images Ok" << std::endl;
			}

		}
		else
		if (!imagesFromFilesAreOk) {
			bool new_images;
			bool images_ok = GetImagesEx(left_image, right_image, &image_localtime, 1, new_images);

			if (images_ok && new_images) {
				new_unprocessed_images_inprogress = true;
			}

			if (!new_images && !new_unprocessed_images_inprogress) {
				AndroidCaptureStillImageRequest request;
				PostObject(request);
				new_unprocessed_images_inprogress = true;
				images_ok = false;
			}

			if (!images_ok) {
				continue;
			}
		}

		if (_g_calibrationimages_frame->_stop_capturing) {
			continue;
		}

		VisualizeCapturedImages(left_image, right_image);

		bool pointsAreFromFile = false;

		std::vector<bool> imagePoints_ok = { imagePoints[0][nl].size() != 0, imagePoints[1][nr].size() != 0 };
		if (!imagePoints_ok[0] && !imagePoints_ok[1]) {
			double rmse = g_aposteriory_minsdistance;
			imagePoints_ok = EvaluateImagePoints(images, imagePoints, *ctl, imagesFromFilesAreOk? rmse * 0.8: rmse, g_configuration._calib_min_confidence);
			new_unprocessed_images_inprogress = false;
		}
		else {
			pointsAreFromFile = true;
			g_imageSize = images[0].size();
		}

		


		nl = imagePoints_ok[0]? static_cast<int>(imagePoints [0].size()) : -1;
		nr = imagePoints_ok[1]? static_cast<int>(imagePoints [1].size()) : -1;





		if(nl > 0 && nr > 0) {
			size_t l = (size_t)nl - 1;
			size_t r = (size_t)nr - 1;
			stereoImagePoints_left.push_back(imagePoints[0][l]);
			stereoImagePoints_right.push_back(imagePoints[1][r]);

			if (stereoImagePoints_left.size() == g_min_images) {
				_g_calibrationimages_frame->_toolbar->SetButtonStateByindex(TBSTATE_ENABLED, 0/*btn - save document*/, false/*set enabled*/);
			}
		}


		if(nl > 0 && nl < (stereoImagePoints_left.size() + 5)) {
			imagePoints[0].resize((size_t)nl + 1);
			size_t x = (size_t)nl - 1;
			DrawImageAndBoard(std::to_string(nl) + '(' + std::to_string(stereoImagePoints_left.size()) + ')', cv_windows[0], images[0], imagePoints[0][x]);
		}
		if(nr > 0 && nr < (stereoImagePoints_right.size() + 5)) {
			imagePoints[1].resize((size_t)nr + 1);
			size_t x = (size_t)nr - 1;
			DrawImageAndBoard(std::to_string(nr) + '(' + std::to_string(stereoImagePoints_right.size()) + ')', cv_windows[1], images[1], imagePoints[1][x]);
		}




		auto lambda_Save_Images = [&](size_t N, int nl, int nr) {
			MyCreateDirectory(CalibrationDirName(), "AcquireImagepointsEx");

			if (ctl->_save_all_calibration_images && !imagesFromFilesAreOk) {
				if (current_N == 1) {
					Delete_FilesInDirectory(CalibrationDirName());
					while (ProcessWinMessages());
				}
				Save_Images(images[0], imagePoints[0], nl, std::to_string(N) + 'l');
				Save_Images(images[1], imagePoints[1], nr, std::to_string(N) + 'r');
			}

			if (nl < 0) {
				nl = static_cast<int>(imagePoints[0].size());
			}
			if (nr < 0) {
				nr = static_cast<int>(imagePoints[1].size());
			}

			std::string xml_name = std::string(CalibrationDirName()) + std::to_string(N) + ".xml";
			std::cout << "Saving xml" << ' ' << xml_name << std::endl;
			FileStorage fw(xml_name, FileStorage::WRITE);
			fw << "left_points" << imagePoints[0][(size_t)nl - 1];
			fw << "right_points" << imagePoints[1][(size_t)nr - 1];
			fw.release();
		};




		g_imageSaveLambda = [&]() {
			if (nl > 0) {
				imageRaw_left.push_back(left_image);
			}
			else {
				imagePoints[0][imagePoints[0].size() - 1].resize(0);
			}
			if (nr > 0) {
				imageRaw_right.push_back(right_image);
			}
			else {
				imagePoints[1][imagePoints[1].size() - 1].resize(0);
			}
			if (nl > 0 && nr > 0) {
				stereoImageRaw_left.push_back(left_image);
				stereoImageRaw_right.push_back(right_image);
			}
			if (nl > 0 || nr > 0) {
				if (!pointsAreFromFile) {
					lambda_Save_Images(std::max((int)current_N, std::max(nl, nr)), nl, nr);
				}
			}
			if (nl > 0 || nr > 0 || imagesFromFilesAreOk) {
				++current_N;
			}

			g_imageSaveLambda = nullptr;
		};

		QueueWorkItem(SaveImagesWorkItem);

		while (g_imageSaveLambda != nullptr) {
			ProcessWinMessages(10);
		}

	}

	while(ProcessWinMessages());


	imagePoints_left.swap(imagePoints[0]);
	imagePoints_right.swap(imagePoints[1]);


	imagePoints_left.resize(imagePoints_left.size() - 1);
	imagePoints_right.resize(imagePoints_right.size() - 1);


	return 0;
}







vector<Point2f> ImagePoints2d_To_ImagePoints2f(vector<Point2d>& imagePoints2d_src) {
	vector<Point2f> imagePoints2f_dst;
	for (const auto& point2d : imagePoints2d_src) {
		imagePoints2f_dst.push_back(point2d);
	}
	return imagePoints2f_dst;
}

void SampleImagepoints(const size_t N, vector<vector<Point2d>>& imagePoints_src, vector<vector<Point2f>>& imagePoints_dst, std::vector<size_t>* selection = 0, vector<vector<Point2d>>* imagePoints_src2 = 0, vector<vector<Point2f>>* imagePoints_dst2 = 0) {
	std::vector<size_t> x;
	x.reserve(N);
	for (size_t j = 0; j < N;) {
		if (N != imagePoints_src.size()) {
			size_t pos = (size_t)(__int64)rand() % imagePoints_src.size();
			if (std::find(x.begin(), x.end(), pos) == x.end()) {
				x.push_back(pos);
				++j;
			}
		}
		else {
			x.push_back(j);
			++j;
		}
	}
	imagePoints_dst.reserve(N);
	imagePoints_dst.resize(0);
	if (imagePoints_dst2) {
		imagePoints_dst2->reserve(N);
		imagePoints_dst2->resize(0);
	}
	std::ostringstream ostr;
	for (auto pos : x) {
		ostr << pos << ' ';
		imagePoints_dst.push_back(ImagePoints2d_To_ImagePoints2f(imagePoints_src[pos]));
		if (imagePoints_src2 && imagePoints_dst2) {
			if (imagePoints_src2->size() > pos) {
				imagePoints_dst2->push_back(ImagePoints2d_To_ImagePoints2f((*imagePoints_src2)[pos]));
			}
		}
	}
	std::cout << "sample: " << ostr.str() << std::endl;

	if (selection != nullptr) {
		selection->swap(x);
	}
}










Mat ShowUndistortedImageAndPoints(Mat& image, vector<Point2d>& imagePoints, Mat map[2], const std::string& cv_window, const std::string& text) {
	Mat undistorted;
	remap(image, undistorted, map[0], map[1], INTER_CUBIC/*INTER_LINEAR*//*INTER_NEAREST*/, BORDER_CONSTANT);

	while (ProcessWinMessages());

	DrawImageAndBoard(text, cv_window, undistorted, imagePoints);

	return undistorted;
}



return_t __stdcall BuildSeedSelection(LPVOID lp) {
	SStereoCalibrationCtl* ctl = (SStereoCalibrationCtl*)lp;

	std::vector<size_t> seedSelection;

	Mat cameraMatrix[2];
	Mat distortionCoeffs[2];

	ctl->_seedSelection.resize(0);

	ctl->_status = 2;
	try {
		std::cout << "Building a good batch of points" << std::endl;



		vector<Point3f> objectPoints;
		for (int i = 0; i < g_boardChessCornersSize.height; ++i) {
			for (int j = 0; j < g_boardChessCornersSize.width; ++j) {
				objectPoints.push_back(Point3f(float(j * g_pattern_distance), float(i * g_pattern_distance), 0));
			}
		}



		double rms_s = 0;

		vector<vector<Point2f>> seedPoints_left(3);
		vector<vector<Point2f>> seedPoints_right(3);


		auto sampleCalibrate = [&]() {
			Mat RVec, TVec, EVec, FVec;
			std::vector<Mat> CMVec(2), DCVec(2);

			int calibrateFlags = 0;// CALIB_FIX_INTRINSIC;// CALIB_USE_INTRINSIC_GUESS;

			vector<vector<Point3f>> _objectPoints;
			_objectPoints.resize(seedPoints_left.size(), objectPoints);

			CMVec[0] = cameraMatrix[0].clone();
			CMVec[1] = cameraMatrix[1].clone();
			DCVec[0] = distortionCoeffs[0].clone();
			DCVec[1] = distortionCoeffs[1].clone();
			rms_s = stereoCalibrate(_objectPoints, seedPoints_left, seedPoints_right,
				CMVec[0], DCVec[0],
				CMVec[1], DCVec[1],
				g_imageSize,
				RVec, TVec, EVec, FVec,
				calibrateFlags,
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, FLT_EPSILON));
		};



		for (int j = 0; !g_bTerminated && j < 100; ++j) {

			SampleImagepoints(3, stereoImagePoints_left, seedPoints_left, &seedSelection, &stereoImagePoints_right, &seedPoints_right);

			sampleCalibrate();

			std::cout << j << ' ' << "Re-projection error: " << rms_s << std::endl;

			if (rms_s < 1) {
				break;
			}
		}



		if (rms_s >= 1) {
			seedSelection.resize(0);
			std::cout << "A minimal set of image points cannot be determined; terminating" << std::endl;
			return 0;
		}

	}
	catch (...) {
	}

	ctl->_gate.lock();
	ctl->_seedSelection.swap(seedSelection);
	ctl->_gate.unlock();

	ctl->_status = 0;

	return 0;
}



return_t __stdcall TryPointsSelection(LPVOID lp) {
	SStereoCalibrationCtl* ctl = (SStereoCalibrationCtl*)lp;

	std::vector<size_t> seedSelection = ctl->_seedSelection;
	size_t pos = ctl->_selection_pos;

	ctl->_pos_ok = false;

	Mat cameraMatrix[2];
	Mat distortionCoeffs[2];
	Mat R; 
	Mat T; 
	Mat E; 
	Mat F;


	ctl->_status = 2;
	try {
		vector<Point3f> objectPoints;
		for (int i = 0; i < g_boardChessCornersSize.height; ++i) {
			for (int j = 0; j < g_boardChessCornersSize.width; ++j) {
				objectPoints.push_back(Point3f(float(j * g_pattern_distance), float(i * g_pattern_distance), 0));
			}
		}



		double rms_s = 0;

		vector<vector<Point2f>> seedPoints_left;
		vector<vector<Point2f>> seedPoints_right;


		auto sampleCalibrate = [&]() {
			int calibrateFlags = 0;// CALIB_FIX_INTRINSIC;// CALIB_USE_INTRINSIC_GUESS;

			vector<vector<Point3f>> _objectPoints;
			_objectPoints.resize(seedPoints_left.size(), objectPoints);

			rms_s = stereoCalibrate(_objectPoints, seedPoints_left, seedPoints_right,
				cameraMatrix[0], cameraMatrix[0],
				distortionCoeffs[1], distortionCoeffs[1],
				g_imageSize,
				R, T, E, F,
				calibrateFlags,
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, FLT_EPSILON));
		};


		if (std::find(seedSelection.begin(), seedSelection.end(), pos) != seedSelection.end()) {
			return 0;
		}


		for (auto x : seedSelection) {
			seedPoints_left.push_back(ImagePoints2d_To_ImagePoints2f(stereoImagePoints_left[x]));
			seedPoints_right.push_back(ImagePoints2d_To_ImagePoints2f(stereoImagePoints_right[x]));
		}

		seedPoints_left.push_back(ImagePoints2d_To_ImagePoints2f(stereoImagePoints_left[pos]));
		seedPoints_right.push_back(ImagePoints2d_To_ImagePoints2f(stereoImagePoints_right[pos]));


		sampleCalibrate();

		if (rms_s < 1) {
			ctl->_pos_ok = true;
		}
		else {
			std::cout << "Re-projection error for pos" << ' ' << pos << " : " << rms_s << std::endl;
		}

		if (ctl->_rms_s != nullptr) {
			*ctl->_rms_s = rms_s;
		}
	}
	catch (...) {
	}

	ctl->_status = 0;

	return 0;
}



return_t __stdcall StereoCalibrateIteration(LPVOID lp) {
	SStereoCalibrationCtl* ctl = (SStereoCalibrationCtl*)lp;

	Mat cameraMatrix[2];
	Mat distortionCoeffs[2];

	ctl->_status = 2;

	srand(static_cast<unsigned int>(ctl->_iter_num));

	try {
		size_t sample_size = ctl->_sample_size;


		vector<Point3f> objectPoints;
		for (int i = 0; i < g_boardChessCornersSize.height; ++i) {
			for (int j = 0; j < g_boardChessCornersSize.width; ++j) {
				objectPoints.push_back(Point3f(float(j * g_pattern_distance), float(i * g_pattern_distance), 0));
			}
		}

		vector<vector<Point3f>> vectorObjectPoints;

		vectorObjectPoints.resize(sample_size, objectPoints);


		vector<vector<Point2f>> stereoImagePoints_left1(sample_size);
		vector<vector<Point2f>> stereoImagePoints_right1(sample_size);

		SampleImagepoints(sample_size, stereoImagePoints_left, stereoImagePoints_left1, nullptr, &stereoImagePoints_right, &stereoImagePoints_right1);

		int calibrateFlags = 0;// CALIB_FIX_INTRINSIC;// CALIB_USE_INTRINSIC_GUESS;
		if (!ctl->_cameraMatrix1->empty() && !ctl->_cameraMatrix2->empty()) {
			calibrateFlags = CALIB_FIX_FOCAL_LENGTH;
		}

		*ctl->_rms_s = stereoCalibrate(vectorObjectPoints,
			stereoImagePoints_left1, stereoImagePoints_right1,
			*ctl->_cameraMatrix1, *ctl->_distortionCoeffs1,
			*ctl->_cameraMatrix2, *ctl->_distortionCoeffs2,
			g_imageSize,
			*ctl->_R, *ctl->_T, *ctl->_E, *ctl->_F,
			calibrateFlags,
			TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, FLT_EPSILON));

		std::cout << ctl->_iter_num << ' ' << "Re-projection error for stereo camera: " << *ctl->_rms_s << std::endl;
	}
	catch (...) {
	}

	ctl->_gate.lock();
	ctl->_gate.unlock();

	ctl->_status = 0;

	return 0;
}



return_t __stdcall ConductCalibration(LPVOID lp) {
	SImageAcquisitionCtl* ctl = (SImageAcquisitionCtl*)lp;


	SStereoCalibrationCtl controls[6];


	std::function<void()> runMessagePipe = [&]() {
		int status = 0;
		do {
			while (ProcessWinMessages(10));

			for (auto& ctl : controls) {
				if (ctl._image_isvalid) {

					ctl._gate.lock();
					cv::Mat image0 = ctl._image2visualize;
					ctl._image2visualize = Mat();
					int64_t image_timestamp = ctl._last_image_timestamp;
					ctl._image_isvalid = false;
					ctl._gate.unlock();

					double fx = 280.0 / image0.rows;

					HWND hwnd = (HWND)cvGetWindowHandle(ctl._outputWindow.c_str());
					RECT clrect;
					if (GetWindowRect(GetParent(hwnd), &clrect)) {
						fx = (double)(clrect.bottom - clrect.top) / (double)image0.rows;
					}

					cv::Mat image1;
					cv::resize(image0, image1, cv::Size(0, 0), fx, fx, INTER_LINEAR);

					cv::imshow(ctl._outputWindow, image1);
				}
			}

			status = 0;
			for (auto& ctl : controls) status += ctl._status;

		} while (status != 0);
	};



	g_aposteriory_minsdistance /= (ctl->_calib_images_from_files ? 1 : 1.01);

	srand((unsigned)time(NULL));




	controls[0]._outputWindow = "IMAGECalibr1";
	controls[1]._outputWindow = "IMAGECalibr2";
	controls[2]._outputWindow = "IMAGECalibr3";
	controls[3]._outputWindow = "IMAGECalibr4";
	controls[4]._outputWindow = "IMAGECalibr5";
	controls[5]._outputWindow = "IMAGECalibr6";





	for (auto& ctl : controls) {
		ctl._status = 0;
	}


	controls[0]._status = 1;
	QueueWorkItem(BuildSeedSelection, &controls[0]);

	runMessagePipe();




	std::vector<size_t> seedSelection = controls[0]._seedSelection;

	if (seedSelection.size() == 0) {
		return 0;
	}



	std::cout << "Building calibration set " << std::endl;


	std::ostringstream ostrSelection;

	for (auto pos : seedSelection) {
		ostrSelection << pos << ' ';
	}


	std::vector<size_t> finalPointsSelection = seedSelection;



	for (size_t itr_num = 0; itr_num < stereoImagePoints_left.size(); itr_num += ARRAY_NUM_ELEMENTS(controls)) {
		for (size_t j = 0; j < ARRAY_NUM_ELEMENTS(controls); ++j) {
			size_t x = itr_num + j;

			if (x >= stereoImagePoints_left.size()) {
				continue;
			}

			auto& ctl = controls[j];
			ctl._rms_s = nullptr;

			ctl._selection_pos = x;
			ctl._pos_ok = false;

			if (std::find(seedSelection.begin(), seedSelection.end(), x) != seedSelection.end()) {
				continue;
			}

			ctl._seedSelection = seedSelection;
			ctl._status = 1;
			QueueWorkItem(TryPointsSelection, &ctl);
		}

		runMessagePipe();

		for (auto& ctl : controls) {
			if (ctl._pos_ok) {
				size_t pos = ctl._selection_pos;

				finalPointsSelection.push_back(pos);
				ostrSelection << pos << ' ';
				std::cout << "calibration set: " << ostrSelection.str() << std::endl;
			}
		}
	}

	std::cout << "calibration set: " << ostrSelection.str() << std::endl;

	if (finalPointsSelection.size() == 0) {
		return 0;
	}




	std::cout << "Calibrating stereo camera" << std::endl;




	vector<vector<Point2d>> points_left;
	vector<vector<Point2d>> points_right;

	points_left.swap(stereoImagePoints_left);
	points_right.swap(stereoImagePoints_right);

	for (auto pos : finalPointsSelection) {
		stereoImagePoints_left.push_back(points_left[pos]);
		stereoImagePoints_right.push_back(points_right[pos]);
	}


	size_t number_of_iterations = (stereoImagePoints_left.size() - g_min_images) * 3 + 1;
	if (number_of_iterations > 64) {
		number_of_iterations = 64;
	}





	std::cout << "Calibrating cameras: iterations " << number_of_iterations << std::endl;

	std::vector<Mat> RVec(number_of_iterations); 
	std::vector<Mat> TVec(number_of_iterations); 
	std::vector<Mat> EVec(number_of_iterations); 
	std::vector<Mat> FVec(number_of_iterations);

	std::vector<Mat> CMVec(number_of_iterations * 2); 
	std::vector<Mat> DCVec(number_of_iterations * 2);

	size_t min_images = number_of_iterations > 1 ? g_min_images : stereoImagePoints_left.size();








	double iteration_rms[256];
	::memset(iteration_rms, 0, sizeof(iteration_rms));

	double average_rms = 0;
	size_t rms_cnt = 0;

	double max_rms = 0;

	size_t iter_num = 0;
	for (; !g_bTerminated && iter_num < number_of_iterations; iter_num += ARRAY_NUM_ELEMENTS(controls)) {
		for (size_t j = 0; j < ARRAY_NUM_ELEMENTS(controls); ++j) {
			size_t x = iter_num + j;

			auto& ctl = controls[j];
			ctl._rms_s = nullptr;

			ctl._iter_num = x;
			if (ctl._iter_num >= number_of_iterations) {
				continue;
			}

			ctl._cameraMatrix1 = &CMVec[2 * x];
			ctl._cameraMatrix2 = &CMVec[2 * x + 1];
			ctl._distortionCoeffs1 = &DCVec[2 * x];
			ctl._distortionCoeffs2 = &DCVec[2 * x + 1];
			ctl._R = &RVec[x];
			ctl._T = &TVec[x];
			ctl._E = &EVec[x];
			ctl._F = &FVec[x];
			ctl._rms_s = &iteration_rms[x];

			ctl._sample_size = min_images;

			////double intrinsicLeft[9] = { 1553.5714, 0, 0, 0, 1553.5714, 0, 0, 0, 0 };
			//double intrinsicLeft[9] = { 3035.7144, 0, 0, 0, 3035.7144, 0, 0, 0, 0 };
			//double intrinsicRight[9] = { 3035.7144, 0, 0, 0, 3035.7144, 0, 0, 0, 0 };

			//*ctl._cameraMatrix1 = cv::Mat(3, 3, CV_64F, intrinsicLeft).clone();
			//*ctl._cameraMatrix2 = cv::Mat(3, 3, CV_64F, intrinsicRight).clone();


			ctl._status = 1;
			QueueWorkItem(StereoCalibrateIteration, &ctl);
		}


		runMessagePipe();

		for (auto& ctl : controls) {
			if (ctl._rms_s != nullptr) {
				if (max_rms < *ctl._rms_s) {
					max_rms = *ctl._rms_s;
				}
				average_rms += *ctl._rms_s;
				++rms_cnt;
			}
		}
	}



	average_rms /= rms_cnt;
	double barrier_rms = (average_rms + 2 * max_rms) / 3;


	Mat cameraMatrix[2];
	Mat distortionCoeffs[2];

	Mat R, T, E, F;

	R = RVec[0]; T = TVec[0]; E = EVec[0]; F = FVec[0];
	cameraMatrix[0] = CMVec[0]; cameraMatrix[1] = CMVec[1]; distortionCoeffs[0] = DCVec[0]; distortionCoeffs[1] = DCVec[1];

	size_t number_of_significant_iterations = number_of_iterations;

	for (iter_num = 0; iter_num < number_of_iterations; ++iter_num) {
		if (iteration_rms[iter_num] <= barrier_rms) {
			R = RVec[iter_num]; T = TVec[iter_num]; E = EVec[iter_num]; F = FVec[iter_num];
			cameraMatrix[0] = CMVec[iter_num * 2]; cameraMatrix[1] = CMVec[iter_num * 2 + 1]; distortionCoeffs[0] = DCVec[iter_num * 2]; distortionCoeffs[1] = DCVec[iter_num * 2 + 1];
			break;
		}
		else {
			--number_of_significant_iterations;
		}
	}

	for (++iter_num; iter_num < number_of_iterations; ++iter_num) {
		if (iteration_rms[iter_num] <= barrier_rms) {
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
			--number_of_significant_iterations;
		}
	}
	if (number_of_significant_iterations > 0) {
		R = R / (double)number_of_significant_iterations;
		T = T / (double)number_of_significant_iterations;
		E = E / (double)number_of_significant_iterations;
		F = F / (double)number_of_significant_iterations;
		cameraMatrix[0] = cameraMatrix[0] / (double)number_of_significant_iterations;
		cameraMatrix[1] = cameraMatrix[1] / (double)number_of_significant_iterations;
		distortionCoeffs[0] = distortionCoeffs[0] / (double)number_of_significant_iterations;
		distortionCoeffs[1] = distortionCoeffs[1] / (double)number_of_significant_iterations;
	}


	if (g_bTerminated) {
		return 0;
	}


	cv::Size rectified_image_size = g_imageSize;

	Mat Rl, Rr, Pl, Pr, Q;
	cv::Rect Roi[2];
	cv::stereoRectify(cameraMatrix[0], distortionCoeffs[0], cameraMatrix[1], distortionCoeffs[1], g_imageSize, R, T, Rl, Rr, Pl, Pr, Q, 0, g_configuration._calib_rectify_alpha_param, rectified_image_size, &Roi[0], &Roi[1]);


	if (g_bTerminated) {
		return 0;
	}


	Mat map_l[4];
	Mat map_r[4];

	cv::initUndistortRectifyMap(cameraMatrix[0], distortionCoeffs[0], Rl, Pl, rectified_image_size, CV_16SC2/*CV_32F*/, map_l[0], map_l[1]);
	cv::initUndistortRectifyMap(cameraMatrix[1], distortionCoeffs[1], Rr, Pr, rectified_image_size, CV_16SC2/*CV_32F*/, map_r[0], map_r[1]);



	for (auto pos : finalPointsSelection) {
		std::string text = std::to_string(pos + 1);
		ShowUndistortedImageAndPoints(stereoImageRaw_left[pos], points_left[pos], map_l, "Calibr1", text);
		ShowUndistortedImageAndPoints(stereoImageRaw_right[pos], points_right[pos], map_r, "Calibr2", text);
		ProcessWinMessages(1000);

		if (g_bTerminated) {
			return 0;
		}
	}

	FileStorage fs(CalibrationFileName(), FileStorage::WRITE);

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
	//fs << "map_l1" << map_l[0];
	//fs << "map_l2" << map_l[1];
	//fs << "map_r1" << map_r[0];
	//fs << "map_r2" << map_r[1];

	fs << "roi_l" << Roi[0];
	fs << "roi_r" << Roi[1];

	fs << "rectified_image_size" << rectified_image_size;

	fs.release();


	while (ProcessWinMessages()) {}

	ctl->_imagepoints_status = 0;

	return 0;
}



void CalibrateCameras(StereoConfiguration& configuration, SImageAcquisitionCtl& image_acquisition_ctl) {

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
	rootCVWindows(_g_calibrationimages_frame, 4, 0, cv_windows);
	rootCVWindows(_g_calibrationimages_frame, 2, 4, &cv_windows[4]);



	if (g_configuration._file_log == 2) {
		VS_FileLog("", true); // close
	}



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

	while(_g_main_frame->StepBackHistory());
}


