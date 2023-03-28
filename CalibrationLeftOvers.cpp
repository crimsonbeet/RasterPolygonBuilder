double CalibrateSingleCamera(vector<vector<Point2f>>& imagePoints, Mat& cameraMatrix, Mat& distortionCoeffs, int flag = cv::CALIB_RATIONAL_MODEL, bool reprojectPoints = false) {
	double pattern_distance = g_pattern_distance;
	cv::Size boardSize;
	if (imagePoints[0].size() == (g_boardSize.height * g_boardSize.width)) {
		boardSize = g_boardSize;
	}
	else
		if (imagePoints[0].size() == (g_boardQuadSize.height * g_boardQuadSize.width)) {
			boardSize = g_boardQuadSize;
			pattern_distance /= 2.0;
		}
		else
			if (imagePoints[0].size() == (g_boardChessCornersSize.height * g_boardChessCornersSize.width)) {
				boardSize = g_boardChessCornersSize;
				pattern_distance = pattern_distance /= 2.0; //3.0 + 2.5 / 8.0;
			}

	vector<vector<Point3f> > objectPoints(1);
	for (int i = 0; i < boardSize.height; ++i) {
		for (int j = 0; j < boardSize.width; ++j) {
			objectPoints[0].push_back(Point3f(float(j * pattern_distance), float(i * pattern_distance), 0));
		}
	}
	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	cameraMatrix = Mat::eye(3, 3, CV_64F);
	cameraMatrix.at<double>(0, 0) = 1.0;

	distortionCoeffs = Mat::zeros(8, 1, CV_64F); // distortion coefficient matrix. Initialize with zero. 

	vector<Mat> rvecs;
	vector<Mat> tvecs;


	double rms = calibrateCamera(objectPoints, imagePoints, g_imageSize, cameraMatrix, distortionCoeffs, rvecs, tvecs, flag);

	for (size_t j = 0; j < objectPoints.size(); ++j) {
		projectPoints(objectPoints[j], rvecs[j], tvecs[j], cameraMatrix, distortionCoeffs, imagePoints[j]);
	}

	g_objectPoints = objectPoints;

	return rms;
}




std::function<void()> g_boostCalibrateLambda = nullptr;
return_t __stdcall BoostCalibrateWorkItem(LPVOID lp) {
	g_boostCalibrateLambda();
	return 0;
}

void BoostCalibrate(const std::string& msg, size_t number_of_iterations, Mat& cameraMatrixL, Mat& cameraMatrixR, Mat& distortionCoeffsL, Mat& distortionCoeffsR, double rms[2]) {
	g_boostCalibrateLambda = [&]() {
		std::cout << msg << std::endl;

		std::vector<Mat> CMVec2(number_of_iterations * 2), DCVec2(number_of_iterations * 2);
		if (number_of_iterations > 1) {
			vector<vector<Point2f>> imagePoints_left1(g_min_images);
			vector<vector<Point2f>> imagePoints_right1(g_min_images);

			double iteration_rms[256];
			::memset(iteration_rms, 0, sizeof(iteration_rms));

			double average_rms = 0;
			size_t rms_cnt = 0;

			double max_rms = 0;

			size_t iter_num;
			for (iter_num = 0; !g_bTerminated && iter_num < number_of_iterations; ++iter_num) {
				SampleImagepoints(g_min_images, imagePoints_left, imagePoints_left1);
				SampleImagepoints(g_min_images, imagePoints_right, imagePoints_right1);

				try {
					rms[0] = CalibrateSingleCamera(imagePoints_left1, CMVec2[iter_num * 2], DCVec2[iter_num * 2]);
					rms[1] = CalibrateSingleCamera(imagePoints_right1, CMVec2[iter_num * 2 + 1], DCVec2[iter_num * 2 + 1]);

					iteration_rms[iter_num] = std::max(rms[0], rms[1]);
					if (max_rms < iteration_rms[iter_num]) {
						max_rms = iteration_rms[iter_num];
					}
					average_rms += iteration_rms[iter_num];
					++rms_cnt;
				}
				catch (...) {
					rms[0] = rms[1] = 100;
				}

				std::cout << iter_num << ' ' << "Pre-Re-projection error for left camera: " << rms[0] << std::endl;
				std::cout << iter_num << ' ' << "Pre-Re-projection error for right camera: " << rms[1] << std::endl;
			}
			average_rms /= rms_cnt;
			average_rms = (average_rms + max_rms) / 2;

			const size_t max_iter = iter_num;

			for (iter_num = 0; iter_num < max_iter; ++iter_num) {
				if (iteration_rms[iter_num] <= average_rms) {
					cameraMatrixL = CMVec2[iter_num * 2]; cameraMatrixR = CMVec2[iter_num * 2 + 1]; distortionCoeffsL = DCVec2[iter_num * 2]; distortionCoeffsR = DCVec2[iter_num * 2 + 1];
					break;
				}
				else {
					--number_of_iterations;
				}
			}

			for (++iter_num; iter_num < max_iter; ++iter_num) {
				if (iteration_rms[iter_num] <= average_rms) {
					cameraMatrixL = cameraMatrixL + CMVec2[iter_num * 2];
					cameraMatrixR = cameraMatrixR + CMVec2[iter_num * 2 + 1];
					distortionCoeffsL = distortionCoeffsL + DCVec2[iter_num * 2];
					distortionCoeffsR = distortionCoeffsR + DCVec2[iter_num * 2 + 1];
				}
				else {
					--number_of_iterations;
				}
			}
			if (number_of_iterations > 0) {
				cameraMatrixL = cameraMatrixL / (double)number_of_iterations;
				cameraMatrixR = cameraMatrixR / (double)number_of_iterations;
				distortionCoeffsL = distortionCoeffsL / (double)number_of_iterations;
				distortionCoeffsR = distortionCoeffsR / (double)number_of_iterations;
			}
		}

		if (number_of_iterations <= 1 && !g_bTerminated) {
			vector<vector<Point2f>> imagePoints_left1(imagePoints_left.size());
			vector<vector<Point2f>> imagePoints_right1(imagePoints_right.size());

			SampleImagepoints(imagePoints_left.size(), imagePoints_left, imagePoints_left1);
			SampleImagepoints(imagePoints_right.size(), imagePoints_right, imagePoints_right1);

			rms[0] = CalibrateSingleCamera(imagePoints_left1, cameraMatrixL, distortionCoeffsL);
			rms[1] = CalibrateSingleCamera(imagePoints_right1, cameraMatrixR, distortionCoeffsR);

			std::cout << "Pre-Re-projection error for left camera: " << rms[0] << std::endl;
			std::cout << "Pre-Re-projection error for right camera: " << rms[1] << std::endl;
		}

		g_boostCalibrateLambda = nullptr;
	};

	QueueWorkItem(BoostCalibrateWorkItem);

	while (g_boostCalibrateLambda != nullptr) {
		ProcessWinMessages(10);
	}

}




bool ReEvaluateUndistortImagePoints(vector<Mat>& imageRaw, vector<vector<Point2d>>& imagePoints, Mat& cameraMatrix, Mat& distortionCoeffs, SImageAcquisitionCtl& ctl, std::string& cv_window, vector<vector<Point2d>>* imagePoints_paired = 0, vector<Mat>* imageRaw_paired = 0) {
	Mat map[2];

	cv::initUndistortRectifyMap(cameraMatrix, distortionCoeffs, cv::Mat(), cv::getOptimalNewCameraMatrix(cameraMatrix, distortionCoeffs, g_imageSize, 0), g_imageSize, CV_32FC2, map[0], map[1]);

	for (int j = 0; j < imagePoints.size(); ++j) {
		auto& image = imageRaw[j];
		image = ShowUndistortedImageAndPoints(image, imagePoints[j], map, cv_window, std::to_string(j + 1));

		if (!buildPointsFromImages(&image, &imagePoints[j], 1, ctl, g_configuration._calib_min_confidence, 2)) {
			std::cout << "image number " << j << " has failed" << std::endl;
			imagePoints.erase(imagePoints.begin() + j);
			imageRaw.erase(imageRaw.begin() + j);
			if (imagePoints_paired) {
				imagePoints_paired->erase(imagePoints_paired->begin() + j);
			}
			if (imageRaw_paired) {
				imageRaw_paired->erase(imageRaw_paired->begin() + j);
			}
			--j;
		}
	}
	return imagePoints.size() >= g_min_images;
}


