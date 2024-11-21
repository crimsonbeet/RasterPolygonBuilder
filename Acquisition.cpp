#include "stdafx.h"
//#include <opencv2/core/hal/hal.hpp>
#include <Mmsystem.h>
#pragma comment(lib, "Winmm.lib")

#include "XClasses.h"
#include "XAndroidCamera.h"

#include "XConfiguration.h"


const unsigned int g_bytedepth_scalefactor = 256;


extern bool g_bTerminated;
extern bool g_bRestart; 

extern HANDLE g_event_SFrameIsAvailable;


void RGB_TO_HSV(Vec<uchar, 3> rgb, double hsv[3], const double maxMinThreshold) {
	BYTE chMax = rgb[0];
	BYTE chMin = chMax;
	std::for_each(&rgb[1], &rgb[2] + 1, [&chMax, &chMin](BYTE& ch) {
		if (ch > chMax) {
			chMax = ch;
		}
		if (ch < chMin) {
			chMin = ch;
		}
	});

	const double chMaxMin = (chMax - chMin);
	if (chMaxMin <= maxMinThreshold) {
		hsv[0] = 361;
		hsv[1] = 0;
		hsv[2] = chMax;
		return;
	}

	double s = chMaxMin / (chMax == 0 ? 1 : chMax);

	double rc = (chMax - rgb[0]) / chMaxMin;
	double gc = (chMax - rgb[1]) / chMaxMin;
	double bc = (chMax - rgb[2]) / chMaxMin;

	double h;

	if (rgb[0] == chMax) {
		h = bc - gc; // color between yellow and magenta (red in the middle)
	}
	else
	if (rgb[1] == chMax) {
		h = 2 + rc - bc; // color between cyan and yellow (green in the middle)
	}
	else {
		h = 4 + gc - rc; // color between magenta and cyan (blue in the middle)
	}

	h *= 60;
	if (h < 0) {
		h += 360;
	}

	if (isnan(h)) {
		h = 361;
	}

	hsv[0] = h;
	hsv[1] = s;
	hsv[2] = chMax;
}

double GetFScore(const cv::Vec<uchar, 3>& ch1, const cv::Vec<uchar, 3>& ch2) {
	double fscore = 1;
	for (int j = 0; j < 3; ++j) {
		if (ch1[j] > 0 && ch2[j] > 0) {
			const double a = ch1[j];
			const double b = ch2[j];
			const double div = a*a + b*b;
			const double num = a*b;
			if (div > 100 && num > 10) {
				fscore *= (2.0 * num) / div;
			}
			else {
				fscore *= 0.01;
			}
		}
	}
	if (isnan(fscore)) {
		fscore = 0;
	}
	return fscore;
}

const double log2_log2_lookup_table[100] = { -2.82849948, -2.82849948, -2.82849948, -2.82849948, -2.82849948, -2.57236354, -2.35681546, -2.17096150, -2.00777528, -1.86245950, -1.73159266, -1.61264865, -1.50370928, -1.40328369, -1.31019016, -1.22347616, -1.14236271, -1.06620465, -0.99446161, -0.92667657, -0.86245950, -0.80147493, -0.74343220, -0.68807788, -0.63518961, -0.58457129, -0.53604909, -0.48946818, -0.44469009, -0.40159049, -0.36005732, -0.31998923, -0.28129427, -0.24388875, -0.20769633, -0.17264710, -0.13867696, -0.10572697, -0.07374279, -0.04267425, -0.01247491, 0.01689828, 0.04548532, 0.07332343, 0.10044728, 0.12688928, 0.15267967, 0.17784682, 0.20241730, 0.22641605, 0.24986653, 0.27279081, 0.29520968, 0.31714277, 0.33860860, 0.35962468, 0.38020759, 0.40037301, 0.42013583, 0.43951018, 0.45850944, 0.47714639, 0.49543313, 0.51338122, 0.53100165, 0.54830491, 0.56530101, 0.58199950, 0.59840951, 0.61453977, 0.63039862, 0.64599407, 0.66133376, 0.67642504, 0.69127494, 0.70589022, 0.72027734, 0.73444255, 0.74839182, 0.76213091, 0.77566534, 0.78900044, 0.80214134, 0.81509297, 0.82786010, 0.84044732, 0.85285903, 0.86509953, 0.87717292, 0.88908319, 0.90083418, 0.91242961, 0.92387306, 0.93516801, 0.94631783, 0.95732575, 0.96819492, 0.97892841, 0.98952914, 1.00000000 };

// calculate (log2(64*log2(x)) - 5)
// x must be in closed range [1,2]
double lookup_log2_64_log2_1_v_5(double x) { 
	size_t i = x * 100 - 101;

	if (i < 0) 
		i = 0;
	else
	if (i > 99) 
		i = 99;

	return log2_log2_lookup_table[i];
}

double GetEScore(const cv::Vec<uchar, 3>& ch1, const cv::Vec<uchar, 3>& ch2) {
	double escore = 0;
	double e[3];
	double w[3];
	double total_sum = 0;
	for (int j = 0; j < 3; ++j) {
		const int a = ch1[j];
		const int b = ch2[j];
		const double div = std::max(a, b);
		if (div > 0) {
			const double p = std::min(a, b) / div;
			//e[j] = p*(approx_log2(32*(1 + p)) - 5);
			// 
			//e[j] = p * (log2(64*log2(1 + p)) - 5);
			e[j] = p * lookup_log2_64_log2_1_v_5(1 + p);
		}
		else {
			e[j] = 0;
		}
		w[j] = div;
		total_sum += div;
	}
	for (int j = 0; j < 3; ++j) {
		escore += e[j] * w[j] / total_sum;
	}
	if (isnan(escore)) {
		escore = 0;
	}
	return escore;
}


double GetAngleDifferenceInGrads(double x, double y) {
	double a1 = x - y;
	double a2 = a1 > 0 ? a1 - 360 : 360 + a1;

	return std::abs(a2) > std::abs(a1) ? a1 : a2;
}


void ConvertColoredImage2Mono(Mat& image, double chWeights[3], std::function<double(double)> convert) {
	cv::Mat aux(image.size(), CV_16UC1);
	typedef Vec<uchar, 3> Vec3c;
	for (int r = 0; r < aux.rows; ++r) {
		for (int c = 0; c < aux.cols; ++c) {
			Vec3c& pixVec = image.at<Vec3c>(r, c);
			ushort& pix = aux.at<ushort>(r, c);
			pix = 0;
			for (int j = 0; j < 3; ++j) {
				pix += (ushort)std::floor(convert(pixVec[j]) * chWeights[j] + 0.5);
			}
		}
	}
	image = aux.clone();
}

void StandardizeImage(Mat& image, double chWeights[3]) {
	if (image.type() == CV_8UC3) {
		ConvertColoredImage2Mono(image, chWeights, [](double ch) {
			return std::min(ch * 256, 256.0 * 256.0);
		});
	}
	if (image.type() != CV_16UC1) {
		if (image.type() != CV_8UC1) {
			image.clone().convertTo(image, CV_8UC1);
		}
		image.clone().convertTo(image, CV_16UC1);
		image *= (size_t)256;
	}
}
void SquareImage(Mat& image, double chWeights[3]) {
	if (image.type() == CV_8UC3) {
		ConvertColoredImage2Mono(image, chWeights, [](double ch) {
			return pow(ch, 1.5);
		});
	}
	if (image.type() != CV_16UC1) {
		if (image.type() != CV_8UC1) {
			image.clone().convertTo(image, CV_8UC1);
		}
		image.clone().convertTo(image, CV_16UC1);
		cv::Mat aux;
		multiply(image, image, aux);
		image = aux.clone();
	}
}

template<typename T, typename S>
double cosineDistance3d(const T& left, const S& right) {
	double dot = 0; 
	for (int j = 0; j < 3; ++j) {
		double r = (double)right[j] - left[j];
		dot += approx_log2(abs(r)) * (double)left[j];
	}
	return dot;
}

template<typename T>
void ConvertColoredImage2Mono_CosineLikeness(Mat& image, T chIdeal[3], std::function<double(uchar)> convert) {
	cv::Mat aux(image.size(), CV_16UC1);
	typedef Vec<uchar, 3> Vec3c;
	for (int r = 0; r < aux.rows; ++r) {
		for (int c = 0; c < aux.cols; ++c) {
			Vec3c& pixVec = image.at<Vec3c>(r, c);
			double w = cosineDistance3d(chIdeal, pixVec) / 256;
			ushort& pix = aux.at<ushort>(r, c);
			pix = (ushort)std::floor(convert(w) + 0.5); ;
		}
	}
	image = aux.clone();
}



void BuildWeights_ByChannel(Mat& image, Point& pt, double weights_out[3]) {
	if (image.type() == CV_8UC3) {
		double weights[3] = { 0, 0, 0 };
		double sum = 0;
		typedef Vec<uchar, 3> Vec3c;
		for (int r = pt.y - 2; r < image.rows && r < pt.y + 3; ++r) {
			for (int c = pt.x - 2; c < image.cols && c < pt.x + 3; ++c) {
				if (r < 0 || c < 0) {
					continue;
				}
				Vec3c& pixVec = image.at<Vec3c>(r, c);
				for (int j = 0; j < 3; ++j) {
					double w = pow(pixVec[j], 1.5);
					sum += w;
					weights[j] += w;
				}
			}
		}
		double wMax = std::numeric_limits<double>::min();

		for (auto& w : weights) {
			w /= sum;
			if (w > wMax) {
				wMax = w;
			}
		}

		for (auto& w : weights) w /= wMax;

		memcpy(weights_out, weights, sizeof(weights));
	}
}



bool BuildIdealChannels_Distribution(Mat& image, Point& pt, Mat& mean, Mat& stdDev, Mat& factorLoadings, Mat& invCovar, Mat& invCholesky, int neighbourhoodRadius) {
	if (image.type() == CV_8UC3) {
		if (neighbourhoodRadius > 5) {
			neighbourhoodRadius = 5;
		}
		Mat_<double> neighbours(neighbourhoodRadius <= 2? 25: neighbourhoodRadius == 3 ? 49: neighbourhoodRadius == 4 ? 81: 121, 3);
		int yStart = pt.y - neighbourhoodRadius;
		int yEnd = pt.y + neighbourhoodRadius + 1;
		while (yStart < 0) ++yStart, ++yEnd;
		while (yEnd > image.rows) --yStart, --yEnd;
		int xStart = pt.x - neighbourhoodRadius;
		int xEnd = pt.x + neighbourhoodRadius + 1;
		while (xStart < 0) ++xStart, ++xEnd;
		while (xEnd > image.cols) --xStart, --xEnd;
		int cnt = 0;
		typedef Vec<uchar, 3> Vec3c;
		for (int r = yStart; r < yEnd; ++r) {
			for (int c = xStart; c < xEnd; ++c) {
				Vec3c& pixVec = image.at<Vec3c>(r, c);
				for (int j = 0; j < 3; ++j) {
					neighbours(cnt, j) = pixVec[j];
				}
				++cnt;
			}
		}
		Mat Q;
		cv::calcCovarMatrix(neighbours, Q, mean, CovarFlags::COVAR_NORMAL | CovarFlags::COVAR_ROWS, CV_64F);


		double deviation[3];
		deviation[0] = sqrt(Q.at<double>(0, 0));
		deviation[1] = sqrt(Q.at<double>(1, 1));
		deviation[2] = sqrt(Q.at<double>(2, 2));

		double loadings[3];
		loadings[1] = sqrt(Q.at<double>(1, 0) * Q.at<double>(2, 1) / Q.at<double>(2, 0)); // l(1) = cov(1, 0) * cov(2, 1) / cov(2, 0)
		loadings[0] = Q.at<double>(1, 0) / loadings[1]; // l(0) = cov(1, 0) / l(1)
		loadings[2] = (Q.at<double>(2, 0) / Q.at<double>(1, 0)) * loadings[1]; // l(2) = (cov(2, 0) / cov(1, 0)) * l(1)

		stdDev = Mat_<double>(3, 1);
		stdDev.at<double>(0, 0) = deviation[0];
		stdDev.at<double>(1, 0) = deviation[1];
		stdDev.at<double>(2, 0) = deviation[2];

		factorLoadings = Mat_<double>(3, 1);
		factorLoadings.at<double>(0, 0) = loadings[0];
		factorLoadings.at<double>(1, 0) = loadings[1];
		factorLoadings.at<double>(2, 0) = loadings[2];


		double invConditionNumber = cv::invert(Q.clone(), invCovar, DECOMP_SVD);
		std::cout << "Covar inverse condition number " << invConditionNumber << std::endl;
		if (invConditionNumber > 0.01) {
			if (cv::Cholesky(&Q.at<double>(0, 0), (size_t)Q.step, Q.rows, nullptr, 0, 0)) {
				for (int i = 0; i < 2; ++i) {
					for (int j = i + 1; j < 3; ++j) {
						Q.at<double>(i, j) = 0;;
					}
				}
				return cv::invert(Q, invCholesky, DECOMP_LU) != 0.0;
			}
		}
	}

	return false;
}

void BuildIdealChannels_Likeness(Mat& image, Point& pt, double chIdeal[3], const int height) {
	if (image.type() == CV_8UC3) {
		double likeness[3] = { 0, 0, 0 };
		double cnt = 0;
		typedef Vec<uchar, 3> Vec3c;
		const int width = height * 2;
		for (int r = pt.y - height; r < image.rows && r < pt.y + height + 1; ++r) {
			for (int c = pt.x - width; c < image.cols && c < pt.x + width + 1; ++c) {
				if (r < 0 || c < 0) {
					continue;
				}
				++cnt;
				Vec3c& pixVec = image.at<Vec3c>(r, c);
				for (int j = 0; j < 3; ++j) {
					likeness[j] += ((double)pixVec[j]);
				}
			}
		}

		for (auto& w : likeness) {
			w /= cnt;
		}

		memcpy(chIdeal, likeness, sizeof(likeness));
	}
}

double Get_Squared_Z_Score(const cv::Vec<uchar, 3>& pixOrig, double mean_data[3], double invCholesky_data[3][3]) {
	double pix_data[3] = { pixOrig(0) - mean_data[0], pixOrig(1) - mean_data[1], pixOrig(2) - mean_data[2] };

	double pix_norm_data[3] = { 
		invCholesky_data[0][0] * pix_data[0],
		invCholesky_data[1][0] * pix_data[0] + invCholesky_data[1][1] * pix_data[1],
		invCholesky_data[2][0] * pix_data[0] + invCholesky_data[2][1] * pix_data[1] + invCholesky_data[2][2] * pix_data[2] 
	};

	double sum = 0;
	sum += pix_norm_data[0] * pix_norm_data[0];
	sum += pix_norm_data[1] * pix_norm_data[1];
	sum += pix_norm_data[2] * pix_norm_data[2];

	return sum;
}

void ConvertColoredImage2Mono_Likeness(cv::Mat& image, cv::Mat mean/*rgb*/, Mat& stdDev, Mat& factorLoadings, cv::Mat invCovar/*inverted covariance of colors*/, Mat invCholesky) {
	typedef Vec<uchar, 3> Vec3c;

	double* mean_data = (double*)mean.data;

	double invCholesky_data[3][3] = {
		{ invCholesky.at<double>(0, 0), invCholesky.at<double>(0, 1), invCholesky.at<double>(0, 2) },
		{ invCholesky.at<double>(1, 0), invCholesky.at<double>(1, 1), invCholesky.at<double>(1, 2) },
		{ invCholesky.at<double>(2, 0), invCholesky.at<double>(2, 1), invCholesky.at<double>(2, 2) }
	};

	cv::Mat aux(image.size(), CV_16UC1);

	for (int r = 0; r < aux.rows; ++r) {
		for (int c = 0; c < aux.cols; ++c) {
			double zScore = Get_Squared_Z_Score(image.at<cv::Vec<uchar, 3>>(r, c), mean_data, invCholesky_data);
			if (zScore < 9) {
				aux.at<ushort>(r, c) = (9 - zScore) * 26 + 0.5;
			}
			else {
				aux.at<ushort>(r, c) = 0;
			}
		}
	}

	image = aux.clone();
}

double hsvLikenessScore(cv::Vec<uchar, 3>& pixOriginal, double hsvIdeal[3]) {
	double hsvOriginal[3];
	RGB_TO_HSV(pixOriginal, hsvOriginal);

	if (hsvOriginal[0] > 360) {
		return 0; // hue is undefined
	}

	double diff = std::abs(GetAngleDifferenceInGrads(hsvIdeal[0], hsvOriginal[0]));
	if (diff > 90) {
		diff = 90;
	}

	const double tanThreshold = 0.5;
	double hueLikeness = tanThreshold - std::abs(std::tan(CV_PI * diff / 180));

	double likeness;
	if (hueLikeness < 0) {
		likeness = 0;
	}
	else {
		likeness = hueLikeness / tanThreshold;


		const double saturationIdeal = hsvIdeal[1];
		const double valueIdeal = hsvIdeal[2];
		const double saturationOriginal = hsvOriginal[1];
		const double valueOriginal = hsvOriginal[2];

		likeness *= std::min(saturationIdeal, saturationOriginal) / std::max(saturationIdeal, saturationOriginal);
		likeness *= std::min(valueIdeal, valueOriginal) / std::max(valueIdeal, valueOriginal);

		likeness *= 256;
	}

	if (isnan(likeness) || likeness < 0 || likeness > 256) {
		likeness = 0;
	}

	return likeness;
}

bool ConvertColoredImage2Mono_HSV_Likeness(Mat& image, double rgbIdeal[3]) {
	double hsvIdeal[3];
	double hsvOriginal[3];
	
	typedef cv::Vec<uchar, 3> Vec3c;
	Vec3c pixIdeal;
	pixIdeal[0] = (uchar)(rgbIdeal[0] + 0.5);
	pixIdeal[1] = (uchar)(rgbIdeal[1] + 0.5);
	pixIdeal[2] = (uchar)(rgbIdeal[2] + 0.5);
	
	RGB_TO_HSV(pixIdeal, hsvIdeal);

	const double saturationIdeal = hsvIdeal[1];
	const double valueIdeal = hsvIdeal[2];

	const double y_componentIdeal = rgbIdeal[0] * 0.114 + rgbIdeal[1] * 0.587 + rgbIdeal[2] * 0.299;

	if (y_componentIdeal < 40) {
		return false;
	}
	if (hsvIdeal[0] > 360) {
		return false;
	}

	double max_likeness = std::numeric_limits<double>::min();

	cv::Mat aux(image.size(), CV_16UC1);

	cv::blur(image.clone(), image, cv::Size(3, 3));

	for (int r = 0; r < aux.rows; ++r) {
		for (int c = 0; c < aux.cols; ++c) {
			Vec3c& pixOriginal = image.at<Vec3c>(r, c);

			double likeness = hsvLikenessScore(pixOriginal, hsvIdeal);

			aux.at<ushort>(r, c) = likeness + 0.5;

			if (max_likeness < likeness) {
				max_likeness = likeness;
			}
		}
	}
	image = aux.clone();

	return true;
}

void StandardizeImage_Likeness(Mat& image, Mat mean/*rgb*/, Mat& stdDev, Mat& factorLoadings, Mat invCovar/*inverted covariance of colors*/, Mat invCholesky) {
	if (image.type() == CV_8UC3) {
		ConvertColoredImage2Mono_Likeness(image, mean/*rgb*/, stdDev, factorLoadings, invCovar/*inverted covariance of colors*/, invCholesky);
		return;
	}
	if (image.type() != CV_16UC1) {
		if (image.type() != CV_8UC1) {
			image.clone().convertTo(image, CV_8UC1);
		}
		image.clone().convertTo(image, CV_16UC1);
		image *= (size_t)256;
	}
}

bool StandardizeImage_HSV_Likeness(Mat& image, double rgbIdeal[3]) {
	if (image.type() == CV_8UC3) {
		return ConvertColoredImage2Mono_HSV_Likeness(image, rgbIdeal);
	}
	if (image.type() != CV_16UC1) {
		if (image.type() != CV_8UC1) {
			image.clone().convertTo(image, CV_8UC1);
		}
		image.clone().convertTo(image, CV_16UC1);
		image *= (size_t)256;
	}

	return true;
}




bool GetImagesFromFile(Mat& left_image, Mat& right_image, std::vector<cv::Point2d>& pointsLeft, std::vector<cv::Point2d>& pointsRight, const std::string& current_N) {
	std::string nl = current_N;
	std::string nr = current_N;

	//std::string nl = std::to_string(current_N);
	//std::string nr = std::to_string(current_N);

	//left_image = imread(std::string(CalibrationDirName()) + nl + 'l' + ".bmp");
	//right_image = imread(std::string(CalibrationDirName()) + nr + 'r' + ".bmp");


	left_image.resize(0);
	right_image.resize(0);

	pointsLeft.resize(0);
	pointsRight.resize(0);

	std::string xmlFilename = std::string(CalibrationDirName()) + current_N + ".xml";

	std::cout << "Getting content from " << xmlFilename << std::endl;

	try {
		FileStorage fs(xmlFilename, FileStorage::READ);
		if (fs.state != 0) {
			std::cout << "Got content" << std::endl;
			fs["left_image"] >> left_image;
			fs["right_image"] >> right_image;
			fs["left_points"] >> pointsLeft;
			fs["right_points"] >> pointsRight;
			fs.release();
			std::cout << "Got images" << std::endl;
		}
	}
	catch (...) {

	}

	if (left_image.rows == 0 || left_image.cols == 0) {
		left_image = imread(std::string(CalibrationDirName()) + nl + 'l' + "-chess.png", ImreadModes::IMREAD_ANYDEPTH); // Mar.4 2015.
	}
	if(right_image.rows == 0 || right_image.cols == 0) {
		right_image = imread(std::string(CalibrationDirName()) + nr + 'r' + "-chess.png", ImreadModes::IMREAD_ANYDEPTH); // Mar.4 2015.
	}

	if (left_image.rows == 0 || left_image.cols == 0) {
		left_image = imread(std::string(CalibrationDirName()) + nl + 'l' + ".png", ImreadModes::IMREAD_ANYDEPTH | ImreadModes::IMREAD_ANYCOLOR); // Mar.4 2015.
	}
	if (right_image.rows == 0 || right_image.cols == 0) {
		right_image = imread(std::string(CalibrationDirName()) + nr + 'r' + ".png", ImreadModes::IMREAD_ANYDEPTH | ImreadModes::IMREAD_ANYCOLOR); // Mar.4 2015.
	}

	if (left_image.rows == 0 || left_image.cols == 0) {
		//left_image = imread(std::string(g_path_nwpu_images_dir) + current_N + ".jpg", CV_LOAD_IMAGE_ANYDEPTH);
		left_image = imread(std::string(g_path_nwpu_images_dir) + current_N + ".jpg", ImreadModes::IMREAD_ANYDEPTH | ImreadModes::IMREAD_ANYCOLOR);
	}
	if (right_image.rows == 0 || right_image.cols == 0) {
		//right_image = imread(std::string(g_path_nwpu_images_dir) + current_N + ".jpg", CV_LOAD_IMAGE_ANYDEPTH);
		right_image = imread(std::string(g_path_nwpu_images_dir) + current_N + ".jpg", ImreadModes::IMREAD_ANYDEPTH | ImreadModes::IMREAD_ANYCOLOR);
	}

	if(left_image.rows <= 10 || right_image.rows <= 10 || left_image.rows != right_image.rows) {
		return false;
	}
	if(left_image.cols <= 10 || right_image.cols <= 10 || left_image.cols != right_image.cols) {
		return false;
	}

	//if (!g_bTerminated) {
	//	g_lastwritten_sframe.gate.lock();
	//	__int64 time_now = OSDayTimeInMilliseconds();
	//	for (int j = 0; j < NUMBER_OF_CAMERAS; ++j) {
	//		g_lastwritten_sframe.frames[j].timestamp = time_now;
	//		g_lastwritten_sframe.frames[j].camera_index = j;
	//	}
	//	g_lastwritten_sframe.local_timestamp = time_now;
	//	g_lastwritten_sframe.isActive = false;
	//	Mat* images[2] = { &g_lastwritten_sframe.frames[0].cv_image, &g_lastwritten_sframe.frames[NUMBER_OF_CAMERAS - 1].cv_image };
	//	matCV_16UC1_memcpy(*images[0], left_image);
	//	matCV_16UC1_memcpy(*images[1], right_image);
	//	g_lastwritten_sframe.gate.unlock();

	//	if (g_event_SFrameIsAvailable != INVALID_HANDLE_VALUE) {
	//		SetEvent(g_event_SFrameIsAvailable);
	//	}
	//}

	return true;
}





void CopyStereoFrame(Mat& left, Mat& right, SStereoFrame* pframe, int64* time_received) {
	// the following code copies the images from frames to the output matrices.
	// the matrices get initalized if they do not match the dimensions. this should happen just one time. 
	// the idea is to initialize the matrices once, and then re-use them after that. 
	Mat* dst[2] = { &left, &right };
	std::vector<int> idx = pframe->GetIdxs();
	int j = 0;
	for (auto dst : dst) {
		Mat& image = pframe->frames[idx[j++]].cv_image;
		if (image.type() == CV_8UC3) {
			matCV_8UC3_memcpy(*dst, image);
		}
		else {
			matCV_16UC1_memcpy(*dst, image);
		}
	}
	if (time_received != NULL) {
		*time_received = pframe->local_timestamp;
	}
}





double g_frameRate[4]; // reported frame rate by the cameras. 
double g_resultingFrameRate;


wsi_gate g_rotatingbuf_gate;
std::vector<SStereoFrame> g_rotating_buf;
const size_t g_rotating_buf_size = 10;
int g_nextwrite_inbuf;
int g_nextread_inbuf;


SStereoFrame g_lastwritten_sframe;


extern HANDLE g_event_SFrameIsAvailable;


SStereoFrame& NextWriteFrame() {
	g_rotatingbuf_gate.lock();
	if (g_rotating_buf.size() == 0) {
		g_rotating_buf.resize(g_rotating_buf_size);
	}
	SStereoFrame& sframe = g_rotating_buf[g_nextwrite_inbuf++];
	if (g_nextwrite_inbuf >= (int)g_rotating_buf.size()) {
		g_nextwrite_inbuf = 0;
	}
	if (g_nextread_inbuf == g_nextwrite_inbuf) {
		if (++g_nextread_inbuf >= (int)g_rotating_buf.size()) {
			g_nextread_inbuf = 0;
		}
	}
	sframe.gate.lock();
	g_rotatingbuf_gate.unlock();

	return sframe;
}

SStereoFrame* NextReadFrame() {
	SStereoFrame* pframe = 0;

	g_rotatingbuf_gate.lock();
	if (g_rotating_buf.size() > 0 && g_nextread_inbuf != g_nextwrite_inbuf) {
		pframe = &g_rotating_buf[g_nextread_inbuf++];
		pframe->gate.lock();
		if (g_nextread_inbuf >= (int)g_rotating_buf.size()) {
			g_nextread_inbuf = 0;
		}
	}
	g_rotatingbuf_gate.unlock();

	return pframe;
}

bool GetLastFrame(Mat& left, Mat& right, int64_t* time_received, int64_t expiration) {
	int64_t start_time = OSDayTimeInMilliseconds();

	int count = 0;

	SStereoFrame* pframe = NULL;
	while (!g_bTerminated) {
		if (g_lastwritten_sframe.isActive) {
			g_lastwritten_sframe.gate.lock();
			if (g_lastwritten_sframe.isActive) {
				CopyStereoFrame(left, right, &g_lastwritten_sframe, time_received);
				g_lastwritten_sframe.isActive = false;
				++count;
			}
			g_lastwritten_sframe.gate.unlock();
		}
		if (count > 0) {
			break;
		}
		if (expiration) {
			if ((OSDayTimeInMilliseconds() - start_time) > expiration) {
				break;
			}
		}
		if (g_event_SFrameIsAvailable != INVALID_HANDLE_VALUE && g_resultingFrameRate > 0) {
			WaitForSingleObjectEx(g_event_SFrameIsAvailable, (1000 / (int)g_resultingFrameRate) + 1, TRUE);
		}
		else {
			OSWait();
		}
	}

	bool ok = !left.empty() && !right.empty();
	return ok;
}




#pragma warning ( push )
#pragma warning ( disable : 6201 ) // Index 'index-name' is out of valid index range 'minimum' to 'maximum' for possibly stack allocated buffer 'variable'
bool GetImages(Mat& left, Mat& right, int64_t* time_received, const int N/*min_frames_2_consider*/, int64_t expiration) {
	__int64 start_time = OSDayTimeInMilliseconds();

	int count = 0;

	int64_t minimal_time_difference = std::numeric_limits<int64_t>::max();
	SStereoFrame* pframe = NULL;
	while (!g_bTerminated) {
		if (count >= N) {
			break;
		}
		if ((pframe = NextReadFrame()) == NULL) {
			if (expiration) {
				if ((OSDayTimeInMilliseconds() - start_time) > expiration) {
					break;
				}
			}
			if (g_event_SFrameIsAvailable != INVALID_HANDLE_VALUE && g_resultingFrameRate > 0) {
				WaitForSingleObjectEx(g_event_SFrameIsAvailable, (1000 / (int)g_resultingFrameRate) + 1, TRUE);
			}
			else {
				OSWait();
			}
			continue;
		}
		if (pframe->isActive) {
			++count;
		}

		std::vector<int> idx = pframe->GetIdxs();
		int64_t dif = max(pframe->frames[idx[0]].timestamp, pframe->frames[idx[1]].timestamp) - min(pframe->frames[idx[0]].timestamp, pframe->frames[idx[1]].timestamp);
		int64_t time_difference = std::numeric_limits<int64_t>::max();
		if (dif < time_difference) {
			time_difference = dif;
		}

		if (pframe->isActive && time_difference < minimal_time_difference) {
			minimal_time_difference = time_difference;
			CopyStereoFrame(left, right, pframe, time_received);
			pframe->isActive = false;
		}
		pframe->gate.unlock();
	}

	bool ok = count > 0 && !left.empty() && !right.empty();
	return ok;
}
#pragma warning ( pop )






void TransferImage2StereoFrame(Mat &image, bool isFirst) {
	SStereoFrame& sframe = g_lastwritten_sframe;
	sframe.gate.lock();
	std::vector<int> idxs = sframe.GetIdxs();

	sframe.frames[idxs[isFirst ? 0 : 1]].cv_image = image;

	if (sframe.isInTransition) {
		SStereoFrame& sNframe = NextWriteFrame(); // also locks the frame
		sNframe = sframe;
		sNframe.isActive = true;
		sNframe.isInTransition = false;
		sNframe.gate.unlock();

		sframe.isActive = true;
		sframe.isInTransition = false;

		sframe.gate.unlock();

		if (g_event_SFrameIsAvailable != INVALID_HANDLE_VALUE) {
			SetEvent(g_event_SFrameIsAvailable);
		}
	}
	else {
		sframe.isInTransition = true;
		sframe.isActive = false;

		sframe.gate.unlock();
	}
}


AndroidCameraRaw10Image* ProcvessRaw10Image(AndroidCameraRaw10Image* obj) {
	uint8_t *ptr = obj->_buffers[0]._buffer.data();

	return obj;
}

void Process_CameraBayerFilterImage(AndroidBayerFilterImage* obj) {
	std::vector<int16_t> rawImage;
	obj->get_image(rawImage);

	Mat image(cv::Size(obj->_width, obj->_height), CV_16UC1, rawImage.data());

	Mat frame;  
	cv::demosaicing(image, frame, cv::COLOR_BayerBGGR2RGB, 3);

	frame -= 64;
	frame.convertTo(image, CV_8UC3);

	TransferImage2StereoFrame(image, obj->_isFirst);
}

void Process_CameraJpegImage(AndroidCameraJpegImage* obj) {
	Mat image = cv::imdecode(obj->_cont._buffer, cv::IMREAD_COLOR);

	TransferImage2StereoFrame(image, obj->_isFirst);
}


void ConvertSynchronizedResults(std::vector<CGrabResultPtr>& ptrGrabResults, SStereoFrame& sframe, std::string* cv_winNames = NULL) {
	for (auto& ptrGrabResult : ptrGrabResults) {
		int width = ptrGrabResult->GetWidth();
		int height = ptrGrabResult->GetHeight();

		CPylonImage pylon_bitmapImage;

		ushort* pylon_buffer = NULL;

		if (ptrGrabResult->GetPixelType() != PixelType_BGR8packed) { // PixelType_RGB16packed might be a better choice
			CImageFormatConverter converter;
			converter.OutputPixelFormat = PixelType_BGR8packed;
			converter.Convert(pylon_bitmapImage, ptrGrabResult);

			pylon_buffer = (ushort*)pylon_bitmapImage.GetBuffer();
		}
		else {
			pylon_buffer = (ushort*)ptrGrabResult->GetBuffer();
		}

		int c = (int)ptrGrabResult->GetCameraContext();
		if (c < 0) {
			g_bTerminated = true;
			break;
		}

		if (sframe.frames[c].cv_image.rows != height || sframe.frames[c].cv_image.cols != width) {
			sframe.frames[c].cv_image = Mat(height, width, CV_8UC3);
		}

		typedef Vec<uchar, 3> Vec3c;
		memcpy(&sframe.frames[c].cv_image.at<Vec3c>(0, 0), pylon_buffer, height * width * 3);

		sframe.frames[c].camera_index = c;
		sframe.frames[c].timestamp = ptrGrabResult->GetTimeStamp();

		if (cv_winNames) {
			cv_winNames[c] = std::to_string((long long)c);
		}
	}
}


uint64_t EvaluateTimestampDifference(uint64_t* timestamp, size_t nsize) {
	uint64_t timestamp_min = std::numeric_limits<uint64_t>::max();
	uint64_t timestamp_max = std::numeric_limits<uint64_t>::min();
	for (size_t j = 0; j < nsize; ++j) {
		if (timestamp_min > timestamp[j]) {
			timestamp_min = timestamp[j];
		}
		if (timestamp_max < timestamp[j]) {
			timestamp_max = timestamp[j];
		}
	}
	return timestamp_max - timestamp_min;
}

bool SynchronizedGrabResults(CBaslerUsbInstantCameraArray& cameras, std::vector<CGrabResultPtr>& ptrGrabResults, bool use_trigger, bool trigger_source_software) {
	bool exception_hashappened = false;

	static bool do_execute_trigger = false;
	static int64_t time_mark = 0;

	if (do_execute_trigger) {
		int time_interval = 0;
		time_interval = (int)ceil(1000.0 / g_resultingFrameRate) + 1;
		do {
			int64_t time_delta = OSDayTimeInMilliseconds() - time_mark;
			if (time_delta < time_interval) {
				OSSleep((DWORD)(time_interval - time_delta));
			}
			else {
				break;
			}

		} while (0 < 1);

		time_mark = OSDayTimeInMilliseconds();

		int trigerCount = 0;
		for (int j = 0; j < (int)cameras.GetSize(); ++j) {
			if (cameras[j].TriggerSource.GetValue() == Basler_UsbCameraParams::TriggerSource_Software) {
				try {
					cameras[j].TriggerSoftware.Execute();
					++trigerCount;
				}
				catch (GenICam::GenericException& e) {
					std::cerr << "TriggerSoftware() An exception occurred: " << e.GetDescription() << std::endl;
					exception_hashappened = true;
				}
			}
		}

		do_execute_trigger = false;

		if (trigerCount > 0) {
			Sleep(10);
			return false;
		}
	}



	static uint64_t timestamp_dif_min = std::numeric_limits<uint64_t>::max();
	uint64_t timestamp_dif = std::numeric_limits<uint64_t>::max();
	uint64_t timestamp[10] = { 0, 0 };

	// Note: the power saving mode must be 
	// maximum performance, otherwise the driver will report eventually that the device has been removed. 
	bool image_isok = true;
	int noresponse_count = 0;
	for (int j = 0; j < (int)cameras.GetSize(); ++j) {
		try {
			cameras[j].RetrieveResult(23, ptrGrabResults[j], TimeoutHandling_Return);
			if (ptrGrabResults[j] == NULL || !ptrGrabResults[j]->GrabSucceeded() || (int)ptrGrabResults[j]->GetCameraContext() != j) {
				++noresponse_count;
			}
			else {
				timestamp[j] = ptrGrabResults[j]->GetTimeStamp();
			}
		}
		catch (GenICam::GenericException& e) {
			std::cerr << "RetrieveResult() An exception occurred: " << e.GetDescription() << std::endl;
			image_isok = false;
			exception_hashappened = true;
		}
	}


	static int partialtimeout_count = 0;
	static int totaltimeout_count = 0;
	static int notimeout_count = 0;
	switch (noresponse_count) {
	case 0: ++notimeout_count; break;
	case 1: ++partialtimeout_count; break;
	case 2: ++totaltimeout_count; break;
	}

	if (noresponse_count > 0) {
		image_isok = false;
	}
	else {
		timestamp_dif = EvaluateTimestampDifference(timestamp, cameras.GetSize());
	}


	if (noresponse_count == 0) {
		if (timestamp_dif_min < timestamp_dif && (timestamp_dif - timestamp_dif_min) > 30000) {
			noresponse_count = 1; // prevent software trigger
			image_isok = false;
			for (size_t j = 0; j < cameras.GetSize(); ++j) {
				bool done = false;
				while (!done) {
					CGrabResultPtr ptr;
					try {
						cameras[j].RetrieveResult(0, ptr, TimeoutHandling_Return);
					}
					catch (GenICam::GenericException& e) {
						std::cerr << "Timestamp. RetrieveResult(). An exception occurred: " << e.GetDescription() << std::endl;
						exception_hashappened = true;
					}
					done = ptr == NULL || !ptr->GrabSucceeded();
					if (!done) {
						ptrGrabResults[j] = ptr;
						try {
							timestamp[j] = ptr->GetTimeStamp();
						}
						catch (GenICam::GenericException& e) {
							std::cerr << "Timestamp. GetTimeStamp(). An exception occurred: " << e.GetDescription() << std::endl;
							exception_hashappened = true;
							done = true;
							j = cameras.GetSize();
						}
					}
					if (!done) {
						timestamp_dif = EvaluateTimestampDifference(timestamp, cameras.GetSize());
						if (timestamp_dif_min >= timestamp_dif || (timestamp_dif - timestamp_dif_min) <= 30000) {
							done = true;
							noresponse_count = 0; // enable software trigger
							image_isok = true;
							j = cameras.GetSize();
						}
					}
				}
			}
		}
		timestamp_dif_min = timestamp_dif;
	}

	if (noresponse_count == 0 || noresponse_count == (int)cameras.GetSize()) {
		if (use_trigger) {
			do_execute_trigger = trigger_source_software;
		}
	}

	if (exception_hashappened) {
		g_bTerminated = true;
		g_bRestart = true;
	}

	return image_isok;
}

return_t __stdcall SynchronizedGrabFrames(LPVOID lp) {
	SImageAcquisitionCtl* ctl = (SImageAcquisitionCtl*)lp;

	ctl->_status = 1;

	__int64 time_previous = OSDayTimeInMilliseconds() - 100;
	__int64 time_average = 100;

	while (!g_bTerminated && !ctl->_terminated) {
		std::vector<CGrabResultPtr> ptrGrabResults(ctl->_cameras->GetSize());
		bool image_isok = SynchronizedGrabResults(*ctl->_cameras, ptrGrabResults, ctl->_use_trigger, ctl->_trigger_source_software);
		if (image_isok) {
			__int64 time_now = OSDayTimeInMilliseconds();
			time_average = (time_average * 20 + (time_now - time_previous)) / 21;
			time_previous = time_now;

			SStereoFrame& sframe = NextWriteFrame();
			try {
				ConvertSynchronizedResults(ptrGrabResults, sframe);
			}
			catch (Exception& ex) {
				std::cout << ex.msg << std::endl;
				g_bTerminated = true;
			}
			sframe.local_timestamp = time_now;
			sframe.isActive = true;
			sframe.gate.unlock();

			if (!g_bTerminated) {
				g_lastwritten_sframe.gate.lock();
				g_lastwritten_sframe = sframe;
				g_lastwritten_sframe.gate.unlock();
			}

			if (g_event_SFrameIsAvailable != INVALID_HANDLE_VALUE) {
				SetEvent(g_event_SFrameIsAvailable);
			}
		}
	}

	ctl->_status = 0;
	return 0;
}



return_t __stdcall AcquireInternalCameraFrames(LPVOID lp) {
	std::cout << "Acquisition has started" << std::endl; 
	timeBeginPeriod(1);
	try {
		SynchronizedGrabFrames(lp);
	}
	catch(Exception& ex) {
		std::cout << ex.msg << std::endl;
		g_bTerminated = true;
	}
	((SImageAcquisitionCtl*)lp)->_status = 0;
	timeEndPeriod(1);
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
	std::cout << "Acquisition has ended" << std::endl;
	return 0;
}




