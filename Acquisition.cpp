#include "stdafx.h"
#include <Mmsystem.h>
#pragma comment(lib, "Winmm.lib")

#include "XClasses.h"
#include "XConfiguration.h"


const unsigned int g_bytedepth_scalefactor = 256;

Size g_imageSize;


extern bool g_bTerminated;
extern bool g_bRestart; 



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
	double powIdeal1[3] = { pow(chIdeal[0], 1), pow(chIdeal[1], 1), pow(chIdeal[2], 1) };
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



void BuildIdealChannels_Likeness(Mat& image, Point& pt, double chIdeal[3]) {
	if (image.type() == CV_8UC3) {
		double likeness[3] = { 0, 0, 0 };
		double cnt = 0;
		typedef Vec<uchar, 3> Vec3c;
		for (int r = pt.y - 1; r < image.rows && r < pt.y + 2; ++r) {
			for (int c = pt.x - 1; c < image.cols && c < pt.x + 2; ++c) {
				if (r < 0 || c < 0) {
					continue;
				}
				++cnt;
				Vec3c& pixVec = image.at<Vec3c>(r, c);
				for (int j = 0; j < 3; ++j) {
					likeness[j] += pixVec[j];
				}
			}
		}

		for (auto& w : likeness) {
			w /= cnt;
		}

		memcpy(chIdeal, likeness, sizeof(likeness));
	}
}

void ConvertColoredImage2Mono_Likeness(Mat& image, double chIdeal[3], std::function<double(uchar)> convert) {
	assert(image.type() == CV_8UC3);
	cv::Mat aux(image.size(), CV_16UC1);
	typedef Vec<uchar, 3> Vec3c;
	for (int r = 0; r < aux.rows; ++r) {
		for (int c = 0; c < aux.cols; ++c) {
			Vec3c pixVec = image.at<Vec3c>(r, c);
			double pixt = 1;

			for (int j = 0; j < 3; ++j) {
				double chMax;
				double chMin;
				double ch = pixVec[j];
				double ch_ideal = chIdeal[j];
				if (ch_ideal > ch) {
					chMax = ch_ideal;
					chMin = ch;
				}
				else {
					chMax = ch;
					chMin = ch_ideal;
				}

				pixt += chMin > 0 ? (chMin * chMin / chMax) : 0;
			}

			aux.at<ushort>(r, c) = (ushort)std::floor(convert(pixt) + 0.5);
		}
	}
	image = aux.clone();
}

void ConvertColoredImage2Mono_FScore(Mat& image, uchar chIdeal[3], std::function<double(uchar)> convert) {
	cv::Mat aux(image.size(), CV_16UC1);
	double powIdeal2[3] = { pow(chIdeal[0], 2), pow(chIdeal[1], 2), pow(chIdeal[2], 2) };
	typedef Vec<uchar, 3> Vec3c;
	for (int r = 0; r < aux.rows; ++r) {
		for (int c = 0; c < aux.cols; ++c) {
			Vec3c& pixVec = image.at<Vec3c>(r, c);
			ushort& pix = aux.at<ushort>(r, c);
			double w = 1;
			for (int j = 0; j < 3; ++j) {
				double ch = pixVec[j];
				w *= (2.0 * ch * chIdeal[j]) / (pow(ch, 2) + powIdeal2[j]);
			}
			pix = (ushort)std::floor(convert(w * 256) + 0.5); ;
		}
	}
	image = aux.clone();
}




void StandardizeImage_Likeness(Mat& image, double chIdeal[3]) {
	if (image.type() == CV_8UC3) {
		ConvertColoredImage2Mono_Likeness(image, chIdeal, [](double ch) {
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
void SquareImage_Likeness(Mat& image, double chIdeal[3]) {
	if (image.type() == CV_8UC3) {
		ConvertColoredImage2Mono_Likeness(image, chIdeal, [](double ch) {
			return pow(ch, 2);
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


void StandardizeImage_Likeness(Mat& image, uchar chIdeal[3]) {
	if (image.type() == CV_8UC3) {
		ConvertColoredImage2Mono_CosineLikeness(image, chIdeal, [](double ch) {
			return ch * 256;
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
void SquareImage_Likeness(Mat& image, uchar chIdeal[3]) {
	if (image.type() == CV_8UC3) {
		ConvertColoredImage2Mono_CosineLikeness(image, chIdeal, [](double ch) {
			return pow(ch, 2);
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





bool GetImagesFromFile(Mat& left_image, Mat& right_image, const std::string& current_N) {
	std::string nl = current_N;
	std::string nr = current_N;

	//std::string nl = std::to_string(current_N);
	//std::string nr = std::to_string(current_N);

	//left_image = imread(std::string(g_path_calib_images_dir) + nl + 'l' + ".bmp");
	//right_image = imread(std::string(g_path_calib_images_dir) + nr + 'r' + ".bmp");


	left_image.resize(0);
	right_image.resize(0);

	FileStorage fs(std::string(g_path_calib_images_dir) + current_N + ".xml", FileStorage::READ);
	fs["left_image"] >> left_image;
	fs["right_image"] >> right_image;
	fs.release();

	if (left_image.rows == 0 || left_image.cols == 0) {
		left_image = imread(std::string(g_path_calib_images_dir) + nl + 'l' + "-chess.png", ImreadModes::IMREAD_ANYDEPTH); // Mar.4 2015.
	}
	if(right_image.rows == 0 || right_image.cols == 0) {
		right_image = imread(std::string(g_path_calib_images_dir) + nr + 'r' + "-chess.png", ImreadModes::IMREAD_ANYDEPTH); // Mar.4 2015.
	}

	if (left_image.rows == 0 || left_image.cols == 0) {
		left_image = imread(std::string(g_path_calib_images_dir) + nl + 'l' + ".jpg", ImreadModes::IMREAD_ANYDEPTH); // Mar.4 2015.
	}
	if (right_image.rows == 0 || right_image.cols == 0) {
		right_image = imread(std::string(g_path_calib_images_dir) + nr + 'r' + ".jpg", ImreadModes::IMREAD_ANYDEPTH); // Mar.4 2015.
	}

	if (left_image.rows == 0 || left_image.cols == 0) {
		//left_image = imread(std::string(g_path_calib_images_dir) + current_N + ".jpg", CV_LOAD_IMAGE_ANYDEPTH);
		left_image = imread(std::string(g_path_calib_images_dir) + current_N + ".jpg", ImreadModes::IMREAD_ANYDEPTH | ImreadModes::IMREAD_ANYCOLOR);
	}
	if (right_image.rows == 0 || right_image.cols == 0) {
		//right_image = imread(std::string(g_path_calib_images_dir) + current_N + ".jpg", CV_LOAD_IMAGE_ANYDEPTH);
		right_image = imread(std::string(g_path_calib_images_dir) + current_N + ".jpg", ImreadModes::IMREAD_ANYDEPTH | ImreadModes::IMREAD_ANYCOLOR);
	}

	if(left_image.rows <= 10 || right_image.rows <= 10 || left_image.rows != right_image.rows) {
		return false;
	}
	if(left_image.cols <= 10 || right_image.cols <= 10 || left_image.cols != right_image.cols) {
		return false;
	}

	return true;
}

void matCV_8UC1_memcpy(Mat& dst, const Mat& src) {
	if(src.type() != CV_8UC1) {
		throw "matCV_8UC1_memcpy src is not CV_16UC1";
	}
	int width = src.cols;
	int height = src.rows;
	if(dst.rows != height || dst.cols != width || dst.type() != CV_8UC1) {
		dst = Mat(height, width, CV_8UC1);
	}
	memcpy(&dst.at<uchar>(0, 0), &src.at<uchar>(0, 0), height * width * sizeof(uchar));
}

void matCV_16UC1_memcpy(Mat& dst, const Mat& src) {
	if(src.type() != CV_16UC1) {
		throw "matCV_16UC1_memcpy src is not CV_16UC1"; 
	}
	int width = src.cols;
	int height = src.rows;
	if(dst.rows != height || dst.cols != width || dst.type() != CV_16UC1) {
		dst = Mat(height, width, CV_16UC1);
	}
	memcpy(&dst.at<ushort>(0, 0), &src.at<ushort>(0, 0), height * width * sizeof(ushort));
}

uint64 EvaluateTimestampDifference(uint64 *timestamp, size_t nsize) {
	uint64 timestamp_min = std::numeric_limits<uint64>::max();
	uint64 timestamp_max = std::numeric_limits<uint64>::min();
	for(size_t j = 0; j < nsize; ++j) {
		if(timestamp_min > timestamp[j]) {
			timestamp_min = timestamp[j];
		}
		if(timestamp_max < timestamp[j]) {
			timestamp_max = timestamp[j];
		}
	}
	return timestamp_max - timestamp_min;
}

return_t __stdcall AcquireImages(LPVOID lp) {
	std::cout << "Acquisition has started" << std::endl; 
	timeBeginPeriod(1);
	try {
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




