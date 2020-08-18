#include "stdafx.h"
#include <Mmsystem.h>
#pragma comment(lib, "Winmm.lib")

#include "XClasses.h"
#include "XConfiguration.h"


const unsigned int g_bytedepth_scalefactor = 256;

Size g_imageSize;


extern bool g_bTerminated;
extern bool g_bRestart; 



void ConvertColoredImage2Mono(Mat& image, double chWeights[3], std::function<double(uchar)> convert) {
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
		ConvertColoredImage2Mono(image, chWeights, [](uchar ch) {
			return (double)ch * 256;
		});
	}
	if(image.type() != CV_16UC1) {
		if(image.type() != CV_8UC1) {
			image.clone().convertTo(image, CV_8UC1);
		}
		image.clone().convertTo(image, CV_16UC1);
		image *= (size_t)256;
	}
}
void SquareImage(Mat& image, double chWeights[3]) {
	if (image.type() == CV_8UC3) {
		ConvertColoredImage2Mono(image, chWeights, [](uchar ch) {
			return (double)pow(ch, 2);
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
void BuildWeightsByChannel(Mat& image, Point& pt, double weights_out[3]) {
	if (image.type() == CV_8UC3) {
		double weights[3] = {0,0,0};
		double sum = 0;
		typedef Vec<uchar, 3> Vec3c;
		for (int r = pt.y - 1; r < image.rows && r < pt.y + 2; ++r) {
			for (int c = pt.x - 1; c < image.cols && c < pt.x + 2; ++c) {
				Vec3c& pixVec = image.at<Vec3c>(r, c);
				for (int j = 0; j < 3; ++j) {
					double w = pow(pixVec[j], 2);
					sum += w; 
					weights[j] += w;
				}
			}
		}
		for(auto& w : weights) w /= sum; 
		memcpy(weights_out, weights, sizeof(weights));
	}
}

bool GetImagesFromFile(Mat& left_image, Mat& right_image, const std::string& current_N) {
	std::string nl = current_N;
	std::string nr = current_N;

	double chWeights[3] = { 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };

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
		left_image = imread(std::string(g_path_calib_images_dir) + nl + 'l' + "-chess.png", CV_LOAD_IMAGE_ANYDEPTH); // Mar.4 2015.
	}
	if(right_image.rows == 0 || right_image.cols == 0) {
		right_image = imread(std::string(g_path_calib_images_dir) + nr + 'r' + "-chess.png", CV_LOAD_IMAGE_ANYDEPTH); // Mar.4 2015.
	}

	if (left_image.rows == 0 || left_image.cols == 0) {
		left_image = imread(std::string(g_path_calib_images_dir) + nl + 'l' + ".jpg", CV_LOAD_IMAGE_ANYDEPTH); // Mar.4 2015.
	}
	if (right_image.rows == 0 || right_image.cols == 0) {
		right_image = imread(std::string(g_path_calib_images_dir) + nr + 'r' + ".jpg", CV_LOAD_IMAGE_ANYDEPTH); // Mar.4 2015.
	}

	if (left_image.rows != 0 && left_image.cols != 0) {
		StandardizeImage(left_image, chWeights);
	}
	if (right_image.rows != 0 || right_image.cols != 0) {
		StandardizeImage(right_image, chWeights);
	}

	if (left_image.rows == 0 || left_image.cols == 0) {
		//left_image = imread(std::string(g_path_calib_images_dir) + current_N + ".jpg", CV_LOAD_IMAGE_ANYDEPTH);
		left_image = imread(std::string(g_path_calib_images_dir) + current_N + ".jpg", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
		StandardizeImage(left_image, chWeights);
	}
	if (right_image.rows == 0 || right_image.cols == 0) {
		//right_image = imread(std::string(g_path_calib_images_dir) + current_N + ".jpg", CV_LOAD_IMAGE_ANYDEPTH);
		right_image = imread(std::string(g_path_calib_images_dir) + current_N + ".jpg", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
		SquareImage(right_image, chWeights);
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




