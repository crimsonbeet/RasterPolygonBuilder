#include "stdafx.h"
#include <Mmsystem.h>
#pragma comment(lib, "Winmm.lib")

#include "XClasses.h"
#include "XConfiguration.h"


const unsigned int g_bytedepth_scalefactor = 256;

Size g_imageSize;


extern bool g_bTerminated;
extern bool g_bRestart; 


void StandardizeImage(Mat& image) {
	if(image.type() != CV_16UC1) {
		if(image.type() != CV_8UC1) {
			image.clone().convertTo(image, CV_8UC1);
		}
		image.clone().convertTo(image, CV_16UC1);
		image *= (size_t)16 * 16;
	}
}
void SquareImage(Mat& image) {
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

	StandardizeImage(left_image);
	StandardizeImage(right_image);

	if (left_image.rows == 0 || left_image.cols == 0) {
		left_image = imread(std::string(g_path_calib_images_dir) + current_N + ".jpg", CV_LOAD_IMAGE_ANYDEPTH);
		StandardizeImage(left_image);
	}
	if (right_image.rows == 0 || right_image.cols == 0) {
		right_image = imread(std::string(g_path_calib_images_dir) + current_N + ".jpg", CV_LOAD_IMAGE_ANYDEPTH);
		//StandardizeImage(right_image);
		SquareImage(right_image);
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




