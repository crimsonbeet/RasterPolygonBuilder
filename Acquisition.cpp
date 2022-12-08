#include "stdafx.h"
#include <Mmsystem.h>
#pragma comment(lib, "Winmm.lib")

#include "XClasses.h"
#include "XAndroidCamera.h"

#include "XConfiguration.h"


const unsigned int g_bytedepth_scalefactor = 256;

Size g_imageSize;


extern bool g_bTerminated;
extern bool g_bRestart; 


void RGB_TO_HSV(Vec<uchar, 3> rgb, double hsv[3]) {
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

	hsv[0] = h;
	hsv[1] = s;
	hsv[2] = chMax;
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

bool ConvertColoredImage2Mono_HSV_Likeness(Mat& image, double rgbIdeal[3], std::function<double(double)> convert) {
	typedef Vec<uchar, 3> Vec3c;
	cv::Mat aux(image.size(), CV_16UC1);
	double hsvIdeal[3];
	double hsvOriginal[3];
	
	Vec3c pixIdeal;
	pixIdeal[0] = (uchar)(rgbIdeal[0] + 0.5);
	pixIdeal[1] = (uchar)(rgbIdeal[1] + 0.5);
	pixIdeal[2] = (uchar)(rgbIdeal[2] + 0.5);
	
	RGB_TO_HSV(pixIdeal, hsvIdeal);

	const double saturationIdeal = hsvIdeal[1];

	const double y_componentIdeal = rgbIdeal[0] * 0.299 + rgbIdeal[1] * 0.587 + rgbIdeal[2] * 0.114;
	//const double y_componentIdeal = rgbIdeal[0] * 0.257 + rgbIdeal[1] * 0.504 + rgbIdeal[2] * 0.098 + 16;

	if (y_componentIdeal < 40) {
		return false;
	}
	//if ((hsvIdeal[1] * hsvIdeal[2]) < 40) {
	//	return false;
	//}
	for (int r = 0; r < aux.rows; ++r) {
		for (int c = 0; c < aux.cols; ++c) {
			Vec3c& pixOriginal = image.at<Vec3c>(r, c);
			RGB_TO_HSV(pixOriginal, hsvOriginal);

			double diff = std::abs(GetAngleDifferenceInGrads(hsvIdeal[0], hsvOriginal[0]));
			if (diff > 90) {
				diff = 90;
			}

			//double hueLikeness = 1 - std::abs(std::tan(3.14159265358979323846 * diff / 180));
			//double likeness;
			//if (hueLikeness < 0) {
			//	likeness = 0;
			//}
			//else {
			//	likeness = hueLikeness;

			//	likeness *= std::min(hsvIdeal[1], hsvOriginal[1]) / std::max(hsvIdeal[1], hsvOriginal[1]);
			//	likeness *= std::max(hsvIdeal[2], hsvOriginal[2]);
			//}

			//const double tanThreshold = 1;
			//double hueLikeness = std::pow(1 - diff / 90, 3);

			const double tanThreshold = 0.5;
			double hueLikeness = tanThreshold - std::abs(std::tan(CV_PI * diff / 180));

			double likeness;
			if (hueLikeness < 0) {
				likeness = 0;
			}
			else {
				double saturationOriginal = hsvOriginal[1];
				const double y_componentOriginal = pixOriginal[0] * 0.299 + pixOriginal[1] * 0.587 + pixOriginal[2] * 0.114;
				//const double y_componentOriginal = pixOriginal[0] * 0.257 + pixOriginal[1] * 0.504 + pixOriginal[2] * 0.098 + 16;


				likeness = hueLikeness / tanThreshold;

				//likeness *= std::min(saturationIdeal, saturationOriginal) / std::max(saturationIdeal, saturationOriginal);

				likeness *= std::min(y_componentIdeal, y_componentOriginal) / std::max(y_componentIdeal, y_componentOriginal);
			}

			//aux.at<ushort>(r, c) = convert(hsvOriginal[2] * likeness) + 0.5;
			aux.at<ushort>(r, c) = convert(likeness) + 0.5;
		}
	}
	image = aux.clone();

	return true;
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


bool StandardizeImage_HSV_Likeness(Mat& image, double chIdeal[3]) {
	if (image.type() == CV_8UC3) {
		return ConvertColoredImage2Mono_HSV_Likeness(image, chIdeal, [](double ch) {
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

	return true;
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




void CopyStereoFrame(Mat& left, Mat& right, SStereoFrame* pframe, int64* time_received) {
	// the following code copies the images from frames to the output matrices.
	// the matrices get initalized if they do not match the dimensions. this should happen just one time. 
	// the idea is to initialize the matrices once, and then re-use them after that. 
	Mat* dst[2] = { &left, &right };
	int idx[2] = { pframe->frames[0].camera_index == 0 ? 0 : 1, pframe->frames[0].camera_index == 0 ? 1 : 0 };
	int j = 0;
	for (auto dst : dst) {
		matCV_16UC1_memcpy(*dst, pframe->frames[idx[j++]].cv_image);
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




/*
Read the frame's buffer to the end and pick
a frame that has images captured at smallest time-difference.
The N controlls a minimum amount of frames to consider.
*/
bool GetImages(Mat& left, Mat& right, int64* time_received, const int N/*min_frames_2_consider*/, __int64 expiration) {
	__int64 start_time = OSDayTimeInMilliseconds();

	int count = 0;

	int64 minimal_time_difference = std::numeric_limits<int64>::max();
	SStereoFrame* pframe = NULL;
	while (!g_bTerminated) {
		if ((pframe = NextReadFrame()) == NULL) {
			if (count >= N) {
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
			continue;
		}
		if (pframe->isActive) {
			++count;
		}

		int64 time_difference = max(pframe->frames[0].timestamp, pframe->frames[1].timestamp) - min(pframe->frames[0].timestamp, pframe->frames[1].timestamp);
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


bool GetLastImages(Mat& left, Mat& right, int64* time_received, __int64 expiration) {
	__int64 start_time = OSDayTimeInMilliseconds();

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

	bool ok = count > 0 && !left.empty() && !right.empty();
	return ok;
}




AndroidCameraRaw10Image* ProcvessRaw10Image(AndroidCameraRaw10Image* obj) {
	uint8_t *ptr = obj->_buffers[0]._buffer.data();

	return obj;
}

void Process_CameraImage(AndroidBayerFilterImage* obj) {

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




