// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define _HAS_AUTO_PTR_ETC 1

#include "stdafx_windows.h"

#include <stdio.h>
#include <tchar.h>

#include <ios>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <map>
#include <set>
#include <utility>
#include <stack>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>
#include <regex>
#include <vector>
#include <cmath>


#if _MSC_VER > 1700
#include <unordered_map>
#endif


#define HAVE_OPENCV_CALIB3D
#define HAVE_OPENCV_FEATURES2D
#define HAVE_OPENCV_DNN
#define HAVE_OPENCV_FLANN
#define HAVE_OPENCV_HIGHGUI
#define HAVE_OPENCV_IMGCODECS

#include "opencv2/opencv.hpp"

using namespace cv;



#include <pylon/PylonIncludes.h>
#include <pylon/PylonGUI.h>
//#include <pylon/gige/PylonGigEIncludes.h>
//#include <pylon/gige/ActionTriggerConfiguration.h>
#include <pylon/usb/PylonUsbIncludes.h>

using namespace Pylon;



#if defined(_DEBUG)
#pragma comment(lib, "opencv_core455d.lib")
#pragma comment(lib, "opencv_imgproc455d.lib")
#pragma comment(lib, "opencv_imgcodecs455d.lib")
#pragma comment(lib, "opencv_calib3d455d.lib")
#pragma comment(lib, "opencv_highgui455d.lib")
#pragma comment(lib, "opencv_objdetect455d.lib")
#pragma comment(lib, "opencv_features2d455d.lib")
#else
#pragma comment(lib, "opencv_core455.lib")
#pragma comment(lib, "opencv_imgproc455.lib")
#pragma comment(lib, "opencv_imgcodecs455.lib")
#pragma comment(lib, "opencv_calib3d455.lib")
#pragma comment(lib, "opencv_highgui455.lib")
#pragma comment(lib, "opencv_objdetect455.lib")
#pragma comment(lib, "opencv_features2d455.lib")
#endif



#include "MASInterface.h"


#pragma comment(lib, "DS7Singleton.lib")
#pragma comment(lib, "KWindows.lib")
#pragma comment(lib, "Polymorpher.lib")





#include "IPCInterface.h" 
#include "XMLWriter.h"


#define NOMINMAX 1


extern const char *IMG_ARROW_LEFT;
extern const char *IMG_ARROW_LEFT_D;
extern const char *IMG_ARROW_LEFT_H;
extern const char *IMG_ARROW_RIGHT;
extern const char *IMG_ARROW_RIGHT_D;
extern const char *IMG_ARROW_RIGHT_H;

extern const char *IMG_NEWDOCUMENT;
extern const char *IMG_NEWDOCUMENT_D;
extern const char *IMG_NEWDOCUMENT_H;
extern const char *IMG_DELETEDOCUMENT;
extern const char *IMG_DELETEDOCUMENT_D;
extern const char *IMG_DELETEDOCUMENT_H;
extern const char *IMG_SAVEDOCUMENT;
extern const char *IMG_SAVEDOCUMENT_D;
extern const char *IMG_SAVEDOCUMENT_H;
extern const char *IMG_STOPDOCUMENT;
extern const char *IMG_STOPDOCUMENT_D;
extern const char *IMG_STOPDOCUMENT_H;
extern const char *IMG_FINISHDOCUMENT;
extern const char *IMG_FINISHDOCUMENT_D;
extern const char *IMG_FINISHDOCUMENT_H;

extern const char *IMG_UNDODELETEDOCUMENT;
extern const char *IMG_UNDODELETEDOCUMENT_D;
extern const char *IMG_UNDODELETEDOCUMENT_H;


#define APPLICATION_INITIALIZED "initialized"
#define APPLICATION_ACTIVE "active"
#define APPLICATION_CLOSED "closed"


BOOL GetStringentity(int iKey, std::string& szText); 
BOOL GetReferencedStringentity(const std::string& szKey, std::string& szText); 

std::string itostdstring(int j);
std::string i64tostdstring(__int64 j);
std::string trim2stdstring(const std::string& str);
void PrintLastError(DWORD dwErr);
int MyCreateDirectory(const std::string& dir_name, const std::string& err_prefix = ""); 
size_t MyGetFileSize(const std::string& filename);
int Delete_FilesInDirectory(const std::string& dir); 

std::string stringify_currentTime(bool file_mode = false);

bool ProcessWinMessages(DWORD dwMilliseconds = 0);



void VS_FileLog(const std::string& msg, bool do_close = false, bool do_ipclog = false); 
void Find_SubDirectories(const std::string& parent_dir, std::vector<std::string>& v_sub_dirs);
void ARFF_FileLog(const std::string& msg, bool do_close = false, bool do_ipclog = false);

std::string CalibrationFileName();



extern const char ESC_KEY;


extern int64_t qcounter_delta;


extern const char *g_path_vsconfiguration;
extern const char *g_path_defaultvsconfiguration;
extern const char *g_path_calib_images_dir;
extern const char* g_path_nwpu_images_dir;

extern const char *g_path_calibrate_file;
extern const char *g_path_calibrate_file_backup;

extern const char *g_path_features_file;
extern const char *g_path_features_file_backup;

extern bool g_bTerminated;
extern bool g_bUserTerminated;
extern bool g_bCamerasAreOk;
extern bool g_bCalibrationExists;
extern bool g_bRestart;


extern uint32_t g_actionDeviceKey;
extern uint32_t g_actionGroupKey;

extern double g_frameRate[4];
extern double g_resultingFrameRate;




extern double g_pattern_distance;
extern double g_max_clusterdistance;
extern double g_axises_ratio;
extern double g_percent_maxintensity;

extern int g_max_boxsize_pixels;

extern double g_max_Y_error;


extern size_t g_min_images;
extern double g_aposteriory_minsdistance;



extern const unsigned int g_bytedepth_scalefactor;


int64_t MyQueryCounter();


template<typename T> struct ATLSMatrixvar {
	size_t _a_dimension;
	size_t _b_dimension;
	T *_matrix;
	T **_vector;

	ATLSMatrixvar<T>() : _matrix(0), _vector(0), _a_dimension(0), _b_dimension(0) {
	}
	~ATLSMatrixvar<T>() {
		if(_matrix) {
			delete[] _matrix;
		}
		if(_vector) {
			delete[] _vector;
		}
	}
	inline T& operator()(size_t a, size_t b) const {
		return _vector[a][b]/*_matrix[a * _b_dimension + b]*/;
	}
	void SetDimensions(size_t a_dim, size_t b_dim) {
		if(_a_dimension < a_dim || _b_dimension < b_dim) {
			if(_matrix) {
				delete[] _matrix;
			}
			if(_vector) {
				delete[] _vector;
			}
			_matrix = new T[a_dim * b_dim];
			_vector = new T*[a_dim];
		}
		if(_a_dimension != a_dim || _b_dimension != b_dim) {
			for(size_t j = 0, k = 0; j < a_dim; ++j, k += b_dim) {
				_vector[j] = &_matrix[k];
			}
		}
		_a_dimension = a_dim;
		_b_dimension = b_dim;

		memset(_matrix, 0, (char*)&_matrix[a_dim * b_dim] - (char*)_matrix);
	}
};


template<typename T> inline int mysign(T v) { return v >= +0 ? 1 : -1; }




template<typename T> struct ATLSMatrix : public ATLSMatrixvar<T> {
	std::vector<size_t> _permutation_vector; // permutation vector returned by lu-decomposition. 

	ATLSMatrix<T>() : ATLSMatrixvar<T>() {
	}
	ATLSMatrix<T>(const ATLSMatrix<T>& other) : ATLSMatrixvar<T>() {
		if (other._a_dimension > 0 || other._b_dimension > 0) {
			SetDimensions(other._a_dimension, other._b_dimension);
			for (size_t j = 0; j < _a_dimension; ++j) {
				memcpy(_vector[j], other._vector[j], _b_dimension * sizeof(T));
			}
		}
	}
	inline void Erase() {
		if (_matrix) {
			delete[] _matrix;
		}
		if (_vector && sizeof(T*) > sizeof(T)) {
			delete[] _vector;
		}
		_matrix = 0;
		_vector = 0;
		_a_dimension = 0;
		_b_dimension = 0;
	}
	~ATLSMatrix<T>() {
		Erase();
	}
	ATLSMatrix<T>& operator=(const ATLSMatrix<T>& other) {
		Erase();
		SetDimensions(other._a_dimension, other._b_dimension);
		for (size_t j = 0; j < _a_dimension; ++j) {
			memcpy(_vector[j], other._vector[j], _b_dimension * sizeof(T));
		}
		return *this;
	}
	bool LUDecompose() {
		// Triangular matrix factorization with column pivot search. 
		bool ok = true;

		const size_t N = std::min(_a_dimension, _b_dimension);
		_permutation_vector.resize(N);
		size_t* p = &_permutation_vector[0];
		for (size_t j = 0; j < N; ++j) {
			p[j] = j;
		}

		for (size_t j = 0; j < (N - 1); ++j) {

			// Pivot search: column-down-search starting with diagonal element (j, j) 
			size_t i = j;
			T abs_value = std::abs(_vector[i][j]);
			T pivot_value = abs_value;
			size_t pivot_index = i;
			while (++i < N) {
				if (pivot_value < (abs_value = std::abs(_vector[i][j]))) {
					pivot_index = i;
					pivot_value = abs_value;
				}
			}
			if (pivot_index != j) {
				std::swap(_vector[j], _vector[pivot_index]);
				std::swap(p[j], p[pivot_index]);
			}
			if (pivot_value < (1.0e-10)) {
				_vector[j][j] = (1.0e-10) * mysign(_vector[j][j]);
				ok = false;
			}
			pivot_value = _vector[j][j];

			// For each row below row j: 
			// Calculate element ij (it actually means that the column j is calculated when rows are processed). 
			// Adjust all elements to the right from j.
			i = j;
			while (++i < N) {
				double vector_ij = (_vector[i][j] /= pivot_value);
				for (size_t k = j + 1; k < N; ++k) {
					_vector[i][k] -= _vector[j][k] * vector_ij;
				}
			}
		}

		return ok;
	}
	void LUBacksubstitute(std::vector<T>& conditions_vector) {
		LUBacksubstitute(&conditions_vector[0]);
	}
	void LUBacksubstitute(T* b) {
		// Solves the set of n linear equations A·X = B. 
		// Here _matrix is input, not as the matrix A but rather as its LU decomposition. 
		// _permutation_vector is input as the permutation vector returned by lu-decomposition. 
		// b[0..N-1] is input as the right-hand side vector B, and returns with the solution vector X. 
		// _matrix and _permutation_vector are not modified by this routine and 
		// can be left in place for successive calls with different right-hand sides b. 
		// This routine takes into account the possibility that b will begin with many zero elements, 
		// so it is efficient for use in matrix inversion.
		const size_t N = std::min(_a_dimension, _b_dimension);
		std::vector<T> results_vector;
		results_vector.resize(N); // it also initializes with default values. 
		T* c = &results_vector[0];
		size_t* p = &_permutation_vector[0];

		T sum;
		for (int i = 0, ii = -1; i < (int)N; ++i) { // Now we do the forward substitution.
			sum = b[p[i]]; // Unscramble the permutation.

			// When ii is set to a positive value, it becomes the index of the first nonvanishing element of b. 

			if (ii != -1) {
				for (int j = ii; j < i; ++j) { // We now do the forward substitution. 
					sum -= _vector[i][j] * c[j];
				}
				c[i] = sum;
			}
			else
				if (sum != (T)0.0) {
					// A nonzero element was encountered, so from now on it will have to 
					// do the sums in the loop above.
					ii = i;
					c[i] = sum;
				}
		}
		for (int i = (int)N - 1; i >= 0; --i) { // Now we do the backsubstitution.
			sum = c[i];
			for (size_t j = i + 1; j < N; ++j) {
				sum -= _vector[i][j] * c[j];
			}
			c[i] = sum / _vector[i][i]; // Store a component of the solution vector X.
		}

		memcpy(b, c, (char*)(b + _b_dimension) - (char*)b);
	}
	ATLSMatrix<T>& Inverse() {
		LUDecompose();

		ATLSMatrix<T> y;
		y.SetDimensions(_a_dimension, _b_dimension);

		const size_t N = std::min(_a_dimension, _b_dimension);
		std::vector<T> unit_vector;
		unit_vector.resize(N);
		T* col = &unit_vector[0];
		for (size_t j = 0; j < N; ++j) { // Find inverse by columns.
			col[j] = (T)1.0;
			LUBacksubstitute(unit_vector);
			for (size_t i = 0; i < N; ++i) {
				y._vector[i][j] = col[i];
				col[i] = 0.0;
			}
		}

		std::swap(_vector, y._vector);
		std::swap(_matrix, y._matrix);

		y._permutation_vector.swap(_permutation_vector);

		return (*this);
	}

	template<typename T1, typename T2>
	ATLSMatrix<T>& Multiply(const ATLSMatrix<T1>& L, const ATLSMatrix<T2>& R) {
		SetDimensions(L._a_dimension, R._b_dimension);
		for (size_t i = 0; i < _a_dimension; ++i) {
			for (size_t n = 0; n < _b_dimension; ++n) {
				T& val = _vector[i][n];
				val = 0;
				for (size_t k = 0; k < R._a_dimension; ++k) {
					val += L(i, k) * R(k, n);
				}
			}
		}
		_permutation_vector.resize(0);

		return (*this);
	}
	template<typename T1, typename T2>
	ATLSMatrix<T>& AxBTMultiply(const ATLSMatrix<T1>& L, const ATLSMatrix<T2>& R) { // right matrix will be transposed.
		SetDimensions(L._a_dimension, R._a_dimension);
		for (size_t i = 0; i < _a_dimension; ++i) {
			for (size_t n = 0; n < _b_dimension; ++n) {
				T& val = _vector[i][n];
				val = 0;
				for (size_t k = 0; k < R._b_dimension; ++k) {
					val += L(i, k) * R(n, k);
				}
			}
		}
		_permutation_vector.resize(0);

		return (*this);
	}
	template<typename T1, typename T2>
	ATLSMatrix<T>& ATxBMultiply(const ATLSMatrix<T1>& L, const ATLSMatrix<T2>& R) { // left matrix will be transposed.
		SetDimensions(L._b_dimension, R._b_dimension);
		for (size_t i = 0; i < _a_dimension; ++i) {
			for (size_t n = 0; n < _b_dimension; ++n) {
				T& val = _vector[i][n];
				val = 0;
				for (size_t k = 0; k < R._a_dimension; ++k) {
					val += L(k, i) * R(k, n);
				}
			}
		}
		_permutation_vector.resize(0);

		return (*this);
	}
	template<typename T1>
	ATLSMatrix<T>& MxMTMultiply(const ATLSMatrix<T1>& M) { // M×MT i.e. Transpose multiply where transpose matr. is on the right
		SetDimensions(M._a_dimension, M._a_dimension);
		for (size_t i = 0; i < _a_dimension; ++i) {
			for (size_t n = 0; n < _a_dimension; ++n) {
				T& val = _vector[i][n];
				val = 0;
				for (size_t k = 0; k < M._b_dimension; ++k) {
					val += M(i, k) * M(n, k);
				}
			}
		}
		_permutation_vector.resize(0);

		return (*this);
	}
	template<typename T1>
	ATLSMatrix<T>& MTxMMultiply(const ATLSMatrix<T1>& M) { // MT×M i.e. Transpose multiply where transpose matr. is on the left
		SetDimensions(M._b_dimension, M._b_dimension);
		for (size_t i = 0; i < _a_dimension; ++i) {
			for (size_t n = 0; n < _a_dimension; ++n) {
				T& val = _vector[i][n];
				val = 0;
				for (size_t k = 0; k < M._a_dimension; ++k) {
					val += M(k, i) * M(k, n);
				}
			}
		}
		_permutation_vector.resize(0);

		return (*this);
	}
	template<typename T1, typename T2>
	void Split(ATLSMatrix<T1>& L, ATLSMatrix<T2>& R) { // It is meaningful after LUDecomposition
		L.SetDimensions(_a_dimension, _b_dimension);
		R.SetDimensions(_a_dimension, _b_dimension);
		L._permutation_vector = _permutation_vector;
		R._permutation_vector = _permutation_vector;
		for (size_t i = 0; i < _a_dimension; ++i) {
			for (size_t j = 0; j < _b_dimension; ++j) {
				if (i <= j) {
					R(i, j) = _vector[i][j];
					L(i, j) = i == j ? 1 : 0;
				}
				else {
					R(i, j) = 0;
					L(i, j) = _vector[i][j];
				}
			}
		}
	}
	void Print(char* pFormat = "%14.7f", std::string* pmsg = 0) {
		char buf[256];
		std::ostringstream ostr;
		if (pmsg == 0) {
			std::cout << std::endl;
		}
		for (size_t i = 0; i < _a_dimension; ++i) {
			for (size_t j = 0; j < _b_dimension; ++j) {
				sprintf(buf, pFormat, _vector[i][j]);
				if (pmsg == 0) {
					std::cout << buf << '\t';
				}
				else {
					ostr << buf << '\t';
				}
			}
			if (pmsg == 0) {
				std::cout << std::endl;
			}
			else {
				ostr << std::endl;
			}
		}
		if (pmsg) {
			*pmsg = ostr.str();
		}
	}
};





struct SCalibrationBaseCtl {
	int _status; // ==1 - unknown, 0 - not running, 2 - running. 
	bool _terminated; // a command to stop execution.

	std::string _outputWindow; // where the _image2visualize to put to.

	wsi_gate _gate;

	bool volatile _data_isvalid;
	bool volatile _image_isvalid;

	cv::Mat _image; // input to FeatureDetector

	cv::Mat _image2visualize; // input to main thread

	long long _last_image_timestamp;

	SCalibrationBaseCtl() : _status(1), _terminated(0), _data_isvalid(false), _image_isvalid(false), _last_image_timestamp(0) {
	}
};



struct SFeatureDetectorCtl : public SCalibrationBaseCtl {
	cv::Ptr<cv::FeatureDetector> _detector;
	double _saturationFactor;

	std::vector<cv::KeyPoint> _keyPoints;
	std::vector<cv::Point2f> _pointBuf; // centers of black squares 

	std::vector<cv::Point2f> _approx2fminQuad; // Qaudrilateral delimiting the area of the image that contains corners.
	cv::Rect _approxBoundingRectMapped;
	cv::Mat _H; // Homography that transforms quadrilateral to rectangle (not rotated).

	std::vector<cv::Point2d> _edgesBuf;

	cv::Rect _roi;


	SFeatureDetectorCtl() : SCalibrationBaseCtl(), _saturationFactor(1.0) {
	}

	SFeatureDetectorCtl(cv::Mat& image) : SFeatureDetectorCtl() {
		_image = image;
	}
};



struct SStereoCalibrationCtl : public SCalibrationBaseCtl {

	std::vector<size_t> _seedSelection;
	//std::vector<size_t> _finalPointsSelection;

	size_t _selection_pos = 0;
	size_t _pos_ok = 0;

	size_t _sample_size = 0;


	size_t _iter_num = 0;

	Mat *_cameraMatrix1 = nullptr;
	Mat *_cameraMatrix2 = nullptr;
	Mat *_distortionCoeffs1 = nullptr;
	Mat *_distortionCoeffs2 = nullptr;
	Mat *_R = nullptr;
	Mat *_T = nullptr;
	Mat *_E = nullptr;
	Mat *_F = nullptr;


	double *_rms_s = nullptr;


	SStereoCalibrationCtl() : SCalibrationBaseCtl() {
	}
};




extern Size g_boardSize;
extern Size g_boardQuadSize;


class ClassBlobDetector : public SimpleBlobDetector {
	bool _chess_board;
	bool _invert_binary;
	double _min_confidence;
	int _min_threshold;

	SFeatureDetectorCtl* _ctl = nullptr;

	struct CV_EXPORTS Center
	{
		Point2d location;
		double radius;
		double confidence;
	};

	void findBlobs(const Mat& image, Mat& binaryImage, std::vector<Center>& centers, std::vector<std::vector<Point>>& contours) const;
	void detectImpl(const Mat& image, std::vector<KeyPoint>& keypoints, const Mat& mask = Mat());

public:
	Params params;

	virtual void detect(InputArray image, CV_OUT std::vector<KeyPoint>& keypoints, InputArray mask = noArray()) {
		detectImpl(image.getMat(), keypoints);
	}
	ClassBlobDetector(double min_confidence = 0.3, size_t min_repeatability = 3, int min_threshold = 80, bool white_on_black = false, bool chess_board = false) {

		_min_confidence = min_confidence;
		_min_threshold = min_threshold;
		_chess_board = chess_board;
		_invert_binary = false; // use blobColor instead
		
		params.blobColor = white_on_black ? 255 : 0;

		params.minDistBetweenBlobs = 2;
		params.minThreshold = _min_threshold * g_bytedepth_scalefactor;
		params.thresholdStep = 20 * g_bytedepth_scalefactor;
		params.maxThreshold = 180 * g_bytedepth_scalefactor;
		params.minRepeatability = min_repeatability/*5*/; // 2015-09-15 Set repeatabilty to 3 because of difficulties with capturing images
		params.minArea = 100;
		params.maxArea = 70000;
		params.minInertiaRatio = 0.3;
		params.minCircularity = 0.8f;
		params.minConvexity = 0.8f;

		params.filterByColor = false;
		params.filterByCircularity = true;
		params.filterByInertia = true;
		params.filterByArea = true; 
		params.filterByConvexity = true;
	}

	ClassBlobDetector(const ClassBlobDetector& other, SFeatureDetectorCtl *ctl) {
		_min_confidence = other._min_confidence;
		_min_threshold = other._min_threshold;
		_chess_board = other._chess_board;
		_invert_binary = other._invert_binary; 

		_ctl = ctl;

		params = other.params;
	}
};









struct ABox {
	int x[2];
	int y[2];
	int intensity; 
	ABox(): intensity(0) {
		memset(x, 0, sizeof(x));
		memset(y, 0, sizeof(y));
	}
	ABox(int x1, int x2, int y1, int y2, int intens = 0): intensity(intens) {
		x[0] = x1;
		x[1] = x2;
		y[0] = y1;
		y[1] = y2;
	}
	bool IsValid() const {
		return x[0] < x[1] && y[0] < y[1];
	}
	std::vector<Point> contour;
	std::vector<Point> contour_notsmoothed;
};



struct ClusteredPoint: public Point2d {
	void Init() {
		_center = 0;
		_isACorner = 0;
		_isACenter = 0;
		_crop_mat_scalefactor = 0;
		_corners_max_Y_error = 2 * g_max_Y_error;
		_centers_min_Y_error = std::numeric_limits<int>::max();

		_shapemeasure = 0;
		_effective_flattening = 0; 
		_flattening = 0;
		_skewness = 0;
		_covar = 0;
		_hull_circularity = 0;
		_intensity_upperquantile = 0;
		_contour_area = 0;
		_contour_area2hull_area = 0; 
		_centers_distance = 0; 
		_intensity_atcenter = 0; 
		_display_weight_factor = 0;
	}
	ClusteredPoint(): _cluster(-1), _intensity(0), _camera(-1) {
		Init(); 
	}
	ClusteredPoint(const Point2d& point, int intensity = 0, int camera = -1, double shapemeasure = 0, Mat crop = Mat()): Point2d(point), _cluster(-1), _intensity(intensity), _camera(camera), _crop((const MatCloner&)crop) {
		Init();
		_shapemeasure = shapemeasure; 
	}
	ClusteredPoint(const Point2d& point, int intensity, int aType, ClusteredPoint *center, double max_Y_error = 2 * g_max_Y_error): ClusteredPoint(point, intensity) {
		_center = center;
		_isACorner = aType == 1? 1: 0;
		_isACenter = aType == 2? 1: 0; 
		_crop_mat_scalefactor = 0;
		_corners_max_Y_error = max_Y_error;
	}
	int aType() {
		return _isACorner? 1: _isACenter? 2: 0;
	}

	void ARFF_Output(bool isValid, bool rectangleDetected); 

	int _cluster;
	int _intensity; 
	int _camera; 

	double _intensity_atcenter;
	double _shapemeasure;
	double _effective_flattening; 
	double _flattening;
	double _skewness; 
	double _covar; 
	double _hull_circularity; 
	double _intensity_upperquantile; 
	double _contour_area; 
	double _contour_area2hull_area; 
	double _centers_distance; // distance between center of gravity and center of rectangle. 
	double _display_weight_factor;

	struct MatCloner: public Mat {
		MatCloner(): Mat() {
		}
		MatCloner(const MatCloner& cloner): Mat() {
			if(cloner.rows > 0 || cloner.cols > 0) {
				(Mat&)*this = cloner.clone();
			}
		}
	};

	MatCloner _crop;
	MatCloner _cropOriginal;
	double _crop_mat_scalefactor;
	Point2f _crop_mat_offset;
	Point2f _crop_center;

	std::vector<Point2d> _corners; // 4 corners of a rectangular patch; they are used to generate 4 2D-points for 3D reconstruction of corners. 
	ClusteredPoint *_center; // pointer to an original ClusteredPoint that was used to generate a ClusteredPoint for 3D reconstruction. 
	int _isACorner;
	int _isACenter;

	std::vector<std::vector<cv::Point2d>> _contours;
	std::vector<Point2d> _contour_notsmoothed;

	double _corners_max_Y_error; // control parameter for partitioning. 
	double _centers_min_Y_error; // work value; min. Y distance between rectangles in left and right image; is used to select correct match.  
};

struct ReconstructedPoint: public Matx41d {
	void inline Init() { 
		_not_astandalone_object = 0; 
		_reprojection_error = 0; 
		_cluster = -1; 
		_isACoordinatePoint = 0; 
		_isACorner = 0; 
		_isACenter = 0; 
	}
	ReconstructedPoint(): _id(-1) { 
		Init(); 
	}
	ReconstructedPoint(const Mat_<double>& point, int id): Matx41d(point(0), point(1), point(2), 1), _id(id) {
		Init();
	}
	ReconstructedPoint(const Matx41d& point, int id, int aType, double reprojection_error = 0, int cluster = -1): Matx41d(point), _id(id) { 
		Init();
		_isACorner = aType == 1? 1: 0;
		_isACenter = aType == 2? 1: 0;
		_isACoordinatePoint = aType == 3? 1: 0;
		_reprojection_error = reprojection_error;
		_cluster = cluster; 
	}
	int aType() const {
		return _isACorner? 1: _isACenter? 2: _isACoordinatePoint? 3: 0; 
	}
	operator Mat_<double>() {
		Mat_<double> dst(4, 1);
		dst(0) = (*this)(0);
		dst(1) = (*this)(1);
		dst(2) = (*this)(2);
		dst(3) = 1;
		return dst;
	}
	int _id; // identifier (or position in a vector) of original image point. 
	int _isACoordinatePoint; 
	int _not_astandalone_object; 
	double _reprojection_error; 
	int _cluster; 
	int _isACorner; 
	int _isACenter;
};





void BlobDetector(std::vector<ABox>& boxes, Mat& image, const unsigned int min_intensity, cv::Rect roi, const unsigned int max_intensity = 255 * g_bytedepth_scalefactor, int max_boxsize_pixels = g_max_boxsize_pixels, const double circularity_ratio = 3.0 / 5.0);

int BlobCentersLoG(std::vector<ABox>& boxes, std::vector<ClusteredPoint>& points, Mat& image, unsigned int& threshold_intensity, cv::Rect roi, Mat_<double>& kmat, bool arff_file_requested = false, ushort* intensity_avg_ptr = 0, double max_LoG_factor = 21.0);

template<typename T1, typename T2>
void CopyVector(std::vector<T1>& dst, std::vector<T2>& src) {
	dst.clear();
	std::transform(src.cbegin(), src.cend(), std::back_inserter(dst), [](T2 p) { return T1(p); });
}



size_t ConductOverlapEliminationEx(const std::vector<std::vector<cv::Point2d>>& contours, std::vector<std::vector<cv::Point2d>>& final_contours,
	bool preserve_scale_factor, // causes to bypass back-scaling and return the used scale_factor
	long& scale_factor, // in/out. specifies what power of 2 to use to get coordinates in integers
	bool conduct_size = false,
	int size_increment = -1,
	bool log_graph = false);





constexpr auto NUMBER_OF_CAMERAS = 2;

/*
A structure to pass (by pointer) to image acquisition thread,
in order to control the termination,
and also to provide a pointer to open cameras.
*/
struct SImageAcquisitionCtl {
	int _status; // -1 - unknown, 0 - not running, 1 - running. 
	bool _terminated; // a command to stop execution.

	CBaslerUsbInstantCameraArray* _cameras = nullptr;

	int _imagepoints_status;

	bool _use_trigger;
	bool _trigger_source_software;

	bool _calib_images_from_files; 
	bool _save_all_calibration_images; 
	bool _two_step_calibration;
	bool _pattern_is_chessBoard;
	bool _pattern_is_gridOfSquares;
	bool _pattern_is_whiteOnBlack; 

	int _image_height;

	std::string _camera_serialnumbers[NUMBER_OF_CAMERAS];

	int _exposure_times[NUMBER_OF_CAMERAS];

	int _12bit_format;

	SImageAcquisitionCtl(): _status(0), _imagepoints_status(0), _terminated(0) {
		_cameras = &_basler_cameras;

		_use_trigger = true;
		_trigger_source_software = true; 

		_calib_images_from_files = false; 
		_save_all_calibration_images = false;
		_two_step_calibration = true;
		_pattern_is_gridOfSquares = true;
		_pattern_is_whiteOnBlack = true; 
		_pattern_is_chessBoard = false; 

		_image_height = 483; 

		_12bit_format = 0; 

		_camera_serialnumbers[0] = "40269283"; // defines what cameras to use
		_camera_serialnumbers[NUMBER_OF_CAMERAS - 1] = "40294791";

		for (auto& exp : _exposure_times) exp = 0;
	}

private:
	CBaslerUsbInstantCameraArray _basler_cameras = CBaslerUsbInstantCameraArray(NUMBER_OF_CAMERAS); // defines number of cameras to use
};



bool GetImagesFromFile(Mat& left_image, Mat& right_image, std::vector<cv::Point2d>& pointsLeft, std::vector<cv::Point2d>& pointsRight, const std::string& current_N);



/*
Main calculation loop. 
*/
return_t __stdcall ReconstructPoints(LPVOID lp); 

/*
Main acquisition loop
*/
return_t __stdcall AcquireImages(LPVOID lp);


return_t __stdcall EvaluateOtsuThreshold(LPVOID lp);




struct ARowend {
	int _column;
	bool _processed[2];
	ARowend(int column = -1) {
		_column = column;
		_processed[0] = _processed[1] = false;
	}
};

struct ABoxedrow {
	int _rownum;
	ARowend _rowends[2];
	ABoxedrow(int rownum, int column = -1) {
		_rownum = rownum;
		_rowends[0]._column = column;
		_rowends[1]._column = column;
	}
	ABoxedrow(int rownum, int column0, int column1) {
		_rownum = rownum;
		_rowends[0]._column = column0;
		_rowends[1]._column = column1;
	}
	ABoxedrow(): _rownum(-1) {
	}
};

struct ABoxedblob {
	int _id;
	std::vector<ABoxedrow> _rows;
	ABoxedblob(int id = -1) {
		_id = id;
	}

	std::vector<int> _associatedBlobs;

	void add_association(ABoxedblob& other) {
		if(std::find(_associatedBlobs.begin(), _associatedBlobs.end(), other._id) == _associatedBlobs.end()) {
			_associatedBlobs.push_back(other._id);
		}
	}
};


Mat_<double> LoG(double sigma, int ksize); // Build Laplacian of Gaussian kernel

int64_t BlobLoG(std::vector<ABoxedblob>& blobs, // it returns counter for time spent in convolving
	const Point2d& aSeed,
	const Mat& image,
	const Mat_<double>& kmat,
	const int row_limits[2],
	const int col_limits[2],
	ATLSMatrixvar<signed short>& tracker,
	ATLSMatrixvar<double>& tracker_value, const double max_LoG_factor = 21.0
); 

int BlobsLoG(std::vector<ABoxedblob>& blobs, 
	Mat& image, 
	unsigned int& threshold_intensity, 
	cv::Rect roi, 
	Mat_<double>& kmat, 
	unsigned short *intensity_avg_ptr = 0, const double max_LoG_factor = 21.0
); 

extern double g_otsu_threshold; 


extern const size_t g_rotating_buf_size;

void matCV_16UC1_memcpy(Mat& dst, const Mat& src);
void matCV_8UC1_memcpy(Mat& dst, const Mat& src);
void matCV_8UC3_memcpy(Mat& dst, const Mat& src);

template<typename T1>
void mat_minMax(Mat& m, T1 minMax[2]); 

template<typename T1, typename T2>
inline void mat_findMinMax(T2 *val, const cv::Size& src_size, const size_t row_step, T1 minMax[2]); 

template<typename T>
inline const T mat_get(Mat& m, int i, int j); 

void mat_threshold(Mat& src, const double threshold_val, double zero_val = -1); 

Mat mat_binarize2byte(const Mat& src, const int bytedepth_scalefactor = g_bytedepth_scalefactor); // returns CV_8UC1 matrix

Mat mat_invert2byte(const Mat& src, const int bytedepth_scalefactor = g_bytedepth_scalefactor, const uchar maxvalue = 255); // returns CV_8UC1 matrix
Mat mat_convert2byte(const Mat& src, const int bytedepth_scalefactor = g_bytedepth_scalefactor, const uchar maxvalue = 255); // returns CV_8UC1 matrix
Mat mat_loginvert2byte(const Mat& src, const int bytedepth_scalefactor = g_bytedepth_scalefactor); // returns CV_8UC1 matrix of log inverted values


void RGB_TO_HSV(cv::Vec<uchar, 3> rgb, double hsv[3]);

double hsvLikenessScore(cv::Vec<uchar, 3>& pixOriginal, double hsvIdeal[3]); // returns likeness score from 0 to 256.

double Get_Squared_Z_Score(const cv::Vec<uchar, 3>& pixOrig, double mean_data[3], double invCholesky_data[3][3]);

void BuildIdealChannels_Likeness(Mat& image, Point& pt, double rgbdeal[3]);
bool BuildIdealChannels_Distribution(Mat& image, Point& pt, Mat& mean, Mat& stdDev, Mat& factorLoadings, Mat& invCovar, Mat& invCholesky, int neighbourhoodRadius = 4);


bool StandardizeImage_HSV_Likeness(Mat& image, double rgbIdeal[3]);
void StandardizeImage_Likeness(Mat& image, Mat mean/*rgb*/, Mat& stdDev, Mat& factorLoadings, Mat invCovar/*inverted covariance of colors*/, Mat invCholesky);

Mat mat_invert2word(const Mat& src, const int bytedepth_scalefactor = g_bytedepth_scalefactor, const uint16_t maxvalue = 65535); // returns CV_16UC1 matrix
Mat mat_loginvert2word(const Mat& src, const int bytedepth_scalefactor = g_bytedepth_scalefactor); // returns CV_16UC1 matrix

void ConvertColoredImage2Mono(Mat& image, double chWeights[3], std::function<double(double)> convert);



inline double approx_log2(double x) {
	//static double pow2_52 = pow(2, 52);
	//static double log2_e = log2(2.71828182845904523536028747135266249775724709369995);

	if(x < 1.00001) {
		return 0;
	}

	__int64& ddwHexImage = *(__int64*)((PVOID)(&x));

	int nExp = (int)((ddwHexImage & 0x7ff0000000000000) >> 52) - 1023;
	//double fMantissa = ((double)(ddwHexImage & 0x000fffffffffffff)) / pow2_52 + 1;

	ddwHexImage |= 0x3ff0000000000000;
	ddwHexImage &= 0x3fffffffffffffff;

	return (((-0.344845 * x/*fMantissa*/ + 2.024658) * x/*fMantissa*/) - 1.674873 + nExp);
}
inline double approx_ln(double x) {
	static double ln_2 = log(2);

	return approx_log2(x) * ln_2;
}







template<typename M, typename T>
inline Mat_<M> mat_vectorize(T *src_val, const cv::Size& src_size, const size_t row_step) { // returns Mat_<M> N by 3.
	Mat_<M> dst(src_size.height * src_size.width, 3);
	int dst_row = 0;
	char *buf = (char*)src_val;
	for(int k = 0; k < src_size.height; ++k, buf += row_step) {
		src_val = (T*)buf;
		for(int j = 0; j < src_size.width; ++j, ++src_val) {
			dst(dst_row, 0) = k;
			dst(dst_row, 1) = j;
			dst(dst_row, 2) = (M)*src_val;
			++dst_row;
		}
	}
	return dst;
}

template<typename M>
Mat_<M> mat_vectorize(Mat& src) { // returns Mat_<M> N by 3.
	Mat_<M> dst;
	switch(src.type()) {
	case CV_8UC1:
	dst = mat_vectorize<M>((unsigned char*)src.data, src.size(), src.step.p[0]);
	break;
	case CV_8SC1:
	dst = mat_vectorize<M>((signed char*)src.data, src.size(), src.step.p[0]);
	break;
	case CV_16UC1:
	dst = mat_vectorize<M>((unsigned short*)src.data, src.size(), src.step.p[0]);
	break;
	case CV_16SC1:
	dst = mat_vectorize<M>((signed short*)src.data, src.size(), src.step.p[0]);
	break;
	case CV_32SC1:
	dst = mat_vectorize<M>((int*)src.data, src.size(), src.step.p[0]);
	break;
	case CV_32FC1:
	dst = mat_vectorize<M>((float*)src.data, src.size(), src.step.p[0]);
	break;
	case CV_64FC1:
	dst = mat_vectorize<M>((long long*)src.data, src.size(), src.step.p[0]);
	break;
	}
	return dst;
}






// This function is a modified version of function partition() from openCv operations.hpp
// This function enforces more accurate partitioning. 
// The original version implements disjoint sets, but not equivalence classes. 
// This function comes closer to equivalence classes, but still admits entries into a partition that are not equivalent. 
//
// The algorithm is described in "Introduction to Algorithms"
// by Cormen, Leiserson and Rivest, the chapter "Data structures for disjoint sets"
template<typename _Tp, class _EqPredicate> int
partitionEx(const std::vector<_Tp>& _vec, std::vector<int>& labels,
_EqPredicate predicate = _EqPredicate()) {
	int i, j, N = (int)_vec.size();
	const _Tp* vec = &_vec[0];

	const int PARENT = 0;
	const int RANK = 1;

	std::vector<int> _nodes(N * 2);
	int(*nodes)[2] = (int(*)[2])&_nodes[0];

	// The first O(N) pass: create N single-vertex trees
	for(i = 0; i < N; i++) {
		nodes[i][PARENT] = -1;
		nodes[i][RANK] = 0;
	}

	// The main O(N^2) pass: merge connected components
	for(i = 0; i < N; i++) {
		int root = i;
		while(nodes[root][PARENT] >= 0)
			root = nodes[root][PARENT];

		for(j = 0; j < N; j++) {
			if(i == j)
				continue;

			if(!predicate(vec[i], vec[j]))
				continue;

			int root2 = j;
			while(nodes[root2][PARENT] >= 0)
				root2 = nodes[root2][PARENT];

			if(root2 == root)
				continue;

			if(j > root2)
				continue;

			if((j != root2 || i != root) && !predicate(vec[root], vec[root2])) // ANB 2016-10-24
				continue;

			// unite both trees
			int rank = nodes[root][RANK], rank2 = nodes[root2][RANK];
			if(rank > rank2)
				nodes[root2][PARENT] = root;
			else {
				nodes[root][PARENT] = root2;
				if(rank == rank2)
					nodes[root2][RANK] += 1;
				root = root2;
			}
			assert(nodes[root][PARENT] < 0);

			int k = j, parent;

			// compress the path from node2 to root
			while((parent = nodes[k][PARENT]) >= 0) {
				nodes[k][PARENT] = root;
				k = parent;
			}

			// compress the path from node to root
			k = i;
			while((parent = nodes[k][PARENT]) >= 0) {
				nodes[k][PARENT] = root;
				k = parent;
			}
		}
	}

	// Final O(N) pass: enumerate classes
	labels.resize(N);
	int nclasses = 0;

	for(i = 0; i < N; i++) {
		int root = i;
		while(nodes[root][PARENT] >= 0)
			root = nodes[root][PARENT];
		// re-use the rank as the class label
		if(nodes[root][RANK] >= 0)
			nodes[root][RANK] = ~nclasses++;
		labels[i] = ~nodes[root][RANK];
	}

	return nclasses;
}




void show_image(Mat& image, const std::string& window_name);




/*
A structure to pass (by pointer) to points reconstruction thread,
in order to control the termination,
and also to provide an input configuration parameters,
and also to provide output parameters for screen feed-back.
*/
struct SPointsReconstructionCtl {
	int _status; // !=2 - unknown, 0 - not running, 2 - running. 
	bool _terminated; // a command to stop execution.

	int _pixel_threshold; // is replaced with Otsu optimal threshold; is used for on-screen output of current value. 

	wsi_gate _gate;


	Mat _cameraMatrix[4];
	Mat _distortionCoeffs[4];

	Mat _R, _T, _E, _F;

	Mat _Rl, _Rr, _Pl, _Pr, _Q;

	Mat _map_l[4];
	Mat _map_r[4];

	cv::Size _rectified_image_size;

	bool _calibration_exists = false;


	cv::Matx44d _world_transform; // a transform matrix that maps to coord. system of 3d features of a detected object. 
	long long _world_transform_timestamp; // a timestamp of when _world_transform has been updated. 

	// onscreen feeedback values. 

	bool volatile _data_isvalid;
	bool volatile _image_isvalid;

	bool volatile _local_acquisition_hasfinished;

	std::vector<int> _labels;
	int _coordsystem_label; // it is the label of the cluster that represents the coordsystem. 
	std::vector<ReconstructedPoint> _points4D; // the _id member of each point links to the point in cv_point array. 
	std::vector<std::vector<ReconstructedPoint>> _coordlines4D;
	std::vector<ReconstructedPoint> _points4Dtransformed;
	std::vector<std::vector<ReconstructedPoint>> _coordlines4Dtransformed;

	Mat _cv_image[2]; // rectified image
	Mat _cv_edges[2]; // is used for drawing the images in "Camera1" and "Camera2" windows
	Mat _unchangedImage[2]; 

	std::vector<ABox> _boxes[2];
	std::vector<ClusteredPoint> _cv_points[2];

	std::string _edgesWindows[2];
	std::string _pointsCropWindows[2];
	std::string _combinedFitWindows[2];

	cv::Rect _roi[2];

	long long _last_image_timestamp;


	// foreground extraction diagnostics. 

	Mat _foreground_image[2];
	Mat _foreground_diff[2];
	Mat _image_mask[2];
	Mat _images_transformed[2][2];

	bool volatile _foreground_extraction_isvalid;

	bool _draw_epipolar_lines;

	SPointsReconstructionCtl(int pixel_threshold = 70): _pixel_threshold(pixel_threshold), _status(0), _terminated(0), _data_isvalid(false), _image_isvalid(false), _last_image_timestamp(0) {
		_local_acquisition_hasfinished = false;
		_world_transform_timestamp = std::numeric_limits<long long>::min();
		_foreground_extraction_isvalid = false;
		_coordsystem_label = -1; 
		_draw_epipolar_lines = true;
	}
};


typedef void vs_callback_launch_workerthreads(SImageAcquisitionCtl& image_acquisition_ctl, SPointsReconstructionCtl *reconstruction_ctl);

void launch_reconstruction(SImageAcquisitionCtl& image_acquisition_ctl, SPointsReconstructionCtl *reconstruction_ctl);



bool DisplayReconstructionData(SPointsReconstructionCtl& reconstruction_ctl, int& time_average);


void InitializeCameras(SImageAcquisitionCtl& ctl);
void OpenCameras(SImageAcquisitionCtl& ctl);








struct SVideoFrame {
	int camera_index;

	Mat cv_image;
	uint64 timestamp;

	SVideoFrame() {
		camera_index = -1;
		timestamp = 0;
	}

	SVideoFrame& operator=(const SVideoFrame& other) {
		if (!other.cv_image.empty()) {
			matCV_8UC3_memcpy(cv_image, other.cv_image);
		}
		else {
			cv_image = Mat();
		}
		camera_index = other.camera_index;
		timestamp = other.timestamp;

		return *this;
	}
};



struct SStereoFrame {
	wsi_gate gate;
	SVideoFrame frames[NUMBER_OF_CAMERAS];

	uint64 local_timestamp;
	bool isActive;
	SStereoFrame() {
		local_timestamp = 0;
		isActive = false;
	}

	SStereoFrame& operator=(const SStereoFrame& other) {
		local_timestamp = other.local_timestamp;
		isActive = other.isActive;
		for (size_t j = 0; j < ARRAY_NUM_ELEMENTS(other.frames); ++j) {
			frames[j] = other.frames[j];
		}
		return *this;
	}
};


extern SStereoFrame g_lastwritten_sframe;

bool GetImages(Mat& left, Mat& right, int64_t* time_received = NULL, const int N = 0/*min_frames_2_consider*/, int64_t expiration = 3000);

/*
Copies from g_lastwritten_sframe. Waits if neccessary.
*/
bool GetLastFrame(Mat& left, Mat& right, int64_t* time_received = NULL, int64_t expiration = 3000);








template<typename T>
VOID XmlSerialize_Generic(std::string& xml, T& data) {
	IWsiSerializerBase& serializer = GetRootSerializer(&data);

	serializer.lock();
	serializer(CXMLWriter(), &data, xml);
	serializer.unlock();
}




