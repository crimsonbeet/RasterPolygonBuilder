#include "stdafx.h"
#include <Mmsystem.h>

#include "XConfiguration.h"

#include "FrameMain.h"
#include "LoGSeedPoint.h"

#pragma warning(disable : 26451)


extern bool g_bTerminated;
extern bool g_bRestart;
extern Size g_imageSize; 
extern HANDLE g_event_SFrameIsAvailable;
extern HANDLE g_event_SeedPointIsAvailable;
extern HANDLE g_event_ContourIsConfirmed;
extern LoGSeedPoint g_LoG_seedPoint;


extern const char *g_path_vsconfiguration;
extern const char *g_path_defaultvsconfiguration;

double g_max_Y_error = 4.0;
double g_max_clusterdistance = 4;
double g_axises_ratio = 7.0 / 9.0;
double g_percent_maxintensity = 1.0 / 3.0; 

int g_max_boxsize_pixels = 25; 


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.4142135623730950488016887242097
#endif



void StandardizeImage(Mat& image, double chWeights[3]);
void SquareImage(Mat& image, double chWeights[3]);
void BuildWeights_ByChannel(Mat& image, Point& pt, double weights_out[3]);


void StandardizeImage_Likeness(Mat& image, double chIdeal[3]);
bool StandardizeImage_HSV_Likeness(Mat& image, double rgbIdeal[3]);
void SquareImage_Likeness(Mat& image, double chIdeal[3]);

void StandardizeImage_Likeness(Mat& image, uchar chIdeal[3]);
void SquareImage_Likeness(Mat& image, uchar chIdeal[3]);



ATLSMatrix<double> X_XXI_X;
ATLSMatrix<double> NEG_X_XXI_X;

void Prefetch_PolyspaceProjection(ATLSMatrix<double>& x_xxi_x, int sampleSize = 33, int sign = 1) {
	ATLSMatrix<double> X;
	X.SetDimensions(sampleSize, 5);

	double t = (M_PI / 2) / sampleSize;

	for (int n = 0; n < sampleSize; ++n) {
		X(n, 0) = 1;
		X(n, 1) = pow(sampleSize - n, pow(2.0, sign));
		X(n, 2) = pow(n, pow(2.0, sign));
		X(n, 3) = sin(t * n) *sampleSize / 2;
		X(n, 4) = cos(t * n) *sampleSize / 2;
	}

	ATLSMatrix<double> aux; 

	x_xxi_x.MTxMMultiply(X);
	x_xxi_x.Inverse();
	x_xxi_x.AxBTMultiply(aux.Multiply(X, x_xxi_x), X);
}



bool AngleFromCosineTheorem(double c, double a, double b, double& angle) { // angle in radians
	bool ok = false;
	if(a > 0 && b > 0 && c >= 0) {
		double cos_c = (pow(a, 2) + pow(b, 2) - pow(c, 2)) / (2 * a * b);
		if(fabs(cos_c) <= 1) {
			angle = std::acos(cos_c);
			ok = true;
		}
	}
	return ok;
}

double Angle(double x, double y, double& tan_xy) {
	int Quadrant = 0;
	if(x > 0)
		Quadrant = y < 0 ? 4 : 1;
	else
	if(x < 0)
		Quadrant = y > 0 ? 2 : 3;
	else
		return y > 0 ? (CV_PI / 2) : (3 * CV_PI / 2);

	double a = std::atan(tan_xy = ((double)y) / x);
	switch(Quadrant) {
	default:
	break; 
	case 2:
	a += CV_PI; 
	break;
	case 3:
	a += CV_PI;
	break;
	case 4:
	a += CV_PI * 2.0;
	break;
	}
	return a; 
}
double Angle(double x, double y) {
	double tan_xy; 
	return Angle(x, y, tan_xy);
}


template<typename T>
double DirectionOfRectangle(const std::vector<T>& points, Point2d& direction, Point2d& offset, bool points_are_presorted = false) { // returns angle
	if(points.size() != 4) {
		return 0; 
	}

	int idx1 = 0; 

	if(!points_are_presorted) {
		double min_y = points[0].y;
		for(int j = 1; j < 4; ++j) {
			if(points[j].y < min_y) {
				min_y = points[j].y;
				idx1 = j;
			}
		}
	}

	int idx0 = idx1 > 0? (idx1 - 1): 3;
	int idx2 = idx1 < 3? (idx1 + 1): 0;
	int idx3 = idx2 < 3? (idx2 + 1): 0;
	double dist0 = std::pow(points[idx0].y - points[idx1].y, 2) + std::pow(points[idx0].x - points[idx1].x, 2);
	double dist2 = std::pow(points[idx2].y - points[idx1].y, 2) + std::pow(points[idx2].x - points[idx1].x, 2);

	int idxn = dist2 < dist0? idx2: idx0; 

	int idx1_o = idx3; 
	int idxn_o = idxn == idx0? idx2: idx0;

	Point2d diag1 = points[idx1_o] - points[idx1]; 
	Point2d diagn = points[idxn_o] - points[idxn];

	//double a = Angle(diag1.x, diag1.y);
	//double b = Angle(diagn.x, diagn.y);

	//if(a < 0) {
	//	a += 2 * CV_PI; 
	//}
	//if(b < 0) {
	//	b += 2 * CV_PI;
	//}

	//double theta = (a + b) / 2; 

	offset = (points[idx1] + points[idxn]) * 0.5;
	direction = (diag1 + diagn) * 0.5;

	if(!points_are_presorted) {
		if(direction.y < 0) {
			offset += direction;
			direction *= -1;
		}
	}

	return Angle(direction.x, direction.y);
}

template<typename T>
double InclinationAngleOfRectangle(const std::vector<T>& points) {
	Point2d direction;
	Point2d offset;
	return DirectionOfRectangle(points, direction, offset); 
}

template<typename T>
bool QuadCenter(const std::vector<T>& corners, Point2d& cornersCenter) {
	bool rc = false;
	double diag1_x[2] = {corners[0].x, corners[2].x};
	double diag1_y[2] = {corners[0].y, corners[2].y};
	double diag2_x[2] = {corners[1].x, corners[3].x};
	double diag2_y[2] = {corners[1].y, corners[3].y};
	double denom = (diag1_x[0] - diag1_x[1])*(diag2_y[0] - diag2_y[1]) - (diag1_y[0] - diag1_y[1])*(diag2_x[0] - diag2_x[1]);
	double a = diag1_x[0] * diag1_y[1] - diag1_y[0] * diag1_x[1];
	double b = diag2_x[0] * diag2_y[1] - diag2_y[0] * diag2_x[1];
	if(std::abs(denom) > 0.001) {
		cornersCenter = Point2d((a * (diag2_x[0] - diag2_x[1]) - b * (diag1_x[0] - diag1_x[1])) / denom, (a * (diag2_y[0] - diag2_y[1]) - b * (diag1_y[0] - diag1_y[1])) / denom);
		rc = true;
	}
	return rc;
}

template<typename T>
void SetQuadCounterclockwiseShorterEdgeOutgoing(T *points, const size_t N) {
	// Set the quad counterclockwise at the lowest point. if next point is on longer edge, position the quad to the previous point. 
	if(N < 3) {
		return;
	}

	//double ylevel = 0;
	//std::function<bool(const Point2d &one, const Point2d &another)> lambda = [&ylevel](const Point2d &one, const Point2d &another) -> bool {
	//	return one.y < (another.y - ylevel) || (std::abs(one.y - another.y) < ylevel && one.x < another.x);
	//};

	// 1. Set the quad to the minimum y point. 
	int idx0 = 0;
	double min_y = points[0].y;
	for(int j = 1; j < N; ++j) {
		if(points[j].y < min_y) {
			min_y = points[j].y;
			idx0 = j;
		}
	}
	if(idx0 > 0) {
		std::rotate(points, points + idx0, points + N);
		idx0 = 0;
	}
	int idx1 = 1;
	int idx_1 = (int)N - 1;

	// 2. Set the quad (polyline as well) counterclockwise. 
	double t;
	double angle1 = Angle(points[idx1].x - points[idx0].x, points[idx1].y - points[idx0].y, t);
	double angle_1 = Angle(points[idx_1].x - points[idx0].x, points[idx_1].y - points[idx0].y, t);
	if(angle_1 < angle1) {
		std::reverse(points + 1, points + N);
	}

	// 3. if next point is on longer edge, position the quad to the previous point. 
	double dist1 = std::pow(points[idx0].y - points[idx1].y, 2) + std::pow(points[idx0].x - points[idx1].x, 2);
	double dist_1 = std::pow(points[idx0].y - points[idx_1].y, 2) + std::pow(points[idx0].x - points[idx_1].x, 2);
	if(dist_1 < dist1) {
		std::rotate(points, points + idx_1, points + N);
	}
}

template<typename T>
void SetQuadCounterclockwiseShorterEdgeOutgoing(std::vector<T>& points) {
	if(points.size() < 3) {
		return;
	}
	bool isOpen = points[0] != points[points.size() - 1];
	SetQuadCounterclockwiseShorterEdgeOutgoing(&points[0], isOpen ? points.size() : (points.size() - 1));
}



double Sort_by_epipolar_equivalence(Point2d p0[4], Point2d p1[4]) {
	// Method: order the points so that the y-error is minimized while maintaining counter clockwise orientation of both quads. 
	// Assumption: aspect ratio is preserved, i.e. longer edge of one rectangle corresponds to longer edge of another. 

	double avg_Y_error = 0;

	// 1. set both quads counterclockwise, also at the point with minimum y and the next point on shorter edge.

	SetQuadCounterclockwiseShorterEdgeOutgoing(p0, 4);
	SetQuadCounterclockwiseShorterEdgeOutgoing(p1, 4);

	// 2. count sum of errors when first quad is at first point, and then at diagonal to it point. 

	double y_error_sum[2] = {0, 0};
	for(int k = 0, i = 0; k < 4; k += 2, ++i) {
		for(int j = 0; j < 4; ++j) {
			y_error_sum[i] += std::abs(p0[j].y - p1[(j + k) % 4].y);
		}
	}

	// 3. roatate first quad to the diagonal point if the corresponding error is smaller. 

	if(y_error_sum[1] < y_error_sum[0]) {
		std::rotate(p1, p1 + 2, p1 + 4);
		avg_Y_error = y_error_sum[1]; 
	}
	else {
		avg_Y_error = y_error_sum[0];
	}

	return avg_Y_error / 4;
}

double Average_y_error_betweenRectangles(const std::vector<Point2d>& points0, const std::vector<Point2d>& points1, double& max_Y_error) {
	Point2d p0[4] = {points0[0], points0[1], points0[2], points0[3]};
	Point2d p1[4] = {points1[0], points1[1], points1[2], points1[3]};

	double ylevel_error = Sort_by_epipolar_equivalence(p0, p1);

	double avg_Y_error = 0;
	max_Y_error = 0; 

	for(int j = 0; j < 4; ++j) {
		double error = p0[j].y - p1[j].y; 
		if(std::abs(error) > std::abs(max_Y_error)) {
			max_Y_error = error;
		}
		avg_Y_error += error;
	}

	return avg_Y_error / 4.0; 
}

double DistanceSquareFromPointToSegment(const Point2d& P, const Point2d& A, const Point2d& B) {
	double PxAx = P.x - A.x;
	double PyAy = P.y - A.y;
	double BxAx = B.x - A.x;
	double ByAy = B.y - A.y;

	double dist = 0;

	double t = PxAx * BxAx + PyAy * ByAy;
	if(t < 0) {
		dist = (std::pow(PxAx, 2) + std::pow(PyAy, 2));
	}
	else {
		double BxPx = B.x - P.x;
		double ByPy = B.y - P.y;
		t = BxPx * BxAx + ByPy * ByAy;
		if(t < 0) {
			dist = (std::pow(BxPx, 2) + std::pow(ByPy, 2));
		}
		else {
			dist = (std::pow(PyAy * BxAx - PxAx * ByAy, 2) / (std::pow(BxAx, 2) + std::pow(ByAy, 2)));
		}
	}
	return dist;
}




double RadiusOfRectangle(const std::vector<Point2d>& points) {
	if(points.size() != 4) {
		return 0;
	}
	double dist0 = std::sqrt(std::pow(points[0].y - points[2].y, 2) + std::pow(points[0].x - points[2].x, 2));
	double dist1 = std::sqrt(std::pow(points[1].y - points[3].y, 2) + std::pow(points[1].x - points[3].x, 2));

	return (dist0 + dist1) / 4; 
}




double FindBestAlignment(const std::vector<int>& pattern, const std::vector<int>& strip2searchForPattern) { // returns center point of alignment
	const int gapCost = 1;

	const size_t M = pattern.size() + 1;
	const size_t N = strip2searchForPattern.size() + 1;

	const size_t X = M / 2;

	std::vector<std::vector<int>> A(M);
	for (auto& v : A) {
		v.resize(N);
	}

	std::vector<std::vector<int>> T(M);
	for (auto& t : T) {
		t.resize(N);
	}

	for (size_t i = 0; i < M; ++i) {
		A[i][0] = 1; // M* N* gapCost;
	}

	for (size_t j = 0; j < N; ++j) {
		A[0][j] = 1; // M* N* gapCost;
	}

	for (size_t i = 1; i < M; ++i) {
		for (size_t j = 1; j < N; ++j) {
			const size_t i_1 = i - 1;
			const size_t j_1 = j - 1;
			int case1Cost = A[i_1][j_1] + std::abs(pattern[i_1] - strip2searchForPattern[j_1]);
			//int case1Cost = A[i_1][j_1] + std::sqrt(approx_log2(std::abs((int)i - (int)X))) * std::abs(pattern[i_1] - strip2searchForPattern[j_1]); // not good: long tails do not work on low texture
			int case2Cost = A[i_1][j] + 2 * gapCost;
			int case3Cost = A[i][j_1] + gapCost;
			if (case1Cost < case2Cost && case1Cost < case3Cost) {
				T[i][j] = 1;
				A[i][j] = case1Cost;
			}
			else 
			if (case2Cost < case3Cost) {
				T[i][j] = 2;
				A[i][j] = case2Cost;
			}
			else {
				T[i][j] = 3;
				A[i][j] = case3Cost;
			}
		}
	}

	double Y = 0; // yet unknown
	size_t w = 0;

	size_t m = M - 1;
	size_t n = N - 1;

	int caseType = T[m][n];
	int caseCost = A[m][n];

	for (size_t j = n; j > M; --j) {
		if (T[m][j] == 1) {
			if (A[m][j] <= caseCost) {
				n = j;
				caseCost = A[m][j];
				caseType = T[m][j];
			}
		}
	}

	if (n != (N - 1)) {
		std::cout << "Changed starting case number to " << n << std::endl;
	}

	while (caseType != 0) {
		switch (caseType) {
		case 1:
			--m;
			--n;
			break;
		case 2:
			--m; // case of allignment matches entry from pattern and gap from strip2search
			break;
		case 3:
			--n;
			break;
		}

		if (m == X) {
			Y += n - 1;
			++w;
		}

		caseType = T[m][n];
	}

	if (w != 0) {
		Y /= w;
	}

	return Y;
}






template<typename T>
inline Mat mat_invert2byte(const T *src_val, const cv::Size& src_size, const size_t row_step, const int bytedepth_scalefactor, const uchar maxvalue) { // returns CV_8UC1 matrix
	Mat dst(src_size, CV_8UC1);
	uchar *val = (uchar*)dst.data;
	char *buf = (char*)src_val;
	for(int k = 0; k < src_size.height; ++k, buf += row_step) {
		src_val = (const T*)buf;
		for(int j = 0; j < src_size.width; ++j, ++val, ++src_val) {
			int ival = (int)((*src_val) / bytedepth_scalefactor + (T(0.49999)));
			*val = (uchar)255 - (uchar)std::min(ival, 255);
			if(*val > maxvalue) {
				*val = (uchar)(approx_log2(maxvalue - *val) + 0.49999) + maxvalue;
			}
		}
	}
	return dst;
}

Mat mat_invert2byte(const Mat& src, const int bytedepth_scalefactor/* = g_bytedepth_scalefactor*/, const uchar maxvalue/* = 255*/) { // returns CV_8UC1 matrix
	Mat dst;
	switch(src.type()) {
	case CV_8UC1:
	dst = mat_invert2byte((unsigned char*)src.data, src.size(), src.step.p[0], 1, maxvalue);
	break;
	case CV_8SC1:
	dst = mat_invert2byte((signed char*)src.data, src.size(), src.step.p[0], 1, maxvalue);
	break;
	case CV_16UC1:
	dst = mat_invert2byte((unsigned short*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor, maxvalue);
	break;
	case CV_16SC1:
	dst = mat_invert2byte((signed short*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor, maxvalue);
	break;
	case CV_32SC1:
	dst = mat_invert2byte((int*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor, maxvalue);
	break;
	case CV_32FC1:
	dst = mat_invert2byte((float*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor, maxvalue);
	break;
	case CV_64FC1:
	dst = mat_invert2byte((long long*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor, maxvalue);
	break;
	}
	return dst;
}

template<typename T>
inline Mat mat_convert2byte(const T *src_val, const cv::Size& src_size, const size_t row_step, const int bytedepth_scalefactor, const uchar maxvalue) { // returns CV_8UC1 matrix
	Mat dst(src_size, CV_8UC1);
	uchar *val = (uchar*)dst.data;
	char *buf = (char*)src_val;
	for(int k = 0; k < src_size.height; ++k, buf += row_step) {
		src_val = (const T*)buf;
		for(int j = 0; j < src_size.width; ++j, ++val, ++src_val) {
			//int ival = (int)floor(double(*src_val) / bytedepth_scalefactor + 0.5); 
			int ival = (int)((*src_val) / bytedepth_scalefactor + (T(0.49999)));
			*val = (uchar)std::min(ival, 255);
			if(*val > maxvalue) {
				*val = (uchar)(approx_log2(maxvalue - *val) + 0.49999) + maxvalue;
			}
		}
	}
	return dst;
}

Mat mat_convert2byte(const Mat& src, const int bytedepth_scalefactor/* = g_bytedepth_scalefactor*/, const uchar maxvalue/* = 255*/) { // returns CV_8UC1 matrix
	Mat dst;
	switch(src.type()) {
	case CV_8UC1:
	dst = mat_convert2byte((unsigned char*)src.data, src.size(), src.step.p[0], 1, maxvalue);
	break;
	case CV_8SC1:
	dst = mat_convert2byte((signed char*)src.data, src.size(), src.step.p[0], 1, maxvalue);
	break;
	case CV_16UC1:
	dst = mat_convert2byte((unsigned short*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor, maxvalue);
	break;
	case CV_16SC1:
	dst = mat_convert2byte((signed short*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor, maxvalue);
	break;
	case CV_32SC1:
	dst = mat_convert2byte((int*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor, maxvalue);
	break;
	case CV_32FC1:
	dst = mat_convert2byte((float*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor, maxvalue);
	break;
	case CV_64FC1:
	dst = mat_convert2byte((long long*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor, maxvalue);
	break;
	}
	return dst;
}

template<typename T>
inline Mat mat_loginvert2byte(const T *src_val, const cv::Size& src_size, const size_t row_step, const int bytedepth_scalefactor) { // returns CV_8UC1 matrix of log inverted values. 
	const double scalefactor = 1.0 / bytedepth_scalefactor;
	const double factor = 256.0 / approx_log2(256);
	const const double log_256 = approx_log2(256);
	Mat dst(src_size, CV_8UC1);
	uchar *val = (uchar*)dst.data;
	char *buf = (char*)src_val;
	for(int k = 0; k < src_size.height; ++k, buf += row_step) {
		src_val = (T*)buf;
		for(int j = 0; j < src_size.width; ++j, ++val, ++src_val) {
			double src_valt = double(*src_val) * scalefactor;
			if(src_valt <= 1) {
				*val = 255;
			}
			else {
				*val = (uchar)((log_256 - approx_log2(src_valt)) * factor);
			}
		}
	}
	return dst;
}

Mat mat_loginvert2byte(const Mat& src, const int bytedepth_scalefactor/* = g_bytedepth_scalefactor*/) { // returns CV_8UC1 matrix
	Mat dst;
	switch(src.type()) {
	case CV_8UC1:
	dst = mat_loginvert2byte((unsigned char*)src.data, src.size(), src.step.p[0], 1);
	break;
	case CV_8SC1:
	dst = mat_loginvert2byte((signed char*)src.data, src.size(), src.step.p[0], 1);
	break;
	case CV_16UC1:
	dst = mat_loginvert2byte((unsigned short*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor);
	break;
	case CV_16SC1:
	dst = mat_loginvert2byte((signed short*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor);
	break;
	case CV_32SC1:
	dst = mat_loginvert2byte((int*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor);
	break;
	case CV_32FC1:
	dst = mat_loginvert2byte((float*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor);
	break;
	case CV_64FC1:
	dst = mat_loginvert2byte((long long*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor);
	break;
	}
	return dst;
}

template<typename T>
inline Mat mat_loginvert2word(const T *src_val, const cv::Size& src_size, const size_t row_step, const int bytedepth_scalefactor) { // returns CV_16UC1 matrix of log inverted values. 
	const double scalefactor = bytedepth_scalefactor;
	const double factor = 65536.0 / approx_log2(65536);
	const const double log_65536 = approx_log2(65536);
	Mat dst(src_size, CV_16UC1);
	uint16_t *val = (uint16_t*)dst.data;
	char *buf = (char*)src_val;
	for (int k = 0; k < src_size.height; ++k, buf += row_step) {
		src_val = (T*)buf;
		for (int j = 0; j < src_size.width; ++j, ++val, ++src_val) {
			double src_valt = double(*src_val) * scalefactor;
			if (src_valt <= scalefactor) {
				*val = 65535;
			}
			else {
				*val = (uint16_t)((log_65536 - approx_log2(src_valt)) * factor);
			}
		}
	}
	return dst;
}

Mat mat_loginvert2word(const Mat& src, const int bytedepth_scalefactor) { // returns CV_16UC1 matrix
	Mat dst;
	switch (src.type()) {
	case CV_8UC1:
		dst = mat_loginvert2word((unsigned char*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor);
		break;
	case CV_8SC1:
		dst = mat_loginvert2word((signed char*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor);
		break;
	case CV_16UC1:
		dst = mat_loginvert2word((unsigned short*)src.data, src.size(), src.step.p[0], 1);
		break;
	case CV_16SC1:
		dst = mat_loginvert2word((signed short*)src.data, src.size(), src.step.p[0], 1);
		break;
	case CV_32SC1:
		dst = mat_loginvert2word((int*)src.data, src.size(), src.step.p[0], 1);
		break;
	case CV_32FC1:
		dst = mat_loginvert2word((float*)src.data, src.size(), src.step.p[0], 1);
		break;
	case CV_64FC1:
		dst = mat_loginvert2word((long long*)src.data, src.size(), src.step.p[0], 1);
		break;
	}
	return dst;
}

template<typename T>
inline Mat mat_invert2word(const T* src_val, const cv::Size& src_size, const size_t row_step, const int bytedepth_scalefactor, const uint16_t maxvalue) { // returns CV_16UC1 matrix of log inverted values. 
	const double scalefactor = bytedepth_scalefactor;
	Mat dst(src_size, CV_16UC1);
	uint16_t* val = (uint16_t*)dst.data;
	char* buf = (char*)src_val;
	for (int k = 0; k < src_size.height; ++k, buf += row_step) {
		src_val = (T*)buf;
		for (int j = 0; j < src_size.width; ++j, ++val, ++src_val) {
			double src_valt = double(*src_val) * scalefactor;
			if (src_valt <= scalefactor) {
				*val = 65535;
			}
			else {
				src_valt = 65535.0 - std::min(src_valt, 65535.0); 
				if (src_valt > maxvalue) {
					src_valt = (approx_log2(maxvalue - src_valt) + 0.49999) + maxvalue;
				}

				*val = (uint16_t)src_valt;
			}
		}
	}
	return dst;
}

Mat mat_invert2word(const Mat& src, const int bytedepth_scalefactor, const uint16_t maxvalue) { // returns CV_16UC1 matrix
	Mat dst;
	switch (src.type()) {
	case CV_8UC1:
		dst = mat_invert2word((unsigned char*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor, maxvalue);
		break;
	case CV_8SC1:
		dst = mat_invert2word((signed char*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor, maxvalue);
		break;
	case CV_16UC1:
		dst = mat_invert2word((unsigned short*)src.data, src.size(), src.step.p[0], 1, maxvalue);
		break;
	case CV_16SC1:
		dst = mat_invert2word((signed short*)src.data, src.size(), src.step.p[0], 1, maxvalue);
		break;
	case CV_32SC1:
		dst = mat_invert2word((int*)src.data, src.size(), src.step.p[0], 1, maxvalue);
		break;
	case CV_32FC1:
		dst = mat_invert2word((float*)src.data, src.size(), src.step.p[0], 1, maxvalue);
		break;
	case CV_64FC1:
		dst = mat_invert2word((long long*)src.data, src.size(), src.step.p[0], 1, maxvalue);
		break;
	}
	return dst;
}

template<typename T>
inline Mat mat_binarize2byte(const T *src_val, const cv::Size& src_size, const size_t row_step, const int bytedepth_scalefactor) { // returns CV_8UC1 matrix
	Mat dst(src_size, CV_8UC1);
	uchar *val = (uchar*)dst.data;
	char *buf = (char*)src_val;
	for(int k = 0; k < src_size.height; ++k, buf += row_step) {
		src_val = (const T*)buf;
		for(int j = 0; j < src_size.width; ++j, ++val, ++src_val) {
			if(*src_val < bytedepth_scalefactor) {
				*val = 0;
			}
			else {
				*val = 255;
			}
		}
	}
	return dst;
}

Mat mat_binarize2byte(const Mat& src, const int bytedepth_scalefactor/* = g_bytedepth_scalefactor*/) { // returns CV_8UC1 matrix
	Mat dst;
	switch(src.type()) {
	case CV_8UC1:
	dst = mat_binarize2byte((unsigned char*)src.data, src.size(), src.step.p[0], 1);
	break;
	case CV_8SC1:
	dst = mat_binarize2byte((signed char*)src.data, src.size(), src.step.p[0], 1);
	break;
	case CV_16UC1:
	dst = mat_binarize2byte((unsigned short*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor);
	break;
	case CV_16SC1:
	dst = mat_binarize2byte((signed short*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor);
	break;
	case CV_32SC1:
	dst = mat_binarize2byte((int*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor);
	break;
	case CV_32FC1:
	dst = mat_binarize2byte((float*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor);
	break;
	case CV_64FC1:
	dst = mat_binarize2byte((long long*)src.data, src.size(), src.step.p[0], bytedepth_scalefactor);
	break;
	}
	return dst;
}



template<typename T>
inline void mat_threshold(T *src_val, const cv::Size& src_size, const size_t row_step, const T threshold_val, const T zero_val) {
	char *buf = (char*)src_val;
	for(int k = 0; k < src_size.height; ++k, buf += row_step) {
		src_val = (T*)buf;
		for(int j = 0; j < src_size.width; ++j, ++src_val) {
			if(*src_val < threshold_val) {
				*src_val = zero_val;
			}
		}
	}
}

void mat_threshold(Mat& src, const double threshold_val, double zero_val) {
	if(zero_val < 0) {
		zero_val = threshold_val; 
	}
	switch(src.type()) {
	case CV_8UC1:
	mat_threshold((unsigned char*)src.data, src.size(), src.step.p[0], (unsigned char)floor(threshold_val), (unsigned char)floor(zero_val));
	break;
	case CV_8SC1:
	mat_threshold((signed char*)src.data, src.size(), src.step.p[0], (signed char)floor(threshold_val), (signed char)floor(zero_val));
	break;
	case CV_16UC1:
	mat_threshold((unsigned short*)src.data, src.size(), src.step.p[0], (unsigned short)floor(threshold_val), (unsigned short)floor(zero_val));
	break;
	case CV_16SC1:
	mat_threshold((signed short*)src.data, src.size(), src.step.p[0], (signed short)floor(threshold_val), (signed short)floor(zero_val));
	break;
	case CV_32SC1:
	mat_threshold((int*)src.data, src.size(), src.step.p[0], (int)floor(threshold_val), (int)floor(zero_val));
	break;
	case CV_32FC1:
	mat_threshold((float*)src.data, src.size(), src.step.p[0], (float)threshold_val, (float)zero_val);
	break;
	case CV_64FC1:
	mat_threshold((long long*)src.data, src.size(), src.step.p[0], (long long)threshold_val, (long long)zero_val);
	break;
	}
}




template<typename T>
inline const T mat_get(Mat& m, int i, int j) {
	T rc;
	switch(m.type()) {
	case CV_8UC1:
	rc = (T)m.at<unsigned char>(i, j);
	break;
	case CV_8SC1:
	rc = (T)m.at<signed char>(i, j);
	break;
	case CV_16UC1:
	rc = (T)m.at<unsigned short>(i, j);
	break;
	case CV_16SC1:
	rc = (T)m.at<signed short>(i, j);
	break;
	case CV_32SC1:
	rc = (T)m.at<int>(i, j);
	break;
	case CV_32FC1:
	rc = (T)m.at<float>(i, j);
	break;
	case CV_64FC1:
	rc = (T)m.at<double>(i, j);
	break;
	default:
	rc = -1;
	break;
	}
	return rc;
}


template<typename T1, typename T2>
inline void mat_findMinMax(T2 *val, const cv::Size& src_size, const size_t row_step, T1 minMax[2]) { // returns CV_8UC1 matrix
	T2 rc[2] = {std::numeric_limits<T2>::max(), std::numeric_limits<T2>::min()};
	char *buf = (char*)val;
	for(int k = 0; k < src_size.height; ++k, buf += row_step) {
		val = (T2*)buf; 
		for(int j = 0; j < src_size.width; ++j, ++val) {
			if(*val < rc[0]) {
				rc[0] = *val;
			}
			if(*val > rc[1]) {
				rc[1] = *val;
			}
		}
	}
	minMax[0] = (T1)rc[0]; 
	minMax[1] = (T1)rc[1];
}

template<typename T1>
void mat_minMax(Mat& m, T1 minMax[2]) {
	switch(m.type()) {
	case CV_8UC1:
	mat_findMinMax((unsigned char*)m.data, m.size(), m.step.p[0], minMax);
	break;
	case CV_8SC1:
	mat_findMinMax((signed char*)m.data, m.size(), m.step.p[0], minMax);
	break;
	case CV_16UC1:
	mat_findMinMax((unsigned short*)m.data, m.size(), m.step.p[0], minMax);
	break;
	case CV_16SC1:
	mat_findMinMax((signed short*)m.data, m.size(), m.step.p[0], minMax);
	break;
	case CV_32SC1:
	mat_findMinMax((int*)m.data, m.size(), m.step.p[0], minMax);
	break;
	case CV_32FC1:
	mat_findMinMax((float*)m.data, m.size(), m.step.p[0], minMax);
	break;
	case CV_64FC1:
	mat_findMinMax((long long*)m.data, m.size(), m.step.p[0], minMax);
	break;
	}
}





void BlobDetector(std::vector<ABox>& boxes, Mat& image, const unsigned int min_intensity, cv::Rect roi, const unsigned int max_intensity, const int max_boxsize_pixels, const double circularity_ratio) {
	if((roi.x + roi.width) > image.cols || (roi.y + roi.height) > image.rows) { 
		roi.width = roi.height = 0; 
	} 
	const int xs = roi.width > 0? roi.x: 0;
	const int ys = roi.height > 0? roi.y: 0;
	const int imcols = image.cols;
	const int imrows = image.rows;
	const int cols = roi.width > 0? (xs + roi.width): imcols;
	const int rows = roi.height > 0? (ys + roi.height): imrows;
	const unsigned short *data = (unsigned short *)image.data; // Mar.4 2015.
	ATLSMatrixvar<char> tracker;
	tracker.SetDimensions(imcols, imrows);
	for(int y1 = ys; y1 < rows; ++y1) {
		for(int x1 = xs; x1 < cols; ++x1) {
			if(tracker(x1, y1) > 0) {
				continue;
			}
			long long intensity = *(data + y1*imcols + x1);
			if(intensity > min_intensity) {
				tracker(x1, y1) = 1;
				size_t max_intensity_count = (intensity >= max_intensity? 1: 0);
				size_t intensity_count = 1;

				int x_max = x1;
				int y_max = y1;
				int x_min = x1;
				int y_min = y1;
				int x_max_last = x1;
				int x_min_last = x1;
				bool done = false;
				for(int y2 = y1; y2 < rows && !done; ++y2) {
					done = true;
					int x2;
					for(x2 = x1 + 1; x2 < cols; ++x2) {
						if(tracker(x2, y2) > 0) {
							break;
						}
						intensity = *(data + y2*imcols + x2);
						if(intensity > min_intensity) {
							done = false;
							if(x_max < x2) {
								x_max = x2;
							}
							tracker(x2, y2) = 1;
							if(intensity >= max_intensity) {
								++max_intensity_count; 
							}
							++intensity_count;
						}
						else
						if(x2 >= (x_max_last)) {
							break;
						}
					}
					x_max_last = x2;
					for(x2 = x1 - 1; x2 >= 0; --x2) {
						if(tracker(x2, y2) > 0) {
							break;
						}
						intensity = *(data + y2*imcols + x2);
						if(intensity > min_intensity) {
							done = false;
							if(x_min > x2) {
								x_min = x2;
							}
							tracker(x2, y2) = 1;
							if(intensity >= max_intensity) {
								++max_intensity_count;
							}
							++intensity_count;
						}
						else
						if(x2 <= (x_min_last)) {
							break;
						}
					}
					x_min_last = x2;

					if(!done) {
						y_max = y2;
					}
				}

				if((x_max - x_min) >= 3 && (y_max - y_min) >= 3 && max_intensity_count > (intensity_count * g_percent_maxintensity)) {
					int dx = x_max - x_min;
					int dy = y_max - y_min;

					if(dx <= max_boxsize_pixels && dy <= max_boxsize_pixels) {
						if((double)std::min(dx, dy) / (double)std::max(dx, dy) > circularity_ratio) {
							boxes.push_back(ABox(x_min, x_max + 1, y_min, y_max + 1, (int)(max_intensity * (double)max_intensity_count / (double)intensity_count + 0.49999)));
						}
					}
				}
			}
		}
	}
}

void BlobCentersEx(ABox& box, std::vector<ClusteredPoint>& points, Mat& image, const unsigned int min_intensity) {
	const int cols = image.cols;
	const int rows = image.rows;

	const unsigned short *data = (unsigned short *)image.data; // Mar.4 2015.

	double x_accumulator = 0;
	double y_accumulator = 0;
	long long counter = 0;

	const int y_max = box.y[1];
	const int y_min = box.y[0];
	const int x_max = box.x[1];
	const int x_min = box.x[0];
	//const double x_center = (x_max + x_min) / 2.0;
	//const double y_center = (y_max + y_min) / 2.0;
	//const double radius2 = pow(std::max((x_max - x_min) / 2.0, (y_max - y_min) / 2.0), 2);
	for(int y = y_min; y < y_max; ++y) {
		int x = x_min;
		const unsigned short *ptr = data + y*cols + x;
		for(; x < x_max; ++x) {
			long long intensity = *(ptr++);
			if(intensity > min_intensity) {
				//double dist = pow(x - x_center - 0.5, 2) + pow(y - y_center - 0.5, 2); 
				//if(dist <= radius2) {
					//x_accumulator += (double)x + 0.5;
					//y_accumulator += (double)y + 0.5;
					//++counter;
					x_accumulator += ((double)x + 0.5)*intensity;
					y_accumulator += ((double)y + 0.5)*intensity;
					counter += intensity;
				//}
				//else {
				//	counter += 0; 
				//}
			}
		}
	}
	if(counter) {
		points.push_back(ClusteredPoint(Point2d(((double)x_accumulator / counter), ((double)rows - (double)y_accumulator / counter)), box.intensity));
	}
}

inline bool AverageCenters(const std::vector<ClusteredPoint>& centers, ClusteredPoint& point, const ABox& box = ABox(), cv::Rect roi = cv::Rect()) {
	bool rc = false;
	if(centers.size() > 0) {
		point = centers[0];
		for(size_t j = 1; j < centers.size(); ++j) {
			point += centers[j];
		}
		point *= 1.0 / (double)centers.size();

		// some atempt to cope with an offset of circle center.
		// {R / sqrt(2)} = (box.x + box.y) / 8
		// {R / sqrt(2)} 1/10 = (box.x + box.y) / 80; but replacing 80 by 120 improved statistics. 
		const double scale_offset = (box.x[1] - box.x[0] + box.y[1] - box.y[0]) / 120.0;
		if(scale_offset > 0) {
			const double image_center_x = roi.width > 0? (roi.x + roi.width / 2.0): g_imageSize.width / 2.0;
			const double image_center_y = roi.height > 0? (roi.y + roi.height / 2.0): g_imageSize.height / 2.0;

			double effective_offset[2] = {scale_offset * (image_center_x - point.x) / image_center_x, scale_offset * (image_center_y - point.y) / image_center_y};
			point.x += effective_offset[0];
			point.y += effective_offset[1];
		}

		rc = true; 
	}
	return rc; 
}

inline bool BoxCenter(ABox& box, ClusteredPoint& point, Mat& image, const unsigned int min_intensity, cv::Rect roi) {
	bool rc = false;
	std::vector<ClusteredPoint> centers;
	centers.reserve(30);
	const unsigned int max_intensity = (unsigned int)(std::max(220 * g_bytedepth_scalefactor, min_intensity));
	for(unsigned int intensity = min_intensity; intensity <= max_intensity; intensity += (10 * g_bytedepth_scalefactor)) {
		BlobCentersEx(box, centers, image, intensity);
	}
	rc = AverageCenters(centers, point, box, roi); 
	return rc; 
}

void inline BlobCenters(std::vector<ABox>& boxes, std::vector<ClusteredPoint>& points, Mat& image, const unsigned int min_intensity, cv::Rect roi) {
	boxes.resize(0);
	points.resize(0); 
	boxes.reserve(10);
	points.reserve(10);
	BlobDetector(boxes, image, min_intensity, roi);
	for(auto& box : boxes) {
		ClusteredPoint point;
		if(BoxCenter(box, point, image, min_intensity, roi)) {
			points.push_back(point);
		}
	}
}




// Detection based on Laplacian of Gaussian. 

Mat_<double> LoG(double sigma, int ksize) { // Laplacian of Gaussian kernel
	const int cx = (int)(ksize - 1) / 2;
	const int cy = (int)(ksize - 1) / 2;
	const double sigmaSquare = pow(sigma, 2); 
	const double sigmaSquare2 = pow(sigmaSquare, 2); 
	Mat_<double> LoGkernel(ksize, ksize);
	double sumLoGkernel = 0;
	for(int x = 0; x < ksize; ++x) { 
		for(int y = 0; y < ksize; ++y) { 
			const int nx = x - cx; 
			const int ny = y - cy; 
			const double sumSquare = 0.5 * (nx * nx + ny * ny) / sigmaSquare; 
			const double val = (1.0 / sigmaSquare2) * (sumSquare - 1) * exp(-sumSquare);
			LoGkernel(x, y) = val; 
			sumLoGkernel += val; 
		} 
	} 
	LoGkernel /= sumLoGkernel; 
	return LoGkernel; 
}


double convolveLoG(const Mat& image, const size_t rowy, const size_t colx, const Mat_<double>& kmat) {
	const size_t knrows = kmat.rows;
	const size_t kncols = kmat.cols;
	size_t offsetrow = rowy - (knrows - 1) / 2; 
	size_t offsetcol = colx - (kncols - 1) / 2; 

	const size_t nrows = image.rows;
	const size_t ncols = image.cols;
	const unsigned short *data = ((const unsigned short *)image.data) + offsetrow * ncols;

	double val = 0;

	for(int k = 0; k < knrows; ++k, data += ncols) {
		const unsigned short *ptr = data + offsetcol;
		for(int j = 0; j < kncols; ++j, ++ptr) {
			val += kmat(k, j) * *ptr;
		}
	}

	return val; 
}



int64_t BlobLoG(std::vector<ABoxedblob>& blobs,
	const Point2d& aSeed, 
	const Mat& image, 
	const Mat_<double>& kmat, 
	const int row_limits[2], 
	const int col_limits[2],
	ATLSMatrixvar<signed short>& tracker, 
	ATLSMatrixvar<double>& tracker_value,
	const double max_LoG_factor) {

	blobs.resize(blobs.size() + 1);
	ABoxedblob& aBlob = blobs[blobs.size() - 1];
	aBlob._id = (int)blobs.size(); 

	const int aBlob_id = aBlob._id; 

	char rowarray_base[sizeof(ABoxedrow[401])];
	const int maxRow = 400;

	ABoxedrow* rowarray = (ABoxedrow*)rowarray_base;

	static const int horizincr[2] = {-1, 1};
	static const int verticalincr[2] = {-1, 1};

	static const int horizincr_stepfactor[2] = {1, 0};
	static const int verticalincr_stepfactor[2] = {0, 1};


	const size_t knrows = kmat.rows;
	const size_t kncols = kmat.cols;
	const size_t knrows2 = (knrows - 1) / 2;
	const size_t kncols2 = (kncols - 1) / 2;
	const size_t nrows = image.rows;
	const size_t ncols = image.cols;

	/*
	 a bright background can make LoG fail, so a value > 21 * g_bytedepth_scalefactor makes it work in that situation.
	*/
	//const double min_LoG_value = (7.0 / kncols) * (121.0 / 255.0) * image.at<unsigned short>((int)aSeed.y, (int)aSeed.x);
	//const double max_LoG_value = (23.0 / kncols) * (121.0 / 255.0) * image.at<unsigned short>((int)aSeed.y, (int)aSeed.x);
	const double min_LoG_value = (9.0 / kncols) * (121.0 / 255.0) * image.at<unsigned short>((int)aSeed.y, (int)aSeed.x);
	const double max_LoG_value = (max_LoG_factor / kncols) * (121.0 / 255.0) * image.at<unsigned short>((int)aSeed.y, (int)aSeed.x);

	int min_max_rows[2] = {200, 200/*g_max_boxsize_pixels, g_max_boxsize_pixels*/};

	int64_t qcounter_convolve = 0;

	for(int i = 0; i < 2; ++i) {
		// up then down
		int& currentrow = min_max_rows[i];
		rowarray[currentrow] = ABoxedrow((int)aSeed.y, (int)aSeed.x);
		rowarray[currentrow + verticalincr[i]]._rownum = -1;

		while ((currentrow > 0 && currentrow < maxRow) && !(rowarray[currentrow]._rownum == -1)) {
			ABoxedrow& arow = rowarray[currentrow]; 
			for(int j = 0; j < 2; ++j) {
				// left then right
				ARowend& arowend = arow._rowends[j]; 

				bool horizontal_done = false; 
				while(!horizontal_done) {
					const int now_col = arowend._column;  
					const int now_row = arow._rownum;
					for(int k = 0; k < 2; ++k) {
						// one step horiziontally then vertically
						if(arowend._processed[k] == false) {
							const int hincr = horizincr[j] * horizincr_stepfactor[k]; 
							const int vincr = verticalincr[i] * verticalincr_stepfactor[k]; 

							const int next_col = now_col + hincr; 
							const int next_row = now_row + vincr; 

							bool valid_next = next_col >= col_limits[0] && next_row >= row_limits[0] && next_col < col_limits[1] && next_row < row_limits[1]; 
							if(valid_next) {
								double& val = tracker_value(next_row, next_col);
								if(val == 0.0) {
									//val = convolveLoG(image, next_row, next_col, kmat);

									int64_t qcounter = MyQueryCounter();// +qcounter_delta;

									const size_t offsetrow = next_row - knrows2;
									const size_t offsetcol = next_col - kncols2;

									const unsigned short *data = ((const unsigned short *)image.data) + offsetrow * ncols;
									const double *kmat_data = (const double *)kmat.data;

									for(int k = 0; k < knrows; ++k, data += ncols) {
										const unsigned short *ptr = data + offsetcol;
										//for(int j = 0; j < kncols; ++j, ++ptr, ++kmat_data) {
										//	val += (*kmat_data) * (*ptr);
										//}
										int j = 0;
										while(j++ < kncols) {
											val += (*kmat_data++) * (*ptr++);
										}
									}

									qcounter_convolve += MyQueryCounter() - qcounter;
								}
								if(val > min_LoG_value && val < max_LoG_value) {
									if(vincr != 0) {
										const int next_vert = currentrow + vincr; 
										ABoxedrow& arow_next_vert = rowarray[next_vert];
										arow_next_vert._rownum = next_row; 
										arow_next_vert._rowends[0] = arow_next_vert._rowends[1] = next_col; 

										const int row_stop = next_vert + vincr; 
										if (row_stop >= 0 && row_stop < maxRow) {
											rowarray[row_stop]._rownum = -1;
										}
										arow._rowends[0]._processed[k] = true;
										arow._rowends[1]._processed[k] = true;
									}
									else {
										arowend._column = next_col;
									}

									signed short& tracker_id = tracker(next_row, next_col);
									if(tracker_id > 0) {
										if(tracker_id != aBlob_id) {
											blobs[tracker_id - 1].add_association(aBlob);
										}
									}
									else {
										tracker_id = aBlob_id;
									}
								}
								else 
								if(hincr != 0) {
									arowend._processed[k] = true;
									horizontal_done = true;
								}
							}
							else {
								horizontal_done = true;
							}
						}
					}
				}
			}
			currentrow += verticalincr[i]; 
		}
	}

	aBlob._rows.reserve(min_max_rows[1] - min_max_rows[0]);

	for(int j = std::max(0, min_max_rows[0]); j < min_max_rows[1]; ++j) {
		if(rowarray[j]._rownum > -1) {
			aBlob._rows.push_back(rowarray[j]);
		}
	}

	if(aBlob._rows.size() == 0) {
		blobs.resize(blobs.size() - 1);
	}

	return qcounter_convolve;
}

void MergeBlobs(std::vector<ABoxedblob>& blobs, const int id) {
	if(id <= 0) {
		return;
	}
	ABoxedblob& aBlob = blobs[id - 1];
	for(auto j : aBlob._associatedBlobs) {
		MergeBlobs(blobs, j);
	}

	std::vector<ABoxedrow> rows;
	for(auto j : aBlob._associatedBlobs) {
		ABoxedblob& other = blobs[j - 1];

		const size_t imax = aBlob._rows.size();
		const size_t kmax = other._rows.size();

		rows.clear(); 
		rows.reserve(imax + kmax);

		for(int i = 0, k = 0; i < imax || k < kmax;) {
			if(k >= kmax || (i < imax && aBlob._rows[i]._rownum < other._rows[k]._rownum)) {
				rows.push_back(aBlob._rows[i]);
				++i;
			}
			else
			if(i >= imax || (k < kmax && aBlob._rows[i]._rownum > other._rows[k]._rownum)) {
				rows.push_back(other._rows[k]);
				++k;
			}
			else {
				ABoxedrow aRow(aBlob._rows[i]._rownum);
				aRow._rowends[0]._column = std::min(aBlob._rows[i]._rowends[0]._column, other._rows[k]._rowends[0]._column);
				aRow._rowends[1]._column = std::max(aBlob._rows[i]._rowends[1]._column, other._rows[k]._rowends[1]._column);
				rows.push_back(aRow);
				++i; 
				++k; 
			}
		}

		aBlob._rows.swap(rows); 
		other._rows.clear();
	}

	aBlob._associatedBlobs.clear(); 
}

void RowsCenterEx(const std::vector<ABoxedrow>& rows, std::vector<ClusteredPoint>& points, Mat& image, const unsigned int min_intensity, const unsigned int avg_intensity) {
	const unsigned short *data = (unsigned short *)image.data; 

	double x_accumulator = 0;
	double y_accumulator = 0;

	const int imcols = image.cols;
	const int imrows = image.rows;

	long long counter = 0;

	for(auto& aRow : rows) {
		const int xl = aRow._rowends[0]._column;
		const int xr = aRow._rowends[1]._column;

		int y = aRow._rownum; 
		const unsigned short *ptr = data + aRow._rownum * imcols + xl;
		for(int x = xl; x <= xr; ++x, ++ptr) {
			long long intensity = *(ptr);
			if(intensity > min_intensity) {
				x_accumulator += ((double)x + 0.5)*intensity;
				y_accumulator += ((double)y + 0.5)*intensity;
				counter += intensity;
			}
		}
	}

	if(counter) {
		points.push_back(ClusteredPoint(Point2d(((double)x_accumulator / counter), ((double)imrows - (double)y_accumulator / counter)), avg_intensity));
	}
}


bool HoughCircleCenter4Points(const std::vector<Point2f>& pixels, const Point& offset, const int rows, const int cols, Point2d& center) {
	const int acc_box_x0 = offset.x << 2;
	const int acc_box_x1 = (offset.x + cols) << 2; // note: quad value is needed in order to accomodate quad precision of accumulator.
	const int acc_box_y0 = offset.y << 2;
	const int acc_box_y1 = (offset.y + rows) << 2;

	ATLSMatrixvar<long long> accumulator;
	accumulator.SetDimensions(rows << 2, cols << 2); // note: quad value is needed in order to accomodate quad precision of accumulator.
	for(auto& pixel : pixels) {
		double x1 = pixel.x;
		double y1 = pixel.y;
		for(auto& other : pixels) {
			if(&other != &pixel) {
				double x2 = other.x;
				double y2 = other.y;
				double xm = ((double)x1 + x2) * 2.0/* / 2.0*/; // note: quad value is needed in order to accomodate quad precision of accumulator. 
				double ym = ((double)y1 + y2) * 2.0/* / 2.0*/;
				double m;
				if(y2 != y1) {
					m = (x2 - x1) / (double)(y2 - y1);
				}
				else {
					m = 999999999;
				}
				if(abs(m) < 1) {
					double y0 = ym + (xm - acc_box_x0) * m;
					for(int x0 = acc_box_x0; x0 < acc_box_x1; ++x0) {
						int y = (int)(y0 + 0.49999);
						if(y >= acc_box_y0 && y < acc_box_y1) {
							accumulator(y - acc_box_y0, x0 - acc_box_x0)++;
						}
						y0 -= m;
					}
				}
				else {
					m = 1 / m;
					double x0 = xm + (ym - acc_box_y0) * m;
					for(int y0 = acc_box_y0; y0 < acc_box_y1; ++y0) {
						int x = (int)(x0 + 0.49999);
						if(x >= acc_box_x0 && x < acc_box_x1) {
							accumulator(y0 - acc_box_y0, x - acc_box_x0)++;
						}
						x0 -= m;
					}
				}
			}
		}
	}

	long long max_accumulator = 0;
	long long count = 1;
	long long x_max = 0;
	long long y_max = 0;

	for(int x = 0; x < cols << 2; ++x) {
		for(int y = 0; y < rows << 2; ++y) {
			long long val = accumulator(y, x);
			if(val > max_accumulator) {
				max_accumulator = val;
				x_max = x;
				y_max = y;
			}
		}
	}

	long long threshold_accumulator = (max_accumulator * 950) / 1000;
	count = 0;
	x_max = 0;
	y_max = 0;

	for(int x = 0; x < cols << 2; ++x) {
		for(int y = 0; y < rows << 2; ++y) {
			long long val = accumulator(y, x);
			if(val) {
				if(val > threshold_accumulator) {
					x_max += x*val;
					y_max += y*val;
					count += val;
				}
			}
		}
	}

	bool rc = false;

	if(count > 10) {
		center = Point2d(offset.x + (double)x_max / (double)(count << 2), offset.y + (double)y_max / (double)(count << 2));
		rc = true;
	}
	else {
		center = Point2d(-1, -1);
	}

	return rc;
}

bool RowsCenterEllipse(std::vector<ABoxedrow> rows, ClusteredPoint& point, Mat& image, const unsigned int min_intensity, const unsigned int avg_intensity, cv::Rect roi) {
	bool rc = false; 

	const unsigned short *data = (unsigned short *)image.data;

	double x_accumulator = 0;
	double y_accumulator = 0;

	const int imcols = image.cols;
	const int imrows = image.rows;

	const double image_center_x = roi.width > 0? (roi.x + roi.width / 2.0): g_imageSize.width / 2.0;
	const double image_center_y = roi.height > 0? (roi.y + roi.height / 2.0): g_imageSize.height / 2.0;

	std::vector<Point2f> pixels;
	long long counter = 0;

	double intensity = 0;

	int x_min = std::numeric_limits<int>::max();
	int x_max = std::numeric_limits<int>::min();
	int y_min = std::numeric_limits<int>::max();
	int y_max = std::numeric_limits<int>::min();

	for(auto& aRow : rows) {
		const int xl = aRow._rowends[0]._column;
		const int xr = aRow._rowends[1]._column;

		int y = aRow._rownum;
		const unsigned short *ptr = data + aRow._rownum * imcols + xl;
		for(int x = xl; x <= xr; ++x, ++ptr) {
			long long intensity = *(ptr);
			if(intensity > min_intensity) {
				x_accumulator += ((double)x + 0.5)*intensity;
				y_accumulator += ((double)y + 0.5)*intensity;
				counter += intensity;
			}
		}
		pixels.push_back(Point2f(xl + 0.5f, y + 0.5f));
		if(xl != xr) {
			pixels.push_back(Point2f(xr + 0.5f, y + 0.5f));
		}
		if(xl < x_min) {
			x_min = xl;
		}
		if(xr > x_max) {
			x_max = xr;
		}
		if(y < y_min) {
			y_min = y;
		}
		if(y > y_max) {
			y_max = y;
		}
	}

	Mat pointsf;
	Mat(pixels).convertTo(pointsf, CV_32F);
	RotatedRect ellipse_rect = fitEllipse(/*pixels*/pointsf); // Least squared distance algorithm.

	Point2d contour_center(ellipse_rect.center.x, ellipse_rect.center.y); 
	Point2d circle_center; 
	if(HoughCircleCenter4Points(pixels, Point(x_min, y_min), x_max - x_min, y_max - y_min, circle_center)) {
		contour_center += circle_center; 
		contour_center *= 0.5; 
	}


	//Khachiyan’s algorithm for the computation of minimum-volume enclosing ellipsoids
	if(pixels.size() > 14) {
		int N = (int)pixels.size();
		Mat_<double> Q(N, 3);
		Mat_<double> u(N, N, 0.0);
		double v = 1.0 / N;
		for(int r = 0; r < N; ++r) {
			Q(r, 0) = pixels[r].x;
			Q(r, 1) = pixels[r].y;
			Q(r, 2) = 1;
			u(r, r) = v;
		}
		int count = 0;
		double err = 1;

		Mat_<double> QT(3, N);
		transpose(Q, QT);

		while(err > 0.01 && count < N) {
			Mat_<double> X = QT * u * Q;
			Mat_<double> I(X.cols, X.rows);
			invert(X, I);
			Mat_<double> M = Q * I * QT;

			double maximum = M(0, 0);
			int jmax = 0;
			for(int j = 1; j < M.rows; ++j) {
				if(M(j, j) > maximum) {
					maximum = M(j, j);
					jmax = j;
				}
			}
			double step_size = (maximum - 3) / (3 * (maximum - 1));

			Mat_<double> new_u = (1 - step_size) * u;
			new_u(jmax, jmax) += step_size;

			double sum = 0;
			for(int j = 0; j < N; ++j) {
				sum += pow(new_u(j, j) - u(j, j), 2);
			}
			err = sqrt(sum);
			u = new_u;

			++count;
		}

		Point2d c;
		for(int r = 0; r < N; ++r) {
			c.x += pixels[r].x * u(r, r);
			c.y += pixels[r].y * u(r, r);
		}

		// some atempt to cope with an offset of circles center.
		// {R / sqrt(2)} = (box.x + box.y) / 8
		// {R / sqrt(2)} 1/10 = (box.x + box.y) / 80; but replacing 80 by 120 improved statistics. 
		const double scale_offset = (x_max - x_min + y_max - y_min) / 120.0;
		double effective_offset[2] = {scale_offset * (image_center_x - c.x) / image_center_x, scale_offset * (image_center_y - c.y) / image_center_y};

		double w = 0.01 / (err < 0.01? 0.01: err); 
		c.x = w * c.x + (1 - w) * ((double)(x_accumulator / counter + contour_center.x) / 2.0) - effective_offset[0];
		c.y = imrows - (w * c.y + (1 - w) * ((double)(y_accumulator / counter + contour_center.y) / 2.0)) + effective_offset[1];

		point = ClusteredPoint(c, avg_intensity);
		counter = 0; 

		rc = true; 
	}

	if(counter) {
		point = ClusteredPoint(Point2d(((double)x_accumulator / counter), ((double)imrows - (double)y_accumulator / counter)), avg_intensity);
		rc = true; 
	}

	return rc; 
}

inline bool RowsCenter(const std::vector<ABoxedrow>& rows, ClusteredPoint& point, Mat& image, const unsigned int min_intensity, const unsigned int avg_intensity, const ABox& box = ABox(), cv::Rect roi = cv::Rect()) {
	bool rc = false; 
	std::vector<ClusteredPoint> centers;
	centers.reserve(30);
	const unsigned int max_intensity = (unsigned int)(avg_intensity/*220 * g_bytedepth_scalefactor*/);
	for(unsigned int intensity = min_intensity; intensity < max_intensity; intensity += (10 * g_bytedepth_scalefactor)) {
		RowsCenterEx(rows, centers, image, intensity, avg_intensity);
	}
	rc = AverageCenters(centers, point, box, roi); 
	return rc;
}


void HoughLineContour(const std::vector<Point>& contour, const Point& offset, std::vector<Vec2f>& houghLines, const int max_rho, const int scale = 100) {
	ATLSMatrixvar<int> accumulator; 
	accumulator.SetDimensions(max_rho, 180); 

	for(auto point : contour) {
		point -= offset; 
		for(int m = 0; m < 180; ++m) {
			int rho = (int)((point.x * cos(m * CV_PI / 180.0) + point.y * sin(m * CV_PI / 180.0)) / scale + 0.49999);
			if(rho >= 0 && rho < max_rho) {
				accumulator(rho, m) += 1; 
			}
		}
	}

	for(int rho = 0; rho < max_rho; ++rho) {
		for(int m = 0; m < 180; ++m) {
			int cnt = accumulator(rho, m); 
			if(cnt > 16) {
				houghLines.push_back(Vec2f((float)rho, (float)(m * CV_PI / 180.0)));
			}
		}
	}
}

void AnisotropicDiffusion(Mat& x0, int anysotropicIntensity = 15, int bytedepth_scalefactor = 0) { // x0 becomes CV_32FC1
	if(bytedepth_scalefactor <= 0) {
		bytedepth_scalefactor = g_bytedepth_scalefactor;
	}

	int t = 0;

	double lambda = 0.25; // Defined in equation (7)
	double lambda_convergence_factor = 1.0; 
	double K = anysotropicIntensity * bytedepth_scalefactor; // defined after equation(13) in text

	double global_maxD = std::numeric_limits<double>::min();

	while(++t < 20 && lambda_convergence_factor > 0.1) {
		Mat D; // defined just before equation (5) in text
		Mat gradxX, gradyX; // Image Gradient t time 
		Sobel(x0, gradxX, CV_32F, 1, 0, 3);
		Sobel(x0, gradyX, CV_32F, 0, 1, 3);

		D = Mat::zeros(x0.size(), CV_32F);
		for(int i = 0; i < x0.rows; i++)
		for(int j = 0; j < x0.cols; j++) {
			double gx = gradxX.at<float>(i, j), gy = gradyX.at<float>(i, j);
			double d;
			if(i == 0 || i == x0.rows - 1 || j == 0 || j == x0.cols - 1) // conduction coefficient set to 1 p633 after equation 13
				d = 1;
			else
				d = 1.0 / (1.0 + (gx*gx + gy*gy) / (K*K)); // expression of g(gradient(I))
			D.at<float>(i, j) = (float)d;
		}

		double local_maxD = std::numeric_limits<double>::min();

		Mat x1 = Mat::zeros(x0.size(), CV_32FC1);

		for(int i = 1; i < x0.rows - 1; i++) {
			float *u1 = (float*)x1.ptr(i);
			u1++;
			for(int j = 1; j < x0.cols - 1; j++, u1++) {
				// Value of I at (i+1,j),(i,j+1)...(i,j)
				double ip10 = mat_get<double>(x0, (i + 1), j), i0p1 = mat_get<double>(x0, (i), j + 1);
				double im10 = mat_get<double>(x0, (i - 1), j), i0m1 = mat_get<double>(x0, (i), j - 1), i00 = mat_get<double>(x0, i, j);
				// Value of D at at (i+1,j),(i,j+1)...(i,j)
				double cp10 = D.at<float>(i + 1, j), c0p1 = D.at<float>(i, j + 1);
				double cm10 = D.at<float>(i - 1, j), c0m1 = D.at<float>(i, j - 1), c00 = D.at<float>(i, j);
				// Equation (7) p632
				double diff = ((cp10 + c00)*(ip10 - i00) + (c0p1 + c00)*(i0p1 - i00) + (cm10 + c00)*(im10 - i00) + (c0m1 + c00)*(i0m1 - i00));
				if(local_maxD < diff) {
					local_maxD = diff;
				}
				*u1 = float(i00 + lambda * lambda_convergence_factor * diff); // equation (9)
			}
		}
		x0 = x1;

		if(global_maxD < local_maxD) {
			global_maxD = local_maxD;
		}
		lambda_convergence_factor = local_maxD / global_maxD;
	}
}


void ScalePrecision(const std::vector<Point>& contour, std::vector<Point>& contourEx, const int scale = 100) {
	contourEx.reserve(contour.size() * 10);

	double invscale2 = 2.0 / scale; 
	double scale7 = scale / 7.0; 

	Point half_scale((scale >> 1), (scale >> 1));
	std::vector<Point>::const_iterator it = contour.cbegin();
	for(auto point : contour) {
		point *= scale;
		point += half_scale;
		contourEx.push_back(point);

		if(++it != contour.cend()) {
			Point point_next = ((*it) * scale) + half_scale;

			Point point_dist = point - point_next;
			double dist = std::sqrt(point_dist.ddot(point_dist));
			int np = (int)(dist * invscale2 + 0.49999);
			double inv_np = 1.0 / double(np); 
			for(int n = 1; n < np; ++n) {
				Point dither(range_rand(scale7) * range_rand(1.0) > 0.5? 1: -1, range_rand(scale7) * range_rand(1.0) > 0.5? 1: -1);
				Point inner = ((np - n) * point + n * point_next) * (inv_np);
				contourEx.push_back(inner + dither);
			}
		}
	}
}


template<typename T>
Point_<T> find_closest(const std::vector<Point_<T>>& points, const Point_<T>& approx_point) {
	double min_dist = std::numeric_limits<double>::max();
	Point_<T> min_point;
	for(auto& point : points) {
		Point2d point_dist = Point2d(point) - Point2d(approx_point);
		double dist = (point_dist.ddot(point_dist));
		if(dist < min_dist) {
			min_point = point;
			min_dist = dist;
		}
	}
	return min_point; 
}

double approximate_withQuad100(const std::vector<Point>& contour, std::vector<Point>& min_quad2i, std::vector<Point>& contour100) { // returns flattening; populates min_quad2i with 4 points scaled by 100.
	ScalePrecision(contour, contour100, 100);

	int flattening_cnt = 0; 
	double flattening = 0;

	RotatedRect minarea_rect = minAreaRect(/*contour*/contour100);
	double a = std::max(minarea_rect.size.width, minarea_rect.size.height);
	double b = std::min(minarea_rect.size.width, minarea_rect.size.height);
	double minarea_rect_flattening = (a - b) / a;
	flattening += minarea_rect_flattening;
	++flattening_cnt; 

	RotatedRect ellipse_rect = fitEllipse(/*contour*/contour100);
	a = std::max(ellipse_rect.size.width, ellipse_rect.size.height);
	b = std::min(ellipse_rect.size.width, ellipse_rect.size.height);
	double ellipse_rect_flattening = (a - b) / a;
	flattening += ellipse_rect_flattening;
	++flattening_cnt;

	flattening /= (double)flattening_cnt;


	//std::vector<Point2f> minarea_rect2f(4);
	//minarea_rect.points(&minarea_rect2f[0]);

	//for(auto& approx_point : minarea_rect2f) {
	//	approx_point = (approx_point * 2.0 + Point2f(find_closest(contour100, Point(approx_point))) * 2.0) * (1.0f / 4.0f);
	//}
	//for(auto& point : minarea_rect2f) {
	//	min_quad2i.push_back(point);
	//}


	std::vector<Point2f> ellipse_rect2f(4);
	ellipse_rect.points(&ellipse_rect2f[0]); // points ==> ellipse_rect2f

	/*
	this moves closer the rectangle's points to real points. 
	*/
	for(auto& approx_point : ellipse_rect2f) {
		approx_point = (approx_point * 2.0 + Point2f(find_closest(contour100, Point(approx_point))) * 2.0) * (1.0f / 4.0f);
	}
	for(auto& point : ellipse_rect2f) {
		min_quad2i.push_back(point);
	}
	return flattening;


	//std::vector<Point2f> ellipse_rect2f(4);
	//ellipse_rect.points(&ellipse_rect2f[0]);

	//for(auto& approx_point : minarea_rect2f) {
	//	approx_point = (approx_point * 2.0 + Point2f(find_closest(contour100, Point(approx_point))) * 2.0) * (1.0f / 4.0f);
	//}
	//for(auto& approx_point : ellipse_rect2f) {
	//	approx_point = (approx_point * 2.0 + Point2f(find_closest(contour100, Point(approx_point))) * 2.0) * (1.0f / 4.0f);
	//}

	//min_quad2i.resize(0);
	//min_quad2i.reserve(4);

	//double angle_diff = std::abs(minarea_rect.angle - ellipse_rect.angle); 

	//ellipse_rect_flattening = 0; // 2016-06-29 ellipse often gets rotated, so use it only when they almost coincide. 

	//if(angle_diff < 3 || (angle_diff > 87 && angle_diff < 93)) {
	//	for(auto& point1 : ellipse_rect2f) {
	//		Point2f point2 = find_closest(minarea_rect2f, point1);
	//		Point point = Point((point1 + point2) * 0.5f);
	//		min_quad2i.push_back(point);
	//	}
	//}
	//else {
	//	if(minarea_rect_flattening < ellipse_rect_flattening) {
	//		for(auto& point : ellipse_rect2f) {
	//			min_quad2i.push_back(point);
	//		}
	//	}
	//	else {
	//		for(auto& point : minarea_rect2f) {
	//			min_quad2i.push_back(point);
	//		}
	//	}
	//}

	//return flattening;
}

void cornerSubPixGravity(Mat& image, std::vector<Point2f>& corners, Size winSize, const unsigned int min_intensity, const unsigned int avg_intensity, TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 5, 0.1)) {
	std::vector<ABoxedrow> rows; 
	double lambda = 1.0 / sqrt(pow(winSize.height, 2) + pow(winSize.width, 2)); 
	rows.reserve(winSize.height * 2 + 1);
	for(auto& corner : corners) {
		const int nSteps = (criteria.type & TermCriteria::MAX_ITER) != 0? criteria.maxCount: 1;
		for(int n = 0; n < nSteps; ++n) {
			int columns[2] = {(int)corner.x - winSize.width, (int)ceil(corner.x) + winSize.width};
			if(columns[0] < 0) {
				columns[0] = 0;
			}
			if(columns[1] >= image.cols) {
				columns[1] = image.cols - 1;
			}
			rows.resize(0);
			for(int r = (int)corner.y - winSize.height; r < (int)ceil(corner.y + winSize.height); ++r) {
				if(r >= 0 && r < image.rows) {
					rows.push_back(ABoxedrow(r, columns[0], columns[1]));
				}
			}
			Point2f prevCorner = corner;
			ClusteredPoint point;
			if(RowsCenter(rows, point, image, min_intensity, avg_intensity)) {
				point.y = image.rows - point.y; 
				corner += (corner - (Point2f)point) * lambda;
			}
			if((criteria.type & TermCriteria::EPS) != 0) {
				Point2f diff = corner - prevCorner;
				if(sqrt(diff.ddot(diff)) < criteria.epsilon) {
					break;
				}
			}
		}
	}
}

void subPixels(Mat& crop, std::vector<Point2f>& corners, int ws) {
	Mat dilatedImage;
	cv::resize(crop, dilatedImage, cv::Size(0, 0), 3, 3, INTER_LINEAR);

	for(auto& corner : corners) {
		corner *= 3.0;
	}
	ws *= 3;

	Mat subpixelImage;
	dilatedImage.convertTo(subpixelImage, CV_16UC1);
	cornerSubPixGravity(subpixelImage, corners, Size(ws, ws), 0, 1);

	//cv::dilate(dilatedImage, dilatedImage, getStructuringElement(MORPH_ELLIPSE, Size(ws, ws)));
	//cv::normalize(dilatedImage, subpixelImage, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	//subpixelImage = mat_convert2byte(dilatedImage);
	//cornerSubPix(subpixelImage, corners, Size(ws, ws)/*window size*/, Size(-1, -1)/*no zero zone*/, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 5, 0.1));

	for(auto& corner : corners) {
		corner *= 1.0 / 3.0;
	}
}


// FIR15 low-pass least-squares Fpass Fpass 0.4, Fstop 2.0, Fs 25 (0.1, Fstop 0.5, Fs 6.25)
static double fir_h1[15] = { 0.021650121520005953, 0.035527801315147732, 0.051061255474874737, 0.06688253424484937, 0.081431003402810065, 0.093166162384329704, 0.1007883612361113, 0.10343054711643812, 0.1007883612361113, 0.093166162384329704, 0.081431003402810065, 0.06688253424484937, 0.051061255474874737, 0.035527801315147732, 0.021650121520005953 };
static double fir_h_gain1 = 1.004445026;
static double fir_h_lowpass1 = 1;

// FIR15 low-pass equiripple Fpass Fpass 1.0, Fstop 3.0, Fs 25 (0.25, Fstop 0.75, Fs 6.25)
static double fir_h2[15] = { -0.032640248918755936, 0.010310761506497982, 0.030918235414101738, 0.061797451018658113, 0.097450419366571825, 0.13069501065073361, 0.15427159413808236, 0.16277411805378134, 0.15427159413808236, 0.13069501065073361, 0.097450419366571825, 0.061797451018658113, 0.030918235414101738, 0.010310761506497982, -0.032640248918755936 };
static double fir_h_gain2 = 1.068380564;
static double fir_h_lowpass2 = 1;

// FIR31 low-pass equiripple Fpass 0.1, Fstop 2.0, Fs 25
static double fir_h3[31] = {
-0.0019405441221921158, 
 0.00021290204002300688, 
 0.0014300506394972258, 
 0.0036700073983595501, 
 0.0071036510002612917, 
 0.011841436920247922, 
 0.017888775504615775, 
 0.025129802358795174, 
 0.033303292007835543, 
 0.042017094622616179, 
 0.050779635536499383, 
 0.059026098905632089, 
 0.066187174918655459, 
 0.071734765422246063, 
 0.075247964514827154, 
 0.076449266526105675, 
 0.075247964514827154, 
 0.071734765422246063, 
 0.066187174918655459, 
 0.059026098905632089, 
 0.050779635536499383, 
 0.042017094622616179, 
 0.033303292007835543, 
 0.025129802358795174, 
 0.017888775504615775, 
 0.011841436920247922, 
 0.0071036510002612917, 
 0.0036700073983595501, 
 0.0014300506394972258, 
 0.00021290204002300688, 
-0.0019405441221921158 
}; 
static double fir_h_gain3 = 1.003713; 
static double fir_h_lowpass3 = 1;


// FIR31 low-pass equiripple Fpass 0.2, Fstop 2.0, Fs 25
static double fir_h4[31] = {
-0.0059597715268290379, 
-0.0033255066471542527, 
-0.0030943467787952726, 
-0.0016883473101459237, 
 0.0012622984347179019, 
 0.0060085788655133074, 
 0.012668745713483077, 
 0.021153442438654222, 
 0.03117063212795055,  
 0.042206945980534413, 
 0.053572753037717545, 
 0.064465905535645782, 
 0.074052339267823022, 
 0.081550732685986824, 
 0.086327349723932259, 
 0.087966531488072763, 
 0.086327349723932259, 
 0.081550732685986824, 
 0.074052339267823022, 
 0.064465905535645782, 
 0.053572753037717545, 
 0.042206945980534413, 
 0.03117063212795055,  
 0.021153442438654222, 
 0.012668745713483077, 
 0.0060085788655133074, 
 0.0012622984347179019, 
-0.0016883473101459237, 
-0.0030943467787952726, 
-0.0033255066471542527, 
-0.0059597715268290379 
}; 
static double fir_h_gain4 = 1.008710035; 
static double fir_h_lowpas4s = 1;


// FIR31 low-pass equiripple Fpass 0.4, Fstop 2.0, Fs 25
static double fir_h5[31] = {
-0.0099292582092195599,
-0.0067525236171752726,
-0.0075324671209586963,
-0.0069516201113122436,
-0.0044812383329006552,
 0.00026980560047938954,
 0.0075264138110493987,
 0.017240760690068219,
 0.029072637531392994,
 0.042388997822594157,
 0.056319462751242541,
 0.069821830791183565,
 0.081797918604640182,
 0.091212125369844924,
 0.097226530964753782,
 0.099293523239023393,
 0.097226530964753782,
 0.091212125369844924,
 0.081797918604640182,
 0.069821830791183565,
 0.056319462751242541,
 0.042388997822594157,
 0.029072637531392994,
 0.017240760690068219,
 0.0075264138110493987,
 0.00026980560047938954,
-0.0044812383329006552,
-0.0069516201113122436,
-0.0075324671209586963,
-0.0067525236171752726,
-0.0099292582092195599
};
static double fir_h_gain5 = 1.013752;
static double fir_h_lowpass5 = 1;


// FIR31 low-pass equiripple Fpass 1.0, Fstop 2.0, Fs 25
static double fir_h6[31] = {
 0.0061473588925496925,
-0.024828335051403173,
-0.018123887209341438,
-0.018357282045492977,
-0.01785715861237289,
-0.014348109109504927,
-0.00703846736991969,
 0.0043366950282197397,
 0.019430567915095332,
 0.037378022312583602,
 0.05687086124862379,
 0.076272459872352857,
 0.093801399975206437,
 0.10777147244930367,
 0.11676932516015852,
 0.11987779266255399,
 0.11676932516015852,
 0.10777147244930367,
 0.093801399975206437,
 0.076272459872352857,
 0.05687086124862379,
 0.037378022312583602,
 0.019430567915095332,
 0.0043366950282197397,
-0.00703846736991969,
-0.014348109109504927,
-0.01785715861237289,
-0.018357282045492977,
-0.018123887209341438,
-0.024828335051403173,
 0.0061473588925496925
}; 
static double fir_h_gain6 = 0.95632764; 
static double fir_h_lowpass6 = 1;


/*
FIR filter designed with
 http://t-filter.appspot.com

sampling frequency: 25 Hz

* 0Hz - 2Hz
  gain = 1.210707
  desired ripple = 5 dB
  actual ripple = 3.7160174286567322 dB

* 4Hz - 12.5Hz
  gain = 0
  desired attenuation = -40 dB
  actual attenuation = -40.97956469696445 dB
*/
static double fir_h7[17] = {
	-0.010933361502522962,
	-0.014672567998546163,
	-0.011698583129917173,
	0.007782962239944857,
	0.04758542654735151,
	0.10347621545687319,
	0.16277625120028424,
	0.20832353497766573,
	0.22542732058293752,
	0.20832353497766573,
	0.16277625120028424,
	0.10347621545687319,
	0.04758542654735151,
	0.007782962239944857,
	-0.011698583129917173,
	-0.014672567998546163,
	-0.010933361502522962
};
static double fir_h_gain7 = 1.210707;
static double fir_h_lowpass7 = 1;



//template<typename T>
//void lowpassFilterContour(const std::vector<Point_<T>>& aux/*in*/, std::vector<Point_<T>>& contour) {
//	double* l_fir_h = fir_h7;
//	double l_fir_h_gain = fir_h_gain7;
//	const int NFilter = ARRAY_NUM_ELEMENTS(fir_h7);
//	const int NFilter2 = NFilter / 2;
//
//	if (contour.size() != aux.size()) {
//		contour.resize(aux.size()); 
//	}
//
//	const int L = (int)contour.size();
//	const int N = (contour[0] == contour[L - 1]) ? (L - 1) : L;
//
//	for (int k = 0, m = NFilter2; k < N; ++k, ++m) {
//		Point2d point(0, 0);
//		for (int j = 0, i = k + NFilter - 1; j < NFilter; ++j, --i) {
//			point += Point2d(aux[i % N]) * l_fir_h[j];
//		}
//		point.x /= l_fir_h_gain;
//		point.y /= l_fir_h_gain;
//
//		if (T(1 + 0.1) == T(1)) {
//			contour[m % N].x = (T)(point.x + 0.49999);
//			contour[m % N].y = (T)(point.y + 0.49999);
//		}
//		else {
//			contour[m % N].x = (T)(point.x);
//			contour[m % N].y = (T)(point.y);
//		}
//	}
//
//	if (L > N) {
//		contour[N] = contour[0];
//	}
//}

template<typename T>
void filterContour(std::vector<Point_<T>>& contour, std::vector<Point2d>& shocks) {
	double* l_fir_h = fir_h6;
	double l_fir_h_gain = fir_h_gain6;
	const int NFilter = ARRAY_NUM_ELEMENTS(fir_h6);
	const int NFilter2 = NFilter/2;


	const int L = (int)contour.size();
	const int N = (contour[0] == contour[L - 1]) ? (L - 1) : L;

	if (L < NFilter) {
		shocks.resize(L);
		return; 
	}

	std::vector<Point_<T>> aux(L);
	std::vector<Point2d> aux_shocks(L);

	// measurement: apply FIR filter. 
	// calculated point is in the center of filter's transfer function. 
	// NFilter/2 future points, and NFilter/2 past points. 

	// movement: use original point. 
	// location of point corresponds to the calculated measurement. 

	Mat_<double> I(2, 2);
	I(0, 1) = 0;
	I(1, 0) = 0;
	I(0, 0) = 1;
	I(1, 1) = 1;

	Mat_<double> Zet(2, 2);  // movement variance
	Zet(0, 0) = 0;
	Zet(1, 1) = 0;
	Zet(0, 1) = 0;
	Zet(1, 0) = 0;

	Mat_<double> maux;
	Mat_<double> Q;
	Mat_<double> R;
	Mat_<double> K(2, 2);
	Mat_<double> X(2, 2);
	Mat_<double> Z(2, 1);

	Mat_<double> pastMeasurements((int)NFilter2, 2);

	Mat_<double> futureMovements((int)NFilter2, 2);

	double max_kval = 1;

	for (int k = 0, m = NFilter2; k < N + NFilter2; ++k, ++m) {
		Point2d point(contour[m % N]);
		Point2d shock; 

		Point2d futurePoint(contour[(k + NFilter + /*NFilter2 + */1) % N]);
		futureMovements(m % NFilter2, 0) = futurePoint.x;
		futureMovements(m % NFilter2, 1) = futurePoint.y;

		Point2d measurement(0, 0);
		//l_fir_h_gain = 0;
		for (int j = 0, i = k + NFilter - 1; j < NFilter; ++j, --i) {
			measurement += Point2d(contour[i % N]) * l_fir_h[j];
			//measurement += Point2d(contour[i % N]);
			//++l_fir_h_gain;
		}

		measurement.x /= l_fir_h_gain;
		measurement.y /= l_fir_h_gain;

		pastMeasurements(m % NFilter2, 0) = measurement.x;
		pastMeasurements(m % NFilter2, 1) = measurement.y;

		shock = -point + measurement;

		// measurement: calculate variance of NFilter/2 last points -> R
		if (m >= (NFilter - 1)) {

			//std::ostringstream ostr;

			cv::calcCovarMatrix(futureMovements, Q, maux = Mat(), CovarFlags::COVAR_NORMAL | CovarFlags::COVAR_ROWS);
			//ostr << "Q(0,0):" << Q(0, 0) << " Q(1,0):" << Q(1, 0) << " Q(0,1):" << Q(0, 1) << " Q(1,1):" << Q(1, 1) << std::endl;
			//Q(0, 1) = 0;
			//Q(1, 0) = 0;

			Zet += Q;
			//ostr << "Zet(0,0):" << Zet(0, 0) << " Zet(1,0):" << Zet(1, 0) << " Zet(0,1):" << Zet(0, 1) << " Zet(1,1):" << Zet(1, 1) << std::endl;

			cv::calcCovarMatrix(pastMeasurements, R, maux = Mat(), CovarFlags::COVAR_NORMAL | CovarFlags::COVAR_ROWS);
			//ostr << "R(0,0):" << R(0, 0) << " R(1,0):" << R(1, 0) << " R(0,1):" << R(0, 1) << " R(1,1):" << R(1, 1) << std::endl;
			//R(0, 1) = 0;
			//R(1, 0) = 0;

			double invConditionNumber = cv::invert(Zet + R, X, DECOMP_SVD);
			if (invConditionNumber < 0.001) {
				//ostr << "invConditionNumber:" << invConditionNumber << std::endl;
				point = measurement;
			}
			else {
				K = Zet * X;
				//ostr << "X(0,0):" << X(0, 0) << " X(1,0):" << X(1, 0) << " X(0,1):" << X(0, 1) << " X(1,1):" << X(1, 1) << std::endl;
				//ostr << "K(0,0):" << K(0, 0) << " K(1,0):" << K(1, 0) << " K(0,1):" << K(0, 1) << " K(1,1):" << K(1, 1) << std::endl;
				for (int i = 0; i < 2; ++i) {
					for (int j = 0; j < 2; ++j) {
						auto kval = std::abs(K(i, j));
						if (kval > max_kval) {
							max_kval = kval;
						}
					}
				}
				if (max_kval > 1) {
					for (int i = 0; i < 2; ++i) {
						for (int j = 0; j < 2; ++j) {
							K(i, j) /= max_kval;
						}
					}

					//max_kval -= 0.1;
					//if (max_kval < 1) {
					//	max_kval = 1;
					//}
					max_kval = 1;
				}
				K(0, 1) = 0;
				K(1, 0) = 0;

				//ostr << "K(0,0):" << K(0, 0) << " K(1,1):" << K(1, 1) << std::endl;

				Z(0, 0) = shock.x;
				Z(1, 0) = shock.y;
				//ostr << "Z(0,0):" << Z(0, 0) << " Z(1,0):" << Z(1, 0) << std::endl;

				Z = K * Z;
				//ostr << "Z(0,0):" << Z(0, 0) << " Z(1,0):" << Z(1, 0) << std::endl;

				shock.x = Z(0, 0);
				shock.y = Z(1, 0);

				point.x += shock.x;
				point.y += shock.y;

				Zet = (I - K) * Zet;
				//ostr << "Zet(0,0):" << Zet(0, 0) << " Zet(1,0):" << Zet(1, 0) << " Zet(0,1):" << Zet(0, 1) << " Zet(1,1):" << Zet(1, 1) << std::endl;
				//ostr << std::endl;
			}

			//printf("%s", ostr.str().c_str());
		}
		else {
			point = measurement;
		}
		//point = measurement;


		if (T(1 + 0.1) == T(1)) {
			aux[m % N].x = (T)(point.x + 0.49999);
			aux[m % N].y = (T)(point.y + 0.49999);
		}
		else {
			aux[m % N].x = (T)(point.x);
			aux[m % N].y = (T)(point.y);
		}

		aux_shocks[m % N] = shock;
	}

	if (L > N) {
		aux[N] = aux[0];
		aux_shocks[N] = aux_shocks[0];
	}

	contour.swap(aux); 
	shocks.swap(aux_shocks); 
}

template<typename T>
void smoothContour(std::vector<Point_<T>>& contour) {
	for (int j = 1; j < contour.size(); ++j) {
		while (contour[j] == contour[j - 1]) {
			contour.erase(contour.begin() + j); 
			if (j == contour.size()) {
				break; 
			}
		}
	}
	std::vector<Point2d> shocks[2];

	std::vector<Point_<T>> contour_reversed(contour.crbegin(), contour.crend());

	filterContour(contour, shocks[0]);
	filterContour(contour_reversed, shocks[1]);

	const size_t L = contour.size();
	for (int j = 0; j < L; ++j) {
		int k = L - j - 1;
		if (cv::norm(shocks[0][j]) > cv::norm(shocks[1][k])) {
			contour[j] = contour_reversed[k];
		}
	}
}

template<typename T>
void projectContour(std::vector<Point_<T>>& contour) {
	smoothContour(contour);
	for (size_t c = contour.size(), nc = c - 1; c > nc && nc > 0; nc = contour.size()) {
		c = nc; 
		linearizeContour(contour, 1, 7);
	}
}





template<typename T>
void polyspaceProjection(std::vector<Point_<T>>& contour, std::vector<Point2d>& shocks) {
	const size_t L = (int)contour.size();
	const size_t N = (contour[0] == contour[L - 1]) ? (L - 1) : L;

	constexpr size_t M = 17;
	constexpr size_t M2 = M / 2;

	if (X_XXI_X._a_dimension != M || X_XXI_X._b_dimension != M) {
		Prefetch_PolyspaceProjection(X_XXI_X, M, 1);
		Prefetch_PolyspaceProjection(NEG_X_XXI_X, M, -1);
	}

	std::vector<Point_<T>> aux(L);
	std::vector<Point2d> aux_shocks(L);

	for (size_t j = M2, k = M, c = 0; c < N; ++j, ++k, ++c) {
		Point2d shock[2];

		Point2d point;

		size_t jN = j % N;

		Point2d pointOriginal = contour[jN];

		point.x = point.y = 0;
		for (size_t n = j - M2, c = 0; c < M; ++n, ++c) {
			point.x += X_XXI_X(M2, c) * contour[n % N].x;
			point.y += X_XXI_X(M2, c) * contour[n % N].y;
		}

		shock[0] = point - pointOriginal;

		point.x = point.y = 0;
		for (size_t n = j - M2, c = 0; c < M; ++n, ++c) {
			point.x += NEG_X_XXI_X(M2, c) * contour[n % N].x;
			point.y += NEG_X_XXI_X(M2, c) * contour[n % N].y;
		}

		shock[1] = point - pointOriginal;

		if (cv::norm(shock[0]) < cv::norm(shock[1])) {
			point = shock[0] + pointOriginal;
		}

		if (T(1 + 0.1) == T(1)) {
			aux[jN].x = (T)(point.x + 0.49999);
			aux[jN].y = (T)(point.y + 0.49999);
		}
		else {
			aux[jN].x = (T)(point.x);
			aux[jN].y = (T)(point.y);
		}

		aux_shocks[jN] = Point2d(aux[jN]) - pointOriginal;
	}

	if (L > N) {
		aux[N] = aux[0];
		aux_shocks[N] = aux_shocks[0];
	}

	contour.swap(aux);
	shocks.swap(aux_shocks);
}


template<typename T>
void projectContour2(std::vector<Point_<T>>& contour) {

	std::vector<Point2d> shocks[2];

	std::vector<Point_<T>> contour_reversed(contour.crbegin(), contour.crend());

	polyspaceProjection(contour, shocks[0]);
	polyspaceProjection(contour_reversed, shocks[1]);

	const size_t L = contour.size();
	for (int j = 0; j < L; ++j) {
		int k = L - j - 1;
		if (cv::norm(shocks[0][j]) > cv::norm(shocks[1][k])) {
			contour[j] = contour_reversed[k];
		}
	}

	for (size_t c = contour.size(), nc = c - 1; c > nc && nc > 0; nc = contour.size()) {
		c = nc;
		linearizeContour(contour, 1, 7);
	}
}




struct AContainingBox {
	double x[2] = { 0, 0 };
	double y[2] = { 0, 0 };
	bool IsValid() const {
		return x[0] < x[1] && y[0] < y[1];
	}
	Point2d endpoints[2] = { { 0, 0 }, { 0, 0 } };
	void Add(Point2d aPoint) {
		int isEndpoint = 0;

		if (aPoint.x <= x[0]) {
			isEndpoint |= 1; // on left
			x[0] = aPoint.x;
		}
		else
		if (aPoint.x >= x[1]) {
			isEndpoint |= 2; // on right
			x[1] = aPoint.x;
		}

		if (aPoint.y <= y[0]) {
			isEndpoint |= 4; // on bottom
			y[0] = aPoint.y;
		}
		else
		if (aPoint.y >= y[1]) {
			isEndpoint |= 8; // on top
			y[1] = aPoint.y;
		}

		if (isEndpoint & (1 | 4)) {
			endpoints[0] = aPoint;
		}
		else
		if (isEndpoint & (2 | 8)) {
			endpoints[1] = aPoint;
		}
	}
};

Point round2dPoint(Point2d p) {
	Point r; 
	r.x = (int)floor(p.x + 0.5); 
	r.y = (int)floor(p.y + 0.5);
	return r;
}

void printPoint(const std::string& prefix, Point p) {
	std::ostringstream ostr; 
	ostr << prefix << ':' << p.x << ',' << p.y << std::endl; 
	printf("%s", ostr.str().c_str()); 
}

template<typename T>
void fitLine2Segment(std::vector<Point2d> &segment, std::vector<Point_<T>>& aux) {
	Vec4f aLine;
	fitLine(segment, aLine, DistanceTypes::DIST_L2, 0, 0.001, 0.001);
	Point2d norm(-aLine[1], aLine[0]); // normal (unit length now, but is going to be offset).
	Point2d mean;
	mean.x = aLine[2];
	mean.y = aLine[3];

	norm = norm * mean.ddot(norm); // make it normal offset

	Point2d colVec(aLine[0], aLine[1]); // collinear normalized

	// find end points of segment

	// Project end points of original set on collinear vector
	Point2d first = colVec * colVec.ddot(segment[0]) + norm;
	Point2d second = colVec * colVec.ddot(segment[segment.size() - 1]) + norm;

	//aux.push_back(first);
	//aux.push_back(second);

	aux.push_back(round2dPoint(first));
	aux.push_back(round2dPoint(second));
}

template<typename T>
void linearizeContour(std::vector<Point_<T>>&contour, double stepSize, const size_t maxSegmentSize) {
	// find first point that has its both neighbours farther than stepSize
	if (contour.size() < 3) {
		contour.resize(0); 
		return; 
	}
	size_t n = contour.size() - 1;
	const size_t N = contour[0] == contour[n]? n: n+1;
	const double D = pow(stepSize, 2) * 2; 
	bool prevIsOk = false; 
	Point2f aPoint = contour[0];
	Point2f aNext;
	size_t k;
	for (k = 1; k < N; ++k) {
		aNext = contour[k % N]; 
		double d = pow(aPoint.x - aNext.x, 2) + pow(aPoint.y - aNext.y, 2); 
		if (d > D) {
			if (prevIsOk) {
				break; 
			}
			prevIsOk = true; 
		}
		else {
			prevIsOk = false;
		}
		aPoint = aNext;
	}

	k = k % N; 

	std::vector<Point_<T>> aux;
	aux.push_back(aPoint);

	aPoint = aNext;
	size_t m;
	std::vector<Point2d> segment;
	for (m = k + 1; (m % N) != k; ++m) {
		aNext = contour[m % N];
		double d = pow(aPoint.x - aNext.x, 2) + pow(aPoint.y - aNext.y, 2);
		if (d == 0) {
		}
		else 
		if (d > D || segment.size() == maxSegmentSize) {
			if (segment.size() == 0) {
				aux.push_back(aPoint);
			}
			else
			if (segment.size() == 2) {
				aux.push_back(round2dPoint((segment[0] + segment[1]) * 0.5));
			}
			else {
				fitLine2Segment(segment, aux); 
			}
			segment.resize(0); 
		}
		else {
			if (segment.size() == 0) {
				segment.push_back(aPoint);
			}
			segment.push_back(aNext);
		}
		aPoint = aNext;
	}
	if (segment.size()) {
		fitLine2Segment(segment, aux);
	}

	if (aux.size() < 3) {
		aux.resize(0); 
	}

	contour.swap(aux); 

	n = contour.size() - 1;
	if (contour.size() && contour[0] != contour[n]) {
		contour.push_back(contour[0]);
	}
}

void linearizeContour(std::vector<long>& x, std::vector<long>& y, double stepSize, const size_t maxSegmentSize) {
	std::vector<Point2d> contour(x.size()); 
	size_t N = x.size(); 
	for (size_t j = 0; j < N; ++j) {
		contour[j].x = x[j]; 
		contour[j].y = y[j];
	}
	for (size_t c = contour.size(), nc = c - 1; c > nc && nc > 0; nc = contour.size()) {
		c = nc;
		linearizeContour(contour, stepSize, maxSegmentSize);
	}
	N = contour.size(); 
	x.resize(N);
	y.resize(N);
	for (size_t j = 0; j < N; ++j) {
		x[j] = contour[j].x;
		y[j] = contour[j].y;
	}
}

bool fitRectangleToPoints(const std::vector<Point>& contour, std::vector<Point>& corners/*in/out*/, double scale = 100) {
	scale = 1.0 / scale;

	// 1. for each point from contour find a closest segment from corners, i.e. do group the points by segments . 
	Point2d PA[4]; // points A of segments
	Point2d PB[4]; // points B of segments

	for(int j = 0; j < 4; ++j) {
		PA[j] = corners[j] * scale; 
		PB[j] = corners[j < 3? (j + 1): 0] * scale;
	}

	//Point2d half_scale(0.5, 0.5);

	std::vector<Point2f> groups[4]; 

	for(auto& point : contour) { 
		double prev_min_dist = 0; 
		double min_dist = std::numeric_limits<double>::max(); 
		int min_idx = -1; 
		Point2d point2d = Point2d(point) * scale/* + half_scale*/;
		for(int j = 0; j < 4; ++j) { 
			double dist = DistanceSquareFromPointToSegment(point2d, PA[j], PB[j]);
			if(dist < min_dist) {
				prev_min_dist = min_dist; 
				min_dist = dist; 
				min_idx = j; 
			}
			else 
			if(dist < prev_min_dist) {
				prev_min_dist = dist;
			}
		}
		if(min_dist / prev_min_dist < 0.33) {
			groups[min_idx].push_back(point2d);
		}
	}
	for(int n = 0; n < 4; ++n) {
		std::vector<Point2f>& points = groups[n];
		if(points.size() < 10) {
			return false; 
		}
	} 

	// 2. find line's equation for each group by building normal to principal direction. 

	Point2d N2[4]; //normals
	double D2[4]{}; //offsets

	for(int n = 0; n < 4; ++n) {
		std::vector<Point2f>& points = groups[n];

		Vec4f aLine; 
		fitLine(points, aLine, DistanceTypes::DIST_L2, 0, 0.001, 0.001);

		N2[n].x = -aLine[1]; 
		N2[n].y = aLine[0];

		Point2f mean;
		mean.x = aLine[2];
		mean.y = aLine[3];

		D2[n] = mean.ddot(N2[n]);
	}

	// 3. intersect the lines. 

	double invscale = 1.0 / scale; 

	for(int j = 0, k = 1; j < 4; ++j, k = k < 3? (k + 1): 0) {
		Matx22d A; 
		Matx21d B;
		A(0, 0) = N2[j].x; 
		A(0, 1) = N2[j].y;
		A(1, 0) = N2[k].x;
		A(1, 1) = N2[k].y;

		B(0, 0) = D2[j];
		B(1, 0) = D2[k];

		Mat solution; 
		cv::solve(A, B, solution); 
		Mat_<double> corner(solution); 

		corners[j].x = int((corner(0, 0) * invscale + 0.49999));
		corners[j].y = int((corner(1, 0) * invscale + 0.49999));
	}

	return true; 
}

double CalculateSkewness(Mat_<double>& vectorized_crop/*prebuilt intensities*/, ABoxedblob& aBlob, const unsigned short *data, const int imcols, Point2f offset, Point2d PB/*base point of rectangle*/, double a/*angle of rectangle*/) {
	double err_sum = 0;
	int idx = 0;
	for(auto& aRow : aBlob._rows) {
		const int xl = aRow._rowends[0]._column;
		const int xr = aRow._rowends[1]._column;
		const unsigned short *ptr = data + aRow._rownum * imcols + xl;
		for(int x = xl; x <= xr; ++x, ++ptr) {
			Point2d pixel(x - offset.x, aRow._rownum - offset.y);
			vectorized_crop(idx, 1) = double(*ptr) / g_bytedepth_scalefactor;

			double theta = Angle(pixel.x - PB.x, pixel.y - PB.y);
			if(theta > (a + CV_PI) || theta < a) { // a is always < PI, because the segment (PB, PA) is oriented up. 
				vectorized_crop(idx, 1) *= -1; // half plane to the right from (PB, PA)
			}
			err_sum += vectorized_crop(idx, 1);
			++idx;
		}
	}

	const int n = vectorized_crop.rows;
	double mean = err_sum / n;

	double err_sum_sqre = 0;
	double err_sum_cube = 0;
	//double err_sum_quad = 0;

	for(int k = 0; k < n; ++k) {
		double err = vectorized_crop(k, 1) - mean;
		err_sum_sqre += pow(err, 2);
		err_sum_cube += pow(err, 3);
		//err_sum_quad += pow(err, 4);
	}

	double sd = sqrt(err_sum_sqre / (n - 1));
	double skewness = std::abs(err_sum_cube / pow(sd, 3)) * (n) / ((n - 1) * (n - 2));
	//double kurtosis = (err_sum_quad / pow(sd, 4)) * (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) - 3;

	return skewness; 
}

int BlobsLoG(std::vector<ABoxedblob>& blobs, Mat& image, unsigned int& threshold_intensity, cv::Rect roi, Mat_<double>& kmat, unsigned short *intensity_avg_ptr, const double max_LoG_factor) {
	const int imcols = image.cols;
	const int imrows = image.rows;
	if((roi.x + roi.width) > imcols || (roi.y + roi.height) > imrows) {
		roi.width = roi.height = 0;
	}
	const int xs = roi.width > 0? roi.x: 0;
	const int ys = roi.height > 0? roi.y: 0;

	const int colMax = roi.width > 0? (xs + roi.width): imcols;
	const int rowMax = roi.height > 0? (ys + roi.height): imrows;

	ATLSMatrixvar<signed short> tracker;
	ATLSMatrixvar<double> tracker_value;

	tracker.SetDimensions(imrows, imcols);
	tracker_value.SetDimensions(imrows, imcols);

	const unsigned short *data = (unsigned short *)image.data;

	blobs.reserve(500);
	blobs.resize(0); 

	//Mat_<double> kmat = LoG(1.25, 7); 

	const int knrows2 = kmat.rows / 2;
	const int kncols2 = kmat.cols / 2;

	const int row_limits[2] = { ys + knrows2, rowMax - knrows2 };
	const int col_limits[2] = { xs + kncols2, colMax - kncols2 };

	const int dy = row_limits[1] - row_limits[0]; 
	const int dx = col_limits[1] - col_limits[0];

	const int blob_row_lim[2] = { dy <= 1 ? knrows2 : row_limits[0], dy <= 1 ? (imrows - knrows2) : row_limits[1] };
	const int blob_col_lim[2] = { dx <= 1 ? kncols2 : col_limits[0], dx <= 1 ? (imcols - kncols2) : col_limits[1] };

	unsigned long long intensity_avg = 0; 


	int64_t qcounter_marker = 0;
	int64_t qcounter_convolve = 0;

	qcounter_marker = MyQueryCounter() + qcounter_delta;


	for(int y1 = row_limits[0]; y1 < row_limits[1]; ++y1) {
		int x1 = col_limits[0];
		const signed short *tracker_data = &tracker(y1, x1);
		const unsigned short *intensity = data + y1 * imcols + x1;
		for(; x1 < col_limits[1]; ++x1, ++tracker_data, ++intensity) {
			intensity_avg += *intensity; 
			if(*tracker_data > 0) {
				continue;
			}
			if(*intensity >= threshold_intensity) {
				int rlims[2] = { std::max(y1 - g_max_boxsize_pixels, blob_row_lim[0]), std::min(y1 + g_max_boxsize_pixels, blob_row_lim[1]) };
				int clims[2] = { std::max(x1 - g_max_boxsize_pixels, blob_col_lim[0]), std::min(x1 + g_max_boxsize_pixels, blob_col_lim[1]) };

				qcounter_convolve += BlobLoG(blobs, Point2d(x1, y1), image, kmat, rlims, clims, tracker, tracker_value, max_LoG_factor);
			}
		}
	}

	qcounter_marker = MyQueryCounter() - qcounter_marker;


	intensity_avg /= (row_limits[1] - row_limits[0]) * (col_limits[1] - col_limits[0]);
	if(intensity_avg_ptr) {
		*intensity_avg_ptr = (ushort)intensity_avg; 
	}

	if(blobs.size() > 50) {
		threshold_intensity += g_bytedepth_scalefactor * (unsigned int)blobs.size() / 50;
	}
	if(blobs.size() < 3) {
		threshold_intensity -= g_bytedepth_scalefactor;
	}

	int cnt = 0; 

	for(auto& aBlob : blobs) {
		if(aBlob._associatedBlobs.size() > 0) {
			MergeBlobs(blobs, aBlob._id);
			++cnt; 
		}
	}

	return cnt; 
}

int BlobCentersLoG(std::vector<ABox>& boxes, std::vector<ClusteredPoint>& points, Mat& image, unsigned int& threshold_intensity, cv::Rect roi, Mat_<double>& kmat, bool arff_file_requested, ushort *intensity_avg_ptr, double max_LoG_factor) {
	const int imcols = image.cols;
	const int imrows = image.rows;

	const unsigned short *data = (unsigned short *)image.data;

	//std::cout << "BlobsLoG: threshold_intensity " << ' ' << threshold_intensity << std::endl;

	std::vector<ABoxedblob> blobs;
	BlobsLoG(blobs, image, threshold_intensity, roi, kmat, intensity_avg_ptr, max_LoG_factor);

	bool supervised_LoG = ((roi.height == 0) && (roi.width == 0)) || (roi.height == kmat.rows) || (roi.width == kmat.cols);

	boxes.resize(0);
	points.resize(0);
	boxes.reserve(10);
	points.reserve(10);

	int max_shapemeasure_rectangle_index = -1;
	double max_shapemeasure = std::numeric_limits<double>::min();

	std::vector<Point> contour_notsmoothed;
	std::vector<Point> contour;
	std::vector<Point> contourRight;
	std::vector<Point> contourSmoothed;
	std::vector<Point> hull;
	std::vector<Point> min_quad2i;
	std::vector<Point> contour100;

	std::vector<Point2f> corners;

	Mat maux; // temporary matrix to hold intermediate results. 

	if (blobs.size() > 10) {
		std::cout << "BlobCentersLoG: total blobs " << ' ' << blobs.size() << std::endl;
	}

	for(auto& aBlob : blobs) {
		if(aBlob._rows.size() > 1) {
			long long counter = 0;

			double intensity = 0; 

			int min_intensity = std::numeric_limits<int>::max();

			int x_min = std::numeric_limits<int>::max(); 
			int x_max = std::numeric_limits<int>::min();

			contour.clear(); 
			contourRight.clear(); 
			contourSmoothed.clear(); 

			contour.reserve(aBlob._rows.size() * 2 + 1);
			contourRight.reserve(aBlob._rows.size() + 1);

			for(auto& aRow : aBlob._rows) {
				const int xl = aRow._rowends[0]._column;
				const int xr = aRow._rowends[1]._column;
				if(xl < x_min) {
					x_min = xl; 
				}
				if(xr > x_max) {
					x_max = xr;
				}
				contour.push_back(Point(xl, aRow._rownum));
				contourRight.push_back(Point(xr, aRow._rownum));

				const unsigned short *ptr = data + aRow._rownum * imcols + xl;
				for(int x = xl; x <= xr; ++x, ++ptr) {
					const unsigned short ival = *ptr; 
					intensity += ival;
					if(ival < min_intensity) {
						min_intensity = ival;
					}
				}

				const int sub_count = (xr - xl + 1); 
				counter += sub_count;
			}

			if(min_intensity < 1) {
				min_intensity = 1; 
			}

			int y_min = aBlob._rows[0]._rownum;
			int y_max = aBlob._rows[aBlob._rows.size() - 1]._rownum;

			ABox aBox; 
			aBox.x[0] = x_min;
			aBox.x[1] = x_max + 1;
			aBox.y[0] = aBlob._rows[0]._rownum;
			aBox.y[1] = aBlob._rows[aBlob._rows.size() - 1]._rownum + 1;

			if((x_max - x_min) >= 3 && (y_max - y_min) >= 3) {
				int dx = x_max - x_min;
				int dy = y_max - y_min;

				if(dx <= g_max_boxsize_pixels && dy <= g_max_boxsize_pixels) {
					if (supervised_LoG || (double)std::min(dx, dy) / (double)std::max(dx, dy) > 3.0 / 5.0) {

						std::reverse(contourRight.begin(), contourRight.end());
						std::vector<Point>::iterator contourRight_begin = contourRight.begin();
						if(contour[contour.size() - 1] == contourRight[0]) {
							++contourRight_begin;
						}
						contour.insert(contour.end(), contourRight_begin, contourRight.end());
						if(contour[0] != contour[contour.size() - 1]) {
							contour.push_back(contour[0]);
						}

						contour_notsmoothed = contour; 

						bool isValid = true;

						double contour_area = contourArea(contour, false/*cannot be negative*/);
						double contour_perimeter = arcLength(contour, false/*tell it to treat as not closed because it is closed*/);
						double contour_circularity = 4 * CV_PI * std::abs(contour_area) / std::pow(contour_perimeter, 2);

						hull.clear();
						hull.reserve(contour.size());
						cv::convexHull(contour, hull);
						double hull_area = contourArea(hull, true/*can be negative*/);
						double hull_perimeter = arcLength(hull, true/*tell it to close it because it is not closed*/);
						double hull_circularity = 4 * CV_PI * std::abs(hull_area) / std::pow(hull_perimeter, 2);

						double contour_area2hull_area = std::abs(contour_area) / std::abs(hull_area);

						double flattening = 0; // defined further down

						double corners_area = 0; // defined further down


						contourSmoothed = contour;

						if (supervised_LoG) {
							projectContour(contourSmoothed);
							contour = contourSmoothed; 
						}
						else {

							isValid = contour_circularity < 1.2;

							if (contour_area2hull_area < 0.7) {
								isValid = false;
							}


							min_quad2i.clear();
							contour100.clear();


							double flattening = approximate_withQuad100(contour/*contourSmoothed*/, min_quad2i, contour100);

							if (flattening > 0.33 && std::abs(contour_area) < 70) {
								isValid = false;
							}

							if (!isValid) {
								continue;
							}


							fitRectangleToPoints(contour100, min_quad2i);


							corners.clear();
							corners.reserve(4);

							for (auto& point : min_quad2i) {
								corners.push_back(Point2f(point) * 0.01f);
							}

							corners_area = contourArea(corners);
						}




						int point_index = (int)points.size();
						points.resize(point_index + 1);
						ClusteredPoint& point = points[point_index];



						intensity /= counter;
						aBox.intensity = (int)(intensity + 0.5);




						Rect approxBoundingRect(aBox.x[0], aBox.y[0], aBox.x[1] - aBox.x[0], aBox.y[1] - aBox.y[0]);


						approxBoundingRect += Size(14, 14);
						approxBoundingRect -= Point(7, 7);
						if (approxBoundingRect.x < 0) {
							approxBoundingRect.x = 0;
						}
						if (approxBoundingRect.y < 0) {
							approxBoundingRect.y = 0;
						}
						if (image.rows < approxBoundingRect.height) {
							approxBoundingRect.height = image.rows;
						}
						if (image.cols < approxBoundingRect.width) {
							approxBoundingRect.width = image.cols;
						}
						if (approxBoundingRect.x > (image.cols - approxBoundingRect.width)) {
							approxBoundingRect.x = image.cols - approxBoundingRect.width;
						}
						if (approxBoundingRect.y > (image.rows - approxBoundingRect.height)) {
							approxBoundingRect.y = image.rows - approxBoundingRect.height;
						}

						Point2f offset((float)approxBoundingRect.x, (float)approxBoundingRect.y);

						Mat crop(image, approxBoundingRect);
						Mat crop_colored;
						Mat crop_coloredOriginal; 


						maux = Mat(); // clear temporary matrix

						double fx = 240.0 / crop.rows;

						if (g_configuration._visual_diagnostics) {
							crop.convertTo(maux, CV_16UC1, std::max(256 / (int)g_bytedepth_scalefactor, 1));
							cv::cvtColor(maux, crop_colored, ColorConversionCodes::COLOR_GRAY2RGB);
							cv::resize(crop_colored, crop_coloredOriginal, cv::Size(0, 0), fx, fx, INTER_AREA);
							crop_coloredOriginal.copyTo(crop_colored);
						}




						if (contour.size() > 0 && crop_colored.rows > 0) {
							Point a = (Point2f(contour[0]) + Point2f(0.5, 0.5) - (offset)) * fx; 
							for (int j = 1; j < contour.size(); ++j) {
								Point b = (Point2f(contour[j]) + Point2f(0.5, 0.5) - (offset)) * fx;
								cv::line(crop_colored, a, b, Scalar(0, (size_t)255 * 256, 0));
								a = b; 
							}
						}





						double minMax[2];
						mat_minMax(crop, minMax); // it assumes that global thresholding is done before calling BlobCentersLoG().

						Mat_<double> vectorized_crop((int)counter, 2);
						int idx = 0;

						int intensity_variation_counter = 0;
						const int intensity_variation_threshold = (int)(minMax[1] + aBox.intensity) / 2;

						Point2d PA;
						Point2d PB;
						double a = DirectionOfRectangle(corners, PA, PB);
						PB -= Point2d(offset); 
						PA += PB; 

						double max_distance = 0; 

						for(auto& aRow : aBlob._rows) {
							const int xl = aRow._rowends[0]._column;
							const int xr = aRow._rowends[1]._column;
							const unsigned short *ptr = data + aRow._rownum * imcols + xl;
							for(int x = xl; x <= xr; ++x, ++ptr) {
								if(*ptr > intensity_variation_threshold) {
									++intensity_variation_counter;
								}

								Point2d pixel((size_t)x - offset.x, (size_t)aRow._rownum - offset.y);
								double dist = std::sqrt(DistanceSquareFromPointToSegment(pixel, PA, PB));
								if(max_distance < dist) {
									max_distance = dist;
								}

								vectorized_crop(idx, 0) = dist;
								vectorized_crop(idx, 1) = double(*ptr - minMax[0]);
								++idx;
							}
						}

						double intensity_upperquantile = double(intensity_variation_counter) / double(counter);
						double form_factor = hull_circularity * (1 - flattening) * intensity_upperquantile; /*hull_circularity < 0.96 && flattening > 0.1 && intensity_upperquantile < 0.4*/



						double intensity_scale = max_distance / (minMax[1] - minMax[0]);
						for(int j = 0; j < vectorized_crop.rows; ++j) {
							vectorized_crop(j, 1) *= intensity_scale;
						}

						Mat covarMat;
						calcCovarMatrix(vectorized_crop, covarMat, maux = Mat(), CovarFlags::COVAR_NORMAL | CovarFlags::COVAR_ROWS);

						double covar = -mat_get<double>(covarMat, 0, 1);
						if (covar < 0) {
							covar = -covar;
						}

						covar /= std::pow(max_distance, 2);

						double effective_flattening = (std::pow(covar, 2) / (form_factor));
						effective_flattening *= contour_area2hull_area;

						//double skewness = CalculateSkewness(vectorized_crop, aBlob, data, imcols, offset, PB, a);
						double skewness = 1;
						double shapemeasure = effective_flattening / (skewness * (intensity / g_bytedepth_scalefactor));





						Point2d cornersCenter;
						bool corners_areValid = false;
						if((std::abs(contour_area) * 0.8) < corners_area) {
							corners_areValid = QuadCenter(corners, cornersCenter);
						}

						double intensity_atcenter = (mat_get<double>(image, (int)(cornersCenter.y), (int)(cornersCenter.x)) + mat_get<double>(image, (int)ceil(cornersCenter.y), (int)ceil(cornersCenter.x))) / 2.0;
						if(covar < 0.95) {
							isValid = false;
						}
						else
						if(form_factor < 0.095) {
							isValid = false;
						}
						else
						if(intensity_atcenter < (0.8 * intensity)) {
							isValid = false;
						}
						else
						if((contour_area * intensity_upperquantile) < 10) {
							isValid = false;
						}
						else
						if(contour_area2hull_area < 0.9) {
							isValid = false;
						}
						else
						if((covar * intensity_upperquantile) < 0.65) {
							isValid = false;
						}

						//if(isValid) {
						//	// the following test capitalizes on existence of multiple contours in a complex structure. 
						//	// it builds all possible contours in the image. A valid blob should have just one. 
						//	// the ground for this is that the valid blobs represent plain object like cylinders, semi-spheres, and circles. 
						//	// a more complex structure is an indication that an object is a random reflection. 
						//	// a reflection as a more complex structure will produce a multitude of contours. 
						//	// complication: the correct outcome depends on thresholding. 

						//	maux = Mat(); 
						//	crop.convertTo(maux, CV_32FC1);
						//	crop = maux; 

						//	double minMax[2];
						//	mat_minMax(crop, minMax); // it assumes that global thresholding is done before calling BlobCentersLoG().
						//	const double thresholdVal = /*(minMax[0] + minMax[1]) / 2*/std::max((minMax[0] + minMax[1]) / 2, intensity);

						//	maux = Mat();
						//	threshold(crop, maux, thresholdVal, (size_t)255 * g_bytedepth_scalefactor, THRESH_TOZERO);
						//	Mat binarizedImage = mat_binarize2byte(maux);
						//	std::vector<std::vector<cv::Point> > contours;
						//	findContours(binarizedImage, contours, RetrievalModes::RETR_LIST, ContourApproximationModes::CHAIN_APPROX_SIMPLE);
						//	if(contours.size() > 1) {
						//		isValid = false;
						//	}
						//}

						bool rectangleDetected = false;
						if(isValid && corners_areValid && (covar > 6 || shapemeasure > 0.36) && (covar * shapemeasure) > (6 * /*0.36*/0.27) && (std::abs(contour_area) * intensity_upperquantile) > 12) { // std::abs(contour_area) > 70 && intensity_upperquantile > 0.18
							if(contour_area2hull_area > 0.9) {
								if((contour_area * 0.8) < corners_area) {
									if(isContourConvex((corners))) {
										const double effective_flattening_min = (60.0 * (255.0 * g_bytedepth_scalefactor) / (intensity)); 
										if(effective_flattening > /*80.3403*/effective_flattening_min && flattening > 0.107626) { // JRip rules. 
											rectangleDetected = true;
										}
										else
										if(effective_flattening > /*80*/effective_flattening_min && (hull_circularity <= 0.95352 || intensity_upperquantile > 0.646154)) { // J48 conditions. 
											rectangleDetected = true;
										}
									}
								}
							}
						}
						//bool rectangleDetected = false;
						//if(covar > 5 && /*flattening > 0.09 && */shapemeasure > 0.35/*0.4*/ && (std::abs(contour_area) * intensity_upperquantile) > 12) { // std::abs(contour_area) > 70 && intensity_upperquantile > 0.18
						//	if((covar > 5.5 && flattening > 0.2) || effective_flattening > 92) { // JRip rules. 
						//		if(contour_area2hull_area > 0.9) {
						//			if((contour_area * 0.8) < corners_area) {
						//				if(isContourConvex((corners))) {
						//					rectangleDetected = true;
						//				}
						//			}
						//		}
						//	}
						//}

						if(crop_colored.rows > 0) {
							if(corners_areValid) {
								cv::circle(crop_colored, (Point2f(cornersCenter) - Point2f(offset)) * fx, 2, Scalar(255 * 256, 0, 255 * 256), -1);
							}

							if (corners.size()) {
								for (int j = 0; j < 4; ++j) {
									cv::line(crop_colored, (corners[j] - (offset)) * fx, (corners[(j + 1) % 4] - (offset)) * fx, Scalar(0, 0, 255 * 256));
								}
							}
						}





						//if ((isValid || supervised_LoG) || corners_areValid) {
						//	if(g_configuration._use_ellipse_fit) {
						//		isValid = RowsCenterEllipse(aBlob._rows, point, image, min_intensity - 1, aBox.intensity, roi) && isValid;
						//	}
						//	else {
						//		isValid = RowsCenter(aBlob._rows, point, image, min_intensity - 1, aBox.intensity, aBox, roi) && isValid;
						//	}
						//}

						if ((isValid || supervised_LoG) && corners_areValid) {
							point._corners.reserve(corners.size());
							for(auto corner : corners) {
								corner.y = float(imrows) - corner.y;
								point._corners.push_back(Point2d(corner));
							}
						}

						double centers_distance = 0;
						if(corners_areValid) {
							Point2d point_dist = point - Point2d(cornersCenter.x, float(imrows) - cornersCenter.y);
							centers_distance = std::sqrt(point_dist.ddot(point_dist));
							if(centers_distance > 1.7/*2.5*/) {
								isValid = false; 
							}
						}

						if ((isValid || supervised_LoG) && rectangleDetected) {
							point._isACenter = 1;
						}

						if (isValid || supervised_LoG) {
							if(crop_colored.rows > 0) {
								cv::circle(crop_colored, point._crop_center, 2, Scalar(0, (size_t)255 * 256, 0), -1);
								crop_colored.copyTo(point._crop);
								crop_coloredOriginal.copyTo(point._cropOriginal); 
							}

							point._crop_mat_scalefactor = fx;
							point._crop_mat_offset = offset;
							point._crop_center = (Point2f((float)point.x * 2, (float)imrows) - Point2f(point) - Point2f(offset)) * fx;
						}

						point._intensity_atcenter = intensity_atcenter;
						point._shapemeasure = shapemeasure;
						point._effective_flattening = effective_flattening;
						point._flattening = flattening;
						point._skewness = skewness;
						point._covar = covar;
						point._hull_circularity = hull_circularity;
						point._intensity_upperquantile = intensity_upperquantile;
						point._contour_area = contour_area;
						point._contour_area2hull_area = contour_area2hull_area;
						point._centers_distance = centers_distance;




						if ((isValid || supervised_LoG) && rectangleDetected) {
							if(max_shapemeasure < shapemeasure) {
								max_shapemeasure = shapemeasure;
								max_shapemeasure_rectangle_index = point_index;
							}
						}

						if(arff_file_requested && corners_areValid && !isValid) {
							point.ARFF_Output(false, false);
						}

						if ((isValid || supervised_LoG)) {
							aBox.contour.swap(contour); 
							aBox.contour_notsmoothed.swap(contour_notsmoothed);
							boxes.push_back(aBox);
						}
						else {
							points.resize(point_index);
						}
					}
				}
			}
		}
	}

	if (blobs.size() > 10) {
		std::cout << "BlobCentersLoG: finished " << std::endl;
	}

	return max_shapemeasure_rectangle_index;
}


void Rotate_corners(const double a, const Point2d& center, std::vector<Point2d>& corners) {
	double sina = std::sin(a);
	double cosa = std::cos(a);
	Matx33d R(
		cosa, -sina, 0,
		sina, cosa, 0,
		0, 0, 1
		);

	Matx33d T1(
		1, 0, -center.x,
		0, 1, -center.y,
		0, 0, 1
		);
	Matx33d T2(
		1, 0, center.x,
		0, 1, center.y,
		0, 0, 1
		);

	Matx33d transform = T2*R*T1;

	for(auto& corner : corners) {
		Matx31d P = transform * Matx31d(corner.x, corner.y, 1);
		corner = Point2d(P(0), P(1));
	}
}

void Rotate_corners(const double a[2], const ClusteredPoint* ppoints[2], std::vector<Point2d> corners[2]) {
	for(int j = 0; j < 2; ++j) {
		Rotate_corners(a[j], *ppoints[j], corners[j]);
	}
}

void Align_Rectangles(const ClusteredPoint& point0, const ClusteredPoint& point1, std::vector<Point2d>& corners0, std::vector<Point2d>& corners1) {
	const ClusteredPoint *ppoints[2] = {&point0, &point1};

	double a[2];
	a[0] = InclinationAngleOfRectangle(point0._corners) * 180 / CV_PI;
	a[1] = InclinationAngleOfRectangle(point1._corners) * 180 / CV_PI;

	double angle_diff = (a[0] - a[1]);

	double wa[2] = {std::abs(a[0]), std::abs(a[1])};
	double wa_sum = wa[0] + wa[1];
	if(wa_sum < 5) {
		return;
	}

	wa[0] = 1 - (wa[0] / wa_sum);
	wa[1] = 1 - (wa[1] / wa_sum);

	//wa[0] = 0.5;
	//wa[1] = 0.5; 

	double w[2] = {std::pow(point0._shapemeasure, 2), std::pow(point1._shapemeasure, 2)};
	double shapemeasure_sum = w[0] + w[1];
	w[0] = 1 - (w[0] / shapemeasure_sum);
	w[1] = 1 - (w[1] / shapemeasure_sum);


	w[0] = (w[0] + wa[0]) / 2;
	w[1] = (w[1] + wa[1]) / 2;


	a[0] = 0.3 * -w[0] * angle_diff * CV_PI / 180;
	a[1] = 0.3 * +w[1] * angle_diff * CV_PI / 180;

	std::vector<Point2d> rotated_corners[2] = {point0._corners, point1._corners};
	Rotate_corners(a, ppoints, rotated_corners);
	corners0.swap(rotated_corners[0]);
	corners1.swap(rotated_corners[1]);
}



double Rectangles_Minimize_y_error(const ClusteredPoint& point0, const ClusteredPoint& point1, double& max_y_error_improvement, std::vector<Point2d>& corners0, std::vector<Point2d>& corners1) {
	// point0 and point1 are the gravity centers of rectangles; _corners are the rectangles. 
	max_y_error_improvement = 0; 

	const ClusteredPoint *ppoints[2] = {&point0, &point1};

	double a[2];
	a[0] = InclinationAngleOfRectangle(point0._corners) * 180 / CV_PI;
	a[1] = InclinationAngleOfRectangle(point1._corners) * 180 / CV_PI;

	double w[2] = {std::pow(point0._shapemeasure, 2), std::pow(point1._shapemeasure, 2)};
	double shapemeasure_sum = w[0] + w[1];
	w[0] = 1 - (w[0] / shapemeasure_sum);
	w[1] = 1 - (w[1] / shapemeasure_sum);

	//double wa[2] = {std::abs(a[0]), std::abs(a[1])};
	//double wa_sum = wa[0] + wa[1];
	//if(wa_sum < 5) {
	//	return 0;
	//}

	//wa[0] = 1 - wa[0] / wa_sum;
	//wa[1] = 1 - wa[1] / wa_sum;

	//w[0] = (w[0] + wa[0]) / 2; 
	//w[1] = (w[1] + wa[1]) / 2;

	double max_Y_error = 0;
	double avg_Y_error = Average_y_error_betweenRectangles(point0._corners, point1._corners, max_Y_error);

	double r[2];
	r[0] = RadiusOfRectangle(point0._corners);
	r[1] = RadiusOfRectangle(point1._corners);

	double angle = avg_Y_error / ((r[0] + r[1]) / 2);

	std::vector<Point2d> corners[2][2];

	double max_err_improvement[2] = {0, 0};
	double improvement_avg_err[2] = {0, 0};

	for(int k = 0; k < 2; ++k) {
		int direction = k == 0? -1: 1; 

		a[0] = direction * +w[0] * angle;
		a[1] = direction * -w[1] * angle;

		corners[k][0] = point0._corners;
		corners[k][1] = point1._corners;

		Rotate_corners(a, ppoints, corners[k]);

		double after_max_Y_error = 0;
		double after_avg_Y_error = Average_y_error_betweenRectangles(corners[k][0], corners[k][1], after_max_Y_error);

		max_err_improvement[k] = std::abs(max_Y_error) - std::abs(after_max_Y_error);
		improvement_avg_err[k] = std::abs(avg_Y_error) - std::abs(after_avg_Y_error);
	}

	int istep = ((max_err_improvement[0] + improvement_avg_err[0]) > (max_err_improvement[1] + improvement_avg_err[1]))? 0: 1;
	istep = 0; 

	corners0.swap(corners[istep][0]);
	corners1.swap(corners[istep][1]);


	max_y_error_improvement = max_err_improvement[istep];

	return improvement_avg_err[istep];
}

double CompensateForLeftRightOcclusions(ClusteredPoint& point0/*order may be changed*/, ClusteredPoint& point1/*order may be changed*/, double& max_y_error_improvement, std::vector<Point2d>& corners0, std::vector<Point2d>& corners1) {
	ClusteredPoint *pcenter[2] = {&point0, &point1};

	std::vector<Point2d> matched_corners[2] = {pcenter[0]->_corners, pcenter[1]->_corners};

	Sort_by_epipolar_equivalence(&matched_corners[0][0], &matched_corners[1][0]); 
	for(int p = 0; p < 2; ++p) {
		while(pcenter[p]->_corners[0] != matched_corners[p][0]) {
			std::rotate(pcenter[p]->_corners.begin(), pcenter[p]->_corners.begin() + 1, pcenter[p]->_corners.end());
		}
	} 


	Point2d obscuredPoints[8]; // 2 from left and 2 from right; need 4, but use 8 to avoid buffer overflow. 
	Point2d visiblePoints[8]; // 2 from left and 2 from right; need 4, but use 8 to avoid buffer overflow. 
	double a[2];
	Point2d PA[2];
	Point2d PB[2];
	// 1. split corners to left and right groups in respect to the center line of the direction of rectangle. 
	// left corners in left image, and right corners in right image will be moved proportionally to the error. 
	// p == 0 means the left image, otherwise right. 
	for(int p = 0; p < 2; ++p) {
		a[p] = DirectionOfRectangle(pcenter[p]->_corners, PA[p], PB[p], true/*points_are_presorted*/);
		int j = 0; 
		int i = 0; 
		std::vector<Point2d> rotated_corners = matched_corners[p];
		Rotate_corners(-(a[p] - CV_PI / 2), *pcenter[p], rotated_corners);
		for(int k = 0; k < rotated_corners.size(); ++k) {
			if(((rotated_corners[k].x - (*pcenter[p]).x) * (p == 0? -1: 1)) > 0) {
				obscuredPoints[j++ + (p * 2)] = matched_corners[p][k];
			}
			else {
				visiblePoints[i++ + (p * 2)] = matched_corners[p][k];
			}
		}
		if(j != 2 || i != 2) {
			return -1; // something is wrong. 
		}
	}

	// 2. calculate error. 
	double error[2]; 
	error[0] = std::abs(std::abs(obscuredPoints[0].y - visiblePoints[2].y) + std::abs(obscuredPoints[1].y - visiblePoints[3].y)) / 2;
	error[1] = std::abs(std::abs(obscuredPoints[2].y - visiblePoints[0].y) + std::abs(obscuredPoints[3].y - visiblePoints[1].y)) / 2;

	int error_sign[2] = {
		obscuredPoints[0].y + obscuredPoints[1].y - visiblePoints[2].y - visiblePoints[3].y < 0? -1: 1, 
		obscuredPoints[2].y + obscuredPoints[3].y - visiblePoints[0].y - visiblePoints[1].y < 0? -1: 1
	};


	double w[2] = {std::sqrt(point0._shapemeasure), std::sqrt(point1._shapemeasure)};
	{
		double flat_sum = w[0] + w[1]; 
		for(auto& W : w) {
			W = 1 - (W / flat_sum);
		}
		int  j = 0; 
		for(auto& E : error) {
			E *= 2 * w[j] * std::sqrt(std::abs(sin(a[j]))); 
			++j; 
		}
	}


	// 3. build trnaslation vector of length equal to the error. 
	Point2d translation[4]; 
	for(int p = 0, k = 0; p < 2; ++p, k += 2) {
		translation[p] = (*pcenter[p]) + (PA[p] * (0.375 * error_sign[p]));
		if(error_sign[p] > 0) {
			translation[p] -= obscuredPoints[1 + k];
		}
		else {
			translation[p] -= obscuredPoints[0 + k];
		}
		translation[p] *= error[p] / std::sqrt(translation[p].ddot(translation[p]));
		//translation[p] *= (2.0 / 3.0) * error[p] / std::sqrt(translation[p].ddot(translation[p]));
	}
	//for(int p = 0, k = 1; p < 2; ++p, --k) {
	//	translation[p + 2] = PA[p] * (0.375 * error_sign[p]);
	//	translation[p + 2] *= (1.0 / 3.0) * error[k] / std::sqrt(translation[p + 2].ddot(translation[p + 2]));
	//}

	// 4. translate sides
	for(int p = 0, k = 0; p < 2; ++p, k += 2) {
		for(int j = 0; j < 2; ++j) {
			obscuredPoints[j + k] += translation[p]; 
		}
	}
	//for(int p = 2, k = 0; p < 4; ++p, k += 2) {
	//	for(int j = 0; j < 2; ++j) {
	//		visiblePoints[j + k] += translation[p];
	//	}
	//}

	double max_Y_error = 0;
	double avg_Y_error = Average_y_error_betweenRectangles(point0._corners, point1._corners, max_Y_error);

	std::vector<Point2d> *corners[2] = {&corners0, &corners1};
	for(int p = 0, k = 0; p < 2; ++p, k += 2) {
		(*corners[p]).resize(4); 
		(*corners[p])[0] = obscuredPoints[0 + k]; 
		(*corners[p])[1] = obscuredPoints[1 + k];
		(*corners[p])[2] = visiblePoints[1 + k];
		(*corners[p])[3] = visiblePoints[0 + k];
	}

	double after_max_Y_error = 0;
	double after_avg_Y_error = Average_y_error_betweenRectangles(corners0, corners1, after_max_Y_error);

	max_y_error_improvement = std::abs(max_Y_error) - std::abs(after_max_Y_error);
	return std::abs(avg_Y_error) - std::abs(after_avg_Y_error);
}

void OptimizeSelectionOfQuadrilaterals(int idx[2], std::vector<ClusteredPoint> cv_points[2], const int (&imrows)[2]/*only for drawing circles in crop images*/) {
	if(idx[0] >= 0 || idx[1] >= 0) {
		int V = idx[0] >= 0? 0: 1;
		int Q = V == 0? 1: 0;
		if(idx[Q] < 0) { // if one image has been identified with a rectangle, but other has been not, then try to locate a near fit in the other image. 
			auto& point = cv_points[V][idx[V]];
			double min_Y_error = g_max_Y_error * 2 + 1;
			for(auto& other : cv_points[Q]) {
				if(other._corners.size() == 4 /*&& other._covar > 3 && other._flattening > 0.09 */&& other._hull_circularity < 0.96) {
					double Y_error = std::abs(other.y - point.y);
					other._centers_min_Y_error = Y_error;
					if(Y_error < min_Y_error) {
						min_Y_error = Y_error;
					}
				}
			}
			for(int j = 0; j < cv_points[Q].size(); ++j) {
				auto& other = cv_points[Q][j];
				if(other._centers_min_Y_error == min_Y_error) {
					other._isACenter = 1;
					idx[Q] = j;
				}
				other._centers_min_Y_error = std::numeric_limits<int>::max();
			}
		}
	}

	if(idx[0] >= 0 && idx[1] >= 0) {

		// correction of selected rectangles. 
		// select left and right rectangles that have minimal Y_error between them. 

		double min_Y_error = std::numeric_limits<int>::max();
		for(auto& point : cv_points[0]) {
			if(point._isACenter) {
				for(auto& other : cv_points[1]) {
					if(other._isACenter) {
						double Y_error = std::abs(other.y - point.y);
						if(Y_error < point._centers_min_Y_error) {
							point._centers_min_Y_error = Y_error;
						}
						if(Y_error < other._centers_min_Y_error) {
							other._centers_min_Y_error = Y_error;
						}
						if(Y_error < min_Y_error) {
							min_Y_error = Y_error;
						}
					}
				}
			}
		}
		for(int k = 0; k < 2; ++k) {
			auto& points = cv_points[k];
			for(int j = 0; j < points.size(); ++j) {
				auto& point = points[j];
				if(point._isACenter) {
					if(point._centers_min_Y_error > min_Y_error) {
						point._isACenter = 0;
					}
					else {
						idx[k] = j;
					}
				}
			}
		}



		std::vector<Point2d> corners[2];

		double max_y_error_improvement = 0;
		double improvement_avg_y_error = 0;

		//Align_Rectangles(/*in*/cv_points[0][idx[0]], /*in*/cv_points[1][idx[1]], /*out*/corners[0], /*out*/corners[1]);
		//for(int j = 0; j < 2; ++j) {
		//	cv_points[j][idx[j]]._corners.swap(corners[j]);
		//}

		//improvement_avg_y_error = Rectangles_Minimize_y_error(/*in*/cv_points[0][idx[0]], /*in*/cv_points[1][idx[1]], /*out*/max_y_error_improvement, /*out*/corners[0], /*out*/corners[1]);
		//improvement_avg_y_error = max_y_error_improvement = 0;
		//if((improvement_avg_y_error + max_y_error_improvement) > -0.1) {
		//	for(int j = 0; j < 2; ++j) {
		//		cv_points[j][idx[j]]._corners.swap(corners[j]);
		//	}
		//}

		improvement_avg_y_error = CompensateForLeftRightOcclusions(cv_points[0][idx[0]], cv_points[1][idx[1]], /*out*/max_y_error_improvement, /*out*/corners[0], /*out*/corners[1]);
		if((improvement_avg_y_error + max_y_error_improvement) > -g_max_Y_error) {
			for(int j = 0; j < 2; ++j) {
				ClusteredPoint& center = cv_points[j][idx[j]];
				center._corners.swap(corners[j]);
				Point2d cornersCenter;
				if(QuadCenter(center._corners, cornersCenter)) {
					center.x = cornersCenter.x;
					center.y = cornersCenter.y;
					if(center._crop.rows > 0 && center._crop.cols > 0) {
						cv::circle(center._crop, (Point2f((float)cornersCenter.x * 2, (float)imrows[j]) - Point2f(cornersCenter) - Point2f(center._crop_mat_offset)) * center._crop_mat_scalefactor, 2, Scalar(100 * 256, 0, 0), -1);
					}
				}
			}
		}

		double max_Y_error = 0;
		double avg_Y_error = Average_y_error_betweenRectangles(cv_points[0][idx[0]]._corners, cv_points[1][idx[1]]._corners, max_Y_error);
		for(int j = 0; j < 2; ++j) {
			cv_points[j][idx[j]]._corners_max_Y_error = std::abs(max_Y_error) * 1.001;
		}

		for(int j = 0; j < 2; ++j) {
			ClusteredPoint& center = cv_points[j][idx[j]];
			for(int p = 0; p < 4; ++p) {
				Point2d corner0 = center._corners[p];
				Point2d corner1 = center._corners[(p + 1) % 4];
				corner0.y = imrows[j] - corner0.y;
				corner1.y = imrows[j] - corner1.y;
				if(center._crop.rows > 0 && center._crop.cols > 0) {
					cv::line(center._crop, (corner0 - Point2d((center._crop_mat_offset))) * center._crop_mat_scalefactor, (corner1 - Point2d((center._crop_mat_offset))) * center._crop_mat_scalefactor, Scalar(255 * 256, 255 * 256, 0), 1);
				}
			}
		}
	}
	else
	if(idx[0] >= 0 || idx[1] >= 0) {
		for(int j = 0; j < 2; ++j) {
			for(auto& point : cv_points[j]) {
				if(point._isACenter) {
					point._isACenter = 0;
				}
			}
		}
	}
}


bool GetFramesFromFilesWithSubdirectorySearch(Mat cv_image[2]/*out*/, std::vector<cv::Point2d> points[2], bool& arff_file_requested/*out*/, std::string& file_name, std::string& path_name) {
	bool image_isok = false; 

	static int s_arff_file_requested = 0;
	static int s_file_number = 0;
	static int s_iteration_count = 0;
	static int s_iteration_count_threshold = 20;
	static std::vector<std::string> s_sub_dirs;
	static std::string s_sub_dir;
	static int s_sub_dir_search = 0; 

	if(s_file_number > -1) {
		file_name = "raw-" + (s_file_number < 0 ? std::string() : std::to_string(++s_file_number));
		path_name = s_sub_dir + file_name;
		image_isok = GetImagesFromFile(cv_image[0], cv_image[1], points[0], points[1], path_name);
		if (!image_isok) {
			std::ostringstream ostr;
			if (s_file_number >= 0) {
				using namespace std;
				ostr << setfill('0') << setw(3) << s_file_number;
			}
			path_name = s_sub_dir + (file_name = "lake_" + ostr.str());
			image_isok = GetImagesFromFile(cv_image[0], cv_image[1], points[0], points[1], path_name);
		}
		std::cout << path_name << ' ' << (image_isok? "Ok": "NOk") << std::endl;
	}
	else {
		image_isok = false;
	}
	if(!image_isok) {
		s_file_number = std::abs(s_file_number) == 1 ? -1: 0;
	}
	if(s_file_number == 1) {
		++s_iteration_count;
	}
	if(s_file_number == 0 && s_sub_dir_search) {
		if(s_iteration_count == s_iteration_count_threshold) { // automatic output of records for data analysis. 
			s_arff_file_requested = 1;
		}
		else
		if(s_iteration_count > s_iteration_count_threshold) {
			s_arff_file_requested = 0;
			s_iteration_count = 0;
			s_iteration_count_threshold = 3;
			if(s_sub_dirs.size() == 0) {
				g_bTerminated = true;
				g_bUserTerminated = true; 
				return false; 
			}
			s_file_number = -1;
		}
	}
	if(s_file_number == -1) {
		if(s_sub_dirs.size() == 0) {
			Find_SubDirectories(g_path_nwpu_images_dir, s_sub_dirs);
		}
		if(s_sub_dirs.size()) {
			size_t idx = s_sub_dirs.size() - 1;
			s_sub_dir = s_sub_dirs[idx] + '\\';
			s_sub_dirs.resize(idx);
			s_file_number = 0;
			s_sub_dir_search = 1; 
		}
	}
	arff_file_requested = s_arff_file_requested != 0 && g_configuration._stereoimage_capture_requested;

	return image_isok; 
}


double g_otsu_threshold = -1;
double g_otsu_thresholdEx[2] = { -1, -1 };

void EvaluateOtsuThresholdEx(Mat cv_edges[6]) {
	double fy = 480.0 / cv_edges[0].rows;
	cv::resize(cv_edges[0], cv_edges[2] = Mat(), cv::Size(0, 0), fy, fy, INTER_AREA);
	cv::resize(cv_edges[1], cv_edges[3] = Mat(), cv::Size(0, 0), fy, fy, INTER_AREA);
	cv::normalize(cv_edges[2], cv_edges[4] = Mat(), 0, 255, NORM_MINMAX, CV_8UC1);
	cv::normalize(cv_edges[3], cv_edges[5] = Mat(), 0, 255, NORM_MINMAX, CV_8UC1);

	double tr[2] = { 91, 91 };
	tr[0] = threshold(cv_edges[4], cv_edges[2] = Mat(), 121, 255, THRESH_OTSU);
	tr[1] = threshold(cv_edges[5], cv_edges[3] = Mat(), 121, 255, THRESH_OTSU);

	g_otsu_thresholdEx[0] = tr[0];
	g_otsu_thresholdEx[1] = tr[1]; 

	int pixel_threshold = std::min((int)((std::max(tr[0], tr[1]))), 251);

	if (g_otsu_threshold < 0) {
		g_otsu_threshold = pixel_threshold;
	}
	else {
		g_otsu_threshold = (g_otsu_threshold * 5.0 + g_otsu_threshold) / 6.0;
	}
}

return_t __stdcall EvaluateOtsuThreshold(LPVOID lp) {
	SPointsReconstructionCtl *ctl = (SPointsReconstructionCtl*)lp;

	ctl->_gate.lock();
	ctl->_status--;
	ctl->_gate.unlock();

	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);

	Mat cv_edges[6];

	// find optimal threshold using Otsu algorithm. 

	while(!g_bTerminated && !ctl->_terminated) {
		bool image_isok = false;

		ctl->_gate.lock();
		image_isok = ctl->_cv_edges[0].rows > 0 && ctl->_cv_edges[1].rows > 0;
		if(image_isok) {
			if(ctl->_cv_edges[0].type() == CV_16UC1) {
				matCV_16UC1_memcpy(cv_edges[0], ctl->_cv_edges[0]);
				matCV_16UC1_memcpy(cv_edges[1], ctl->_cv_edges[1]);
			}
			else {
				image_isok = false;
			}
		}
		ctl->_gate.unlock();

		if(image_isok) {
			EvaluateOtsuThresholdEx(cv_edges);
		}
		OSSleep(100); 
	}

	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);

	ctl->_gate.lock();
	ctl->_status--;
	ctl->_gate.unlock();

	return 0;
}


void IntensifyImage(Mat& cv_image) {
	normalize(cv_image.clone(), cv_image, 0, (size_t)256 * g_bytedepth_scalefactor, NORM_MINMAX, CV_16UC1, Mat());

	Mat aux = cv_image.clone();
	//GaussianBlur(cv_image, aux, Size(5, 5), 0.9, 0.9);
	//AnisotropicDiffusion(aux, 21);
	medianBlur(aux, cv_image, 3);
	//GaussianBlur(cv_image, aux, Size(5, 5), 0.9, 0.9);
	//AnisotropicDiffusion(aux, 14);
	AnisotropicDiffusion(aux, 10);

	aux.convertTo(cv_image, CV_16UC1);
}

std::string BuildMASLayerName(size_t cntrNumber, std::string *out_name = nullptr) {
	std::ostringstream ostr;
	ostr << "contour" << '_' << cntrNumber;

	std::string layer_name = ostr.str();

	if (out_name != nullptr) {
		ostr << "out";
		(*out_name) = ostr.str();
	}

	return layer_name;
}


size_t ConductOverlapElimination(
	const std::vector<std::vector<cv::Point2d>>& contours, 
	std::vector<std::vector<cv::Point2d>>& final_contours,
	bool preserve_scale_factor, // causes to bypass back-scaling and return the used scale_factor
	long& scale_factor, //in/out, specifies what scale factor to use
	bool conduct_size = false, 
	int size_increment = -1, 
	bool log_graph = false) {

	if (contours.size() == 0) {
		return 0; 
	}

	MASInitialize(1, 1);

	std::string cl; 

	size_t cntrNmbr = 0;
	for (auto& contour : contours) {
		if (contour.size() < 3) {
			continue; 
		}

		std::string out_name;
		std::string layer_name = BuildMASLayerName(++cntrNmbr, &out_name);

		MASLayerCreateDBStorage(layer_name.c_str(), scale_factor);

		const double s_mult = ((size_t)1) << scale_factor;

		_sAlong x;
		_sAlong y;
		size_t j = 0;
		for (auto p : contour) {
			x[j] = p.x * s_mult;
			y[j] = p.y * s_mult;
			++j;
		}
		if (j > 0) {
			MASLayerAddPolygon2DBStorage(layer_name.c_str(), x, y);
		}

		bool rc;

		if (cntrNmbr == 1) {
			rc = MASEvaluate1(MAS_OverlapElim, layer_name.c_str(), out_name.c_str()/*"out"*/, log_graph, preserve_scale_factor);
			cl = out_name.c_str();
		}
		else {
			rc = MASEvaluate1(MAS_OverlapElim, layer_name.c_str(), out_name.c_str()/*"out"*/, log_graph, preserve_scale_factor);
			if (rc) {
				const char* layers[] = { cl.c_str(), out_name.c_str() };
				MASEvaluate2(MAS_Or, layers, 2, out_name.c_str()/*"out"*/, log_graph, preserve_scale_factor);
				cl = out_name.c_str();
			}
		}
	}

	if (cntrNmbr == 0) {
		return 0; 
	}

	bool ok = true; 
	if (size_increment != 0) {
		ok = MASSize(cl.c_str()/*"out"*/, "out", size_increment, log_graph, preserve_scale_factor);
		if (ok) {
			cl = "out";
			if (!conduct_size) {
				ok = MASSize("out", "out", -size_increment, log_graph, preserve_scale_factor);
			}
		}
	}

	MASEvaluate1(MAS_ActivateLayer, cl.c_str()/*"out"*/);

	size_t nFirst = 0;
	long nPolygons = MASLayerCountRTPolygons(cl.c_str()/*"out"*/, &nFirst);

	size_t count = 0;
	for (size_t n = nFirst; count < nPolygons && n < ((size_t)nPolygons << 2); ++n) {
		_sAlong x;
		_sAlong y;
		long s_factor = 0;
		long nPoints = MASLayerGetRTPolygon(cl.c_str()/*"out"*/, n, x, y, s_factor);
		if (nPoints > 4) {
			if (count == 0) {
				final_contours.resize(0);
				final_contours.resize(nPolygons + 1);
			}

			std::vector<Point2d>& contour = final_contours[count++];
			contour.resize(nPoints);

			const double s_mult = 1.0 / (double)(((size_t)1) << s_factor);

			long j = 0;
			while (j < nPoints) {
				contour[j] = Point2d(x[j] * s_mult, y[j] * s_mult);
				++j;
			}

			if (s_factor > scale_factor) {
				scale_factor = s_factor;
			}
		}
	}

	return ok? count: 0; 
}

size_t ConductOverlapEliminationEx(const std::vector<std::vector<cv::Point2d>>& contours, std::vector<std::vector<cv::Point2d>>& final_contours, 
	bool preserve_scale_factor, // causes to bypass back-scaling and return the used scale_factor
	long& scale_factor, // in/out. specifies what power of 2 to use to get coordinates in integers
	bool conduct_size, // false
	int size_increment, // -1
	bool log_graph/*false*/) {

	size_t count = ConductOverlapElimination(contours, final_contours, preserve_scale_factor, scale_factor, conduct_size, size_increment, log_graph);
	if (count == 0) {
		size_t j = 0; 
		std::vector<std::vector<cv::Point2d>> aux(contours.size());
		for (auto& contour : contours) {
			if (contour.size()) {
				aux[j].resize(contour.size());
				std::reverse_copy(contour.cbegin(), contour.cend(), aux[j].begin());
				if (!size_increment) {
					smoothContour(aux[j]);
				}
				for (size_t c = aux[j].size(), nc = c - 1; c > nc && nc > 0; nc = aux[j].size()) {
					c = nc;
					linearizeContour(aux[j], 2.25, 7);
				}
				if (!size_increment) {
					final_contours.push_back(aux[j]);
				}
				++j;
			}
		}
		if (size_increment) {
			count = ConductOverlapElimination(aux, final_contours, preserve_scale_factor, scale_factor, conduct_size, size_increment);
		}
		else {
			count = j;
		}
	}

	return count; 
}

return_t __stdcall EvaluateContours(LPVOID lp) {
	SPointsReconstructionCtl *ctl = (SPointsReconstructionCtl*)lp;



	ctl->_gate.lock();
	ctl->_status--;
	ctl->_edgesWindows[0] = "Camera1"; // windowNumber 1 - number is assigned in DisplayReconstructionData
	ctl->_edgesWindows[1] = "Camera2"; // windowNumber 2
	ctl->_pointsCropWindows[0] = "Camera1Fit"; // windowNumber 3
	ctl->_pointsCropWindows[1] = "Camera2Fit"; // windowNumber 4
	ctl->_combinedFitWindows[0] = "Combined1Fit"; // windowNumber 5
	ctl->_combinedFitWindows[1] = "Combined2Fit"; // windowNumber 6
	ctl->_gate.unlock();


	g_configuration._visual_diagnostics = true;


	int64_t image_localtime = 0;


	Mat cv_image[4];
	std::vector<ABox> boxes[2];
	std::vector<ClusteredPoint> cv_points[2];
	std::vector<cv::Point2d> in_points[2];

	Mat unchangedImage;
	Mat finalContoursImage;



	std::vector<std::vector<cv::Point2d>> final_contours(1);


	bool image_isok = false;
	while (!g_bTerminated && !ctl->_terminated) {

		Mat cv_edges[6];

		bool arff_file_requested = false;

		std::string file_name; 
		std::string path_name;

		if (g_configuration._frames_acquisition_mode > 0) {
			if (!image_isok) {
				image_isok = GetFramesFromFilesWithSubdirectorySearch(cv_image/*out*/, in_points, arff_file_requested/*out*/, file_name, path_name);
				Sleep(20);
			}
		}
		else {
			if (!image_isok) {
				image_isok = GetImages(cv_image[0], cv_image[1], &image_localtime, 1);
				if (ctl->_calibration_exists) {
					Mat undistorted;
					remap(cv_image[0], undistorted, ctl->_map_l[0], ctl->_map_l[1], INTER_CUBIC/*INTER_LINEAR*//*INTER_NEAREST*/, BORDER_CONSTANT);
					cv_image[0] = undistorted;
					undistorted = Mat();
					remap(cv_image[1], undistorted, ctl->_map_r[0], ctl->_map_r[1], INTER_CUBIC/*INTER_LINEAR*//*INTER_NEAREST*/, BORDER_CONSTANT);
					cv_image[1] = undistorted;
					undistorted = Mat();
				}
				//Sleep(20);
			}
		}
		_g_images_frame->Invalidate();


		final_contours.resize(0); 

		if (image_isok) {

			int64_t time_start = OSDayTimeInMilliseconds();

			if (cv_image[0].cols == 0 || cv_image[1].cols == 0) {
				continue;
			}

			cv_image[2] = cv_image[0].clone();
			cv_image[3] = cv_image[1].clone();


			if (g_configuration._frames_acquisition_mode > 0) {
				unchangedImage = imread(std::string(g_path_nwpu_images_dir) + path_name + ".jpg", ImreadModes::IMREAD_ANYDEPTH | ImreadModes::IMREAD_ANYCOLOR);
			}
			else {
				unchangedImage = cv_image[0].clone();
			}





			double chWeights[3] = { 0.3, 0.59, 0.11 };

			auto prepImages = [&cv_image, &chWeights]() {
				StandardizeImage(cv_image[0], chWeights);
				SquareImage(cv_image[1], chWeights);
			};

			auto readyImages = [&cv_image, &cv_edges](int window) {
				normalize(cv_image[window].clone(), cv_image[window], 0, (size_t)256 * g_bytedepth_scalefactor, NORM_MINMAX, CV_16UC1, Mat());

				Mat aux = cv_image[window];
				int anysotropicIntensity = 7 - 3 * (aux.rows / 480); // takes too long on large images, so skip it alltogether for large images.
				if (anysotropicIntensity > 0) {
					medianBlur(aux.clone(), aux, 3);

					AnisotropicDiffusion(aux, anysotropicIntensity);
				}
				aux.convertTo(cv_image[window], CV_16UC1);

				cv_edges[window] = cv_image[window].clone();
			};





			prepImages();

			cv_image[0] = mat_invert2word(cv_image[0]);
			readyImages(0);

			cv_image[1] = mat_loginvert2byte(cv_image[1]);
			readyImages(1);



			g_imageSize = cv_image[0].size();


			auto submitGraphics = [&](Mat& originalImage, bool data_is_valid = false) {
				ctl->_gate.lock();
				//ctl->_cv_image[0] = cv_image[0].clone();
				//ctl->_cv_image[1] = cv_image[1].clone();
				ctl->_cv_edges[0] = cv_edges[0].clone();
				ctl->_cv_edges[1] = cv_edges[1].clone();
				ctl->_pixel_threshold = 91;
				ctl->_data_isvalid = false;
				ctl->_image_isvalid = true;
				ctl->_last_image_timestamp = image_localtime;
				ctl->_unchangedImage[0] = originalImage.clone();
				ctl->_unchangedImage[1] = unchangedImage.clone();

				ctl->_draw_epipolar_lines = false; 

				if (data_is_valid) {
					ctl->_boxes[0] = boxes[0];
					ctl->_boxes[1] = boxes[1];
					ctl->_cv_points[0] = cv_points[0];
					ctl->_cv_points[1] = cv_points[1];
					ctl->_data_isvalid = true;
				}
				ctl->_gate.unlock();
			};


			submitGraphics(unchangedImage);



			std::vector<std::vector<cv::Point2d>> contours(1);
			size_t contours_count = 0;

			long max_scale_factor = 0;

			while (!g_bTerminated && !ctl->_terminated && image_isok) {
				HANDLE handles[] = { g_event_SeedPointIsAvailable, g_event_ContourIsConfirmed };
				DWORD anEvent = WaitForMultipleObjectsEx(ARRAY_NUM_ELEMENTS(handles), handles, FALSE, 100, TRUE);
				if (anEvent == WAIT_TIMEOUT) {
					continue;
				}
				if (anEvent == (WAIT_OBJECT_0 + 1)) {
					switch (g_LoG_seedPoint.params.windowNumber) {
					case 3:
					case 4:
						break;
					case 5:
					case 6:
						image_isok = false;
						break;
					}

					if (image_isok) {
						contours.resize(contours_count);
						for (auto& contour : final_contours) {
							if (contour.size()) {
								contours.push_back(contour);
							}
						}

						size_t count = ConductOverlapEliminationEx(contours, final_contours, max_scale_factor != 0, max_scale_factor, false, 0);
						if (count == 0) {
							final_contours = contours;
						}

						finalContoursImage = unchangedImage.clone();
						for (auto& contour : final_contours) {
							if (contour.size()) {
								Point a = Point2f(contour[0]);
								for (int j = 1; j < contour.size(); ++j) {
									Point b = Point2f(contour[j]);
									cv::line(finalContoursImage, a, b, Scalar(0, (size_t)256 * 256, 0));
									a = b;
								}
							}
						}

						submitGraphics(finalContoursImage);
					}
				}
				if (anEvent == WAIT_OBJECT_0) {
					//int pixel_threshold = (int)g_otsu_threshold; // gets calculated in separate thread. 
					//if (pixel_threshold < 0) {
					//	pixel_threshold = 31;
					//}
					//pixel_threshold *= g_bytedepth_scalefactor;

					//pixel_threshold /= 2;



					int kmatN = cv_image[0].cols / 50;
					if (kmatN <= 5) {
						kmatN = 5;
					}
					else {
						kmatN = 7;
					}

					Mat_<double> kmat = LoG(1.25 * (kmatN / 7.0), kmatN);


					cv::Rect roi;

					ImageScaleFactors sf = g_LoG_seedPoint.params.scaleFactors;
					roi.x = (int)(g_LoG_seedPoint.x / sf.fx + 0.5) - 4;
					roi.y = (int)(g_LoG_seedPoint.y / sf.fy + 0.5) - 4;

					roi.height = kmatN;
					roi.width = kmatN;

					Point pt;
					pt.x = roi.x + roi.width / 2;
					pt.y = roi.y + roi.height / 2;


					int windowNumber = g_LoG_seedPoint.params.windowNumber - 1;

					if (g_LoG_seedPoint.eventValue == 2/*right button*/) {
						Mat aux = cv_image[2 + windowNumber].clone();

						Mat mean;
						Mat invCovar;
						Mat invCholesky;
						Mat stdDev;
						Mat factorLoadings;

						if (BuildIdealChannels_Distribution(aux, pt, mean, stdDev, factorLoadings, invCovar, invCholesky)) {
							StandardizeImage_Likeness(aux, mean, stdDev, factorLoadings, invCovar, invCholesky);

							cv_image[windowNumber] = mat_loginvert2word(aux);
							cv_image[windowNumber] = mat_invert2word(cv_image[windowNumber]);

						}
						else {
								double chIdeal[3];
								BuildIdealChannels_Likeness(aux, pt, chIdeal);

								auto prepImages_HSV_Likeness = [&cv_image, &chWeights](double chIdeal[3], int window) {
									Mat aux = cv_image[2 + window].clone();
									if (StandardizeImage_HSV_Likeness(aux, chIdeal)) {
										cv_image[window] = mat_loginvert2word(aux);
										cv_image[window] = mat_invert2word(cv_image[window]);
									}
									else {
										Mat aux = cv_image[2 + window].clone();
										if (window == 0) {
											StandardizeImage(aux, chWeights);
											cv_image[window] = mat_invert2word(aux);
										}
										else {
											SquareImage(aux, chWeights);
											cv_image[window] = mat_loginvert2byte(aux);
										}
									}
								};

								prepImages_HSV_Likeness(chIdeal, windowNumber);
						}

						readyImages(windowNumber);
					}


					submitGraphics(unchangedImage);



					cv_edges[0].convertTo(cv_edges[4], CV_8UC1);
					cv_edges[1].convertTo(cv_edges[5], CV_8UC1);

					double tr[2] = { (double)mat_get<uint16_t>(cv_edges[4], pt.y, pt.x), (double)mat_get<uint16_t>(cv_edges[5], pt.y, pt.x) };
					tr[0] = threshold(cv_edges[4], cv_edges[2] = Mat(), tr[0], 65535, THRESH_OTSU);
					tr[1] = threshold(cv_edges[5], cv_edges[3] = Mat(), tr[1], 65535, THRESH_OTSU);

					cv_edges[2] = cv_edges[0].clone();
					cv_edges[3] = cv_edges[1].clone();
					mat_threshold(cv_edges[2], (tr[0] * 0.6) * g_bytedepth_scalefactor);
					mat_threshold(cv_edges[3], (tr[1] * 0.6) * g_bytedepth_scalefactor);

					cv_image[0] = cv_edges[2].clone();
					cv_image[1] = cv_edges[3].clone();

					unsigned int effective_threshold[2] = { mat_get<uint16_t>(cv_edges[2], pt.y, pt.x), mat_get<uint16_t>(cv_edges[3], pt.y, pt.x) };
					unsigned int effective_threshold_min = std::min(effective_threshold[0], effective_threshold[1]);

					int idx[2] = { -1, -1 }; // indices of detected rectangles
					ushort intensity[2] = { 0, 0 };

					g_max_boxsize_pixels = std::max(g_imageSize.height, g_imageSize.width);

					ctl->_gate.lock();
					idx[0] = BlobCentersLoG(boxes[0], cv_points[0], cv_edges[2], effective_threshold[0], roi, kmat, arff_file_requested, &intensity[0]);
					idx[1] = BlobCentersLoG(boxes[1], cv_points[1], cv_edges[3], effective_threshold[1], roi, kmat, arff_file_requested, &intensity[1]);
					ctl->_gate.unlock();


					if (arff_file_requested) {
						for (auto& points : cv_points) {
							for (auto& point : points) {
								bool isValid = point._cluster > -1;
								bool rectangleDetected = isValid && point._isACenter != 0;
								point.ARFF_Output(isValid, rectangleDetected);
							}
						}
					}

					std::vector<ABox>& boxes_selected = boxes[windowNumber];


					if (boxes_selected.size() == 0) {
						continue;

					}


					contours.resize(1);
					CopyVector(contours[0], boxes_selected[0].contour);


					printf("\n\n");


					int pass_number = 0;
					int size_increment = 1; 
					int iteration_number = 0; 
					int max_passes = 3; 

					max_scale_factor = 0;

					finalContoursImage = unchangedImage.clone();
					while (0<1) {
						submitGraphics(finalContoursImage, true);

						if (boxes_selected.size() == 0) {
							contours_count = 0;
							break;
						}


						if (pass_number++ == max_passes) {
							pass_number = 1;
							size_increment = -size_increment;
							if (++iteration_number >= 3) {
								break;
							}
							switch (iteration_number) {
							case 1:
								max_passes <<= 1; 
								break; 
							case 2:
								max_passes >>= 1;
								break;
							}
						}

						std::vector<std::vector<cv::Point2d>> local_contours;

						std::ostringstream ostr;
						ostr << "---" << "size_increment=" << size_increment << ' ' << "pass_number=" << pass_number << ' ' << "max_passes=" << max_passes << std::endl;
						printf(ostr.str().c_str());

						size_t count = ConductOverlapEliminationEx(contours, local_contours, false, max_scale_factor, true, size_increment, g_LoG_seedPoint.eventValue == 3/*central button*/);

						if (count == 0) {
							contours.resize(1);
							CopyVector(contours[0], boxes_selected[0].contour);
							pass_number == max_passes;
							iteration_number = 3; 
							continue;
						}

						contours.swap(local_contours);
						contours_count = count;

						ctl->_gate.lock();
						ClusteredPoint& point = cv_points[windowNumber][0];
						point._contours = contours;
						CopyVector(point._contour_notsmoothed, boxes_selected[0].contour_notsmoothed);
						ctl->_gate.unlock();
					}
				}
			}
		}

	}

	ctl->_gate.lock();
	ctl->_status--;
	ctl->_gate.unlock();

	return 0;
}

return_t __stdcall RenderCameraImages(LPVOID lp) {
	SPointsReconstructionCtl* ctl = (SPointsReconstructionCtl*)lp;


	g_configuration._visual_diagnostics = true;



	ctl->_gate.lock();
	ctl->_status--;
	ctl->_edgesWindows[0] = "Camera1";
	ctl->_edgesWindows[1] = "Camera2";
	ctl->_pointsCropWindows[0] = "Camera1Fit";
	ctl->_pointsCropWindows[1] = "Camera2Fit";
	ctl->_combinedFitWindows[0] = "Combined1Fit";
	ctl->_combinedFitWindows[1] = "Combined2Fit";
	ctl->_gate.unlock();



	int64 image_localtime = 0;


	Mat cv_image[4];
	std::vector<ABox> boxes[2];
	std::vector<ClusteredPoint> cv_points[2];

	Mat unchangedImage;

	cv::Mat finalContoursImage;
	cv::Point detectedPoint(6, 6);
	int targetWindow = 0;



	std::vector<std::vector<cv::Point2d>> final_contours(1);


	bool image_isok = false;
	while (!g_bTerminated && !ctl->_terminated) {

		Mat cv_edges[6];

		bool arff_file_requested = false;

		std::string file_name;
		std::string path_name;

		if (g_configuration._frames_acquisition_mode < 0) {
			image_isok = GetLastFrame(cv_image[0], cv_image[1], &image_localtime);
		}

		final_contours.resize(0);

		if (image_isok) {

			__int64 time_start = OSDayTimeInMilliseconds();

			if (cv_image[0].cols == 0 || cv_image[1].cols == 0) {
				continue;
			}

			if (ctl->_calibration_exists) {
				Mat undistorted;
				remap(cv_image[0], undistorted, ctl->_map_l[0], ctl->_map_l[1], INTER_CUBIC/*INTER_LINEAR*//*INTER_NEAREST*/, BORDER_CONSTANT);
				cv_image[0] = undistorted;
				undistorted = Mat();
				remap(cv_image[1], undistorted, ctl->_map_r[0], ctl->_map_r[1], INTER_CUBIC/*INTER_LINEAR*//*INTER_NEAREST*/, BORDER_CONSTANT);
				cv_image[1] = undistorted;
				undistorted = Mat();
			}

			cv_image[2] = cv_image[0].clone();
			cv_image[3] = cv_image[1].clone();


			cv_edges[0] = cv_image[0].clone();
			cv_edges[1] = cv_image[1].clone();


			unchangedImage = cv_image[0].clone();

			finalContoursImage = cv_image[targetWindow].clone();
			cv::circle(finalContoursImage, detectedPoint, 10, Scalar(0, 255, 0), -1);

			g_imageSize = cv_image[0].size();


			auto submitGraphics = [&](Mat& originalImage, bool data_is_valid = false) {
				ctl->_gate.lock();
				ctl->_cv_edges[0] = cv_edges[0].clone();
				ctl->_cv_edges[1] = cv_edges[1].clone();
				ctl->_data_isvalid = false;
				ctl->_image_isvalid = true;
				ctl->_last_image_timestamp = image_localtime;
				ctl->_unchangedImage[0] = originalImage.clone();
				ctl->_unchangedImage[1] = unchangedImage.clone();

				ctl->_draw_epipolar_lines = true;

				if (data_is_valid) {
					ctl->_boxes[0] = boxes[0];
					ctl->_boxes[1] = boxes[1];
					ctl->_cv_points[0] = cv_points[0];
					ctl->_cv_points[1] = cv_points[1];
					ctl->_data_isvalid = true;
				}
				ctl->_gate.unlock();
			};


			submitGraphics(finalContoursImage, cv_points[0].size() || cv_points[1].size());


			while (!g_bTerminated && !ctl->_terminated && image_isok) {
				HANDLE handles[] = { g_event_SeedPointIsAvailable, g_event_ContourIsConfirmed };
				DWORD anEvent = WaitForMultipleObjectsEx(ARRAY_NUM_ELEMENTS(handles), handles, FALSE, 0, TRUE);
				if (anEvent == WAIT_TIMEOUT) {
					break;
				}
				if (anEvent == WAIT_OBJECT_0) {
					switch (g_LoG_imageWindowNumber) {
					case 1:
					case 2:
						break;
					default:
						continue;
					}
				}
				else {
					continue;
				}

				ImageScaleFactors sf = g_LoG_seedPoint.params.scaleFactors;
				Point pt;
				pt.x = g_LoG_seedPoint.x / sf.fx + 0.5;
				pt.y = g_LoG_seedPoint.y / sf.fy + 0.5;

				int windowNumber = g_LoG_seedPoint.params.windowNumber - 1;
				int direction = windowNumber == 1 ? 1 : -1;

				targetWindow = windowNumber - direction;


				Mat aux = cv_image[2 + windowNumber].clone();

				auto checkPoint = [&aux](cv::Point& pt) {
					if (pt.x < 0) {
						pt.x = 0;
					}
					if (pt.y < 0) {
						pt.y = 0;
					}
					if (pt.x > aux.cols) {
						pt.x = aux.cols;
					}
					if (pt.y > aux.rows) {
						pt.y = aux.rows;
					}
				};
				auto checkRectangle = [&aux, &checkPoint](cv::Rect& rect) {
					cv::Point pt1(rect.x, rect.y);
					cv::Point pt2(rect.x + rect.width, rect.y + rect.height);
					checkPoint(pt1);
					checkPoint(pt2);

					rect = cv::Rect(pt1, pt2);
				};
				auto default_cv_points = [&cv_points]() {
					cv_points[0].resize(1);
					cv_points[1].resize(1);
					cv_points[0][0]._crop = imread(IMG_DELETEDOCUMENT_H, cv::IMREAD_ANYCOLOR);
					cv_points[1][0]._crop = imread(IMG_DELETEDOCUMENT_H, cv::IMREAD_ANYCOLOR);
				};


				int strip2searchWidth = 0.15 * aux.cols;
				const int patternHalfWidth = ((strip2searchWidth/15) >> 1) << 1; //40
				const int blurHeight = 5;

				Mat crop(aux, cv::Rect(pt.x - patternHalfWidth, pt.y - blurHeight / 2, 2 * patternHalfWidth + 1, blurHeight));
				cv::blur(crop.clone(), crop, cv::Size(1, blurHeight));


				cv::Rect strip2searchRect(pt.x, pt.y - blurHeight / 2, strip2searchWidth, blurHeight);
				if (direction == -1) {
					strip2searchRect.x -= strip2searchWidth;
				}
				checkRectangle(strip2searchRect);


				Mat strip2search(cv_image[2 + windowNumber - direction], strip2searchRect);
				cv::blur(strip2search.clone(), strip2search, cv::Size(1, blurHeight));



				Mat mean;
				Mat invCovar;
				Mat_<double> invCholesky(3, 3);
				Mat stdDev;
				Mat factorLoadings;

				double hsvIdeal[3];

				double* mean_data = nullptr;
				double likeness = 0;


				std::function<int(cv::Mat&, cv::Point&)> likenessScore; // returns a value discretized from 0 to 10 to represent likeness scrore.

				if (BuildIdealChannels_Distribution(crop, cv::Point(patternHalfWidth, blurHeight / 2), mean, stdDev, factorLoadings, invCovar, invCholesky, blurHeight / 2)) {
					mean_data = (double*)mean.data;
				}
				else {
					std::cout << "Using HSV transform" << std::endl;

					double rgbIdeal[3];
					BuildIdealChannels_Likeness(crop, cv::Point(patternHalfWidth, blurHeight / 2), rgbIdeal, blurHeight / 2);

					const double y_componentIdeal = rgbIdeal[0] * 0.299 + rgbIdeal[1] * 0.587 + rgbIdeal[2] * 0.114;
					if (y_componentIdeal < 40) {
						default_cv_points();
						continue;
					}

					cv::Vec<uchar, 3> pixIdeal;
					pixIdeal[0] = (uchar)(rgbIdeal[0] + 0.5);
					pixIdeal[1] = (uchar)(rgbIdeal[1] + 0.5);
					pixIdeal[2] = (uchar)(rgbIdeal[2] + 0.5);

					RGB_TO_HSV(pixIdeal, hsvIdeal);
				}

				double invCholesky_data[3][3] = {
					{ invCholesky.at<double>(0, 0), invCholesky.at<double>(0, 1), invCholesky.at<double>(0, 2) },
					{ invCholesky.at<double>(1, 0), invCholesky.at<double>(1, 1), invCholesky.at<double>(1, 2) },
					{ invCholesky.at<double>(2, 0), invCholesky.at<double>(2, 1), invCholesky.at<double>(2, 2) }
				};

				if (mean_data == nullptr) {
					likenessScore = [&hsvIdeal](cv::Mat& aux, cv::Point& pt) -> int {
						double likeness = hsvLikenessScore(aux.at<cv::Vec<uchar, 3>>(pt.y, pt.x), hsvIdeal);
						return 2 * likeness / 25.6 + 0.45;
					};
				}
				else {
					likenessScore = [&mean_data, &invCholesky_data](cv::Mat& aux, cv::Point& pt) -> int {
						double zScore = Get_Squared_Z_Score(aux.at<cv::Vec<uchar, 3>>(pt.y, pt.x), mean_data, invCholesky_data);
						if (zScore < 10) {
							return 2 * (10 - zScore) + 0.45;
						}

						return 0;
					};
				}

				std::vector<int> pattern(crop.cols);
				std::vector<int> strip2searchForPattern(strip2search.cols);

				for (int c = 0; c < crop.cols; ++c) {
					pattern[c] = likenessScore(crop, cv::Point(c, 2));
				}
				for (int c = 0; c < strip2search.cols; ++c) {
					strip2searchForPattern[c] = likenessScore(strip2search, cv::Point(c, 2));
				}

				int pos = FindBestAlignment(pattern, strip2searchForPattern) + 0.45;
				if (pos <= 0) {
					std::cout << "Unable to determnine the match for the selected point" << std::endl; 
					default_cv_points();
					continue;
				}

				detectedPoint.x = pt.x + direction * pos + 0.45;
				detectedPoint.y = pt.y;

				cv::line(crop, cv::Point(patternHalfWidth, 0), cv::Point(patternHalfWidth, blurHeight - 1), Scalar(0, 255, 0));
				cv::line(strip2search, cv::Point(pos, 0), cv::Point(pos, blurHeight - 1), Scalar(0, 255, 0));

				cv_points[windowNumber].resize(1);
				cv_points[windowNumber][0]._crop = crop;

				cv::Point pt1(pos - 2 * patternHalfWidth, 0);
				cv::Point pt2(pos + 2 * patternHalfWidth + 1, blurHeight);
				if (pt1.x < 0) {
					pt1.x = 0;
				}
				if (pt2.x > (strip2search.cols)) {
					pt2.x = strip2search.cols;
				}

				cv_points[targetWindow].resize(1);
				cv_points[targetWindow][0]._crop = Mat(strip2search, cv::Rect(pt1, pt2));
			}
		}
	}

	ctl->_gate.lock();
	ctl->_status--;
	ctl->_gate.unlock();

	return 0;
}

return_t __stdcall ReconstructPoints(LPVOID lp) {
	timeBeginPeriod(1);
	try {
		if (g_configuration._evaluate_contours) {
			EvaluateContours(lp); 
		}
		else{
			RenderCameraImages(lp);
		}
	}
	catch(Exception& ex) {
		std::cout << ex.msg << std::endl;
		g_bTerminated = true;
		((SPointsReconstructionCtl*)lp)->_status--;
	}
	timeEndPeriod(1);
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
	return 0;
}




void launch_reconstruction(SImageAcquisitionCtl& image_acquisition_ctl, SPointsReconstructionCtl *ctl) {
	if(!g_bTerminated) {

		FileStorage fs(CalibrationFileName(), FileStorage::READ);

		fs["cameraMatrix_l"] >> ctl->_cameraMatrix[0];
		fs["cameraMatrix_r"] >> ctl->_cameraMatrix[1];
		fs["distCoeffs_l"] >> ctl->_distortionCoeffs[0];
		fs["distCoeffs_r"] >> ctl->_distortionCoeffs[1];
		fs["cameraMatrix_l_first"] >> ctl->_cameraMatrix[2]; // undistort on first step of calibration
		fs["cameraMatrix_r_first"] >> ctl->_cameraMatrix[3];
		fs["distCoeffs_l_first"] >> ctl->_distortionCoeffs[2];
		fs["distCoeffs_r_first"] >> ctl->_distortionCoeffs[3];
		fs["R"] >> ctl->_R;
		fs["T"] >> ctl->_T;
		fs["E"] >> ctl->_E;
		fs["F"] >> ctl->_F;
		fs["R_l"] >> ctl->_Rl;
		fs["R_r"] >> ctl->_Rr;
		fs["P_l"] >> ctl->_Pl;
		fs["P_r"] >> ctl->_Pr;
		fs["Q"] >> ctl->_Q;
		//fs["map_l1"] >> ctl->_map_l[0];
		//fs["map_l2"] >> ctl->_map_l[1];
		//fs["map_r1"] >> ctl->_map_r[0];
		//fs["map_r2"] >> ctl->_map_r[1];
		//fs["map_l1_first"] >> ctl->_map_l[2];
		//fs["map_l2_first"] >> ctl->_map_l[3];
		//fs["map_r1_first"] >> ctl->_map_r[2];
		//fs["map_r2_first"] >> ctl->_map_r[3];
		fs["roi_l"] >> ctl->_roi[0];
		fs["roi_r"] >> ctl->_roi[1];

		fs["rectified_image_size"] >> ctl->_rectified_image_size;

		fs.release();


		ctl->_calibration_exists = ctl->_cameraMatrix[0].cols > 0 && ctl->_cameraMatrix[1].cols > 0;

		if (ctl->_calibration_exists) {
			cv::initUndistortRectifyMap(ctl->_cameraMatrix[0], ctl->_distortionCoeffs[0], ctl->_Rl, ctl->_Pl, ctl->_rectified_image_size, CV_16SC2/*CV_32F*/, ctl->_map_l[0], ctl->_map_l[1]);
			cv::initUndistortRectifyMap(ctl->_cameraMatrix[1], ctl->_distortionCoeffs[1], ctl->_Rr, ctl->_Pr, ctl->_rectified_image_size, CV_16SC2/*CV_32F*/, ctl->_map_r[0], ctl->_map_r[1]);
		}


		ctl->_status = 4; // each thread decrements twice; default 2 threads. 
		ctl->_terminated = 0;
		if(g_configuration._frames_acquisition_mode == 1) { // reads from files sequentially, so just one thread.
			ctl->_status -= 2;
		}
		else
		if(g_configuration._frames_acquisition_mode > 1) { // for memory leak detection. reads just one image from file per thread (N alltogether), then cycles through those images. 
			ctl->_status = g_configuration._frames_acquisition_mode * 2;
		}
		else
		if(g_configuration._frames_acquisition_mode < 0) { // -N means read from cameras with N threads. 
			ctl->_status = -g_configuration._frames_acquisition_mode * 2;
		}

		const int N = (ctl->_status >> 1); // number of reconstruction threads. 

		//ctl->_status += 2; // threshold calculation thread also decrements twice.  

		for(int j = 0; j < N; ++j) {
			QueueWorkItem(ReconstructPoints, ctl);
		}

		//QueueWorkItem(EvaluateOtsuThreshold, ctl);
	}
}





struct ComparePointsOrigin {
	const ReconstructedPoint& _point;
	ComparePointsOrigin(const ReconstructedPoint& point): _point(point) {
	}
	bool operator()(const ReconstructedPoint& point) const {
		return point._id == _point._id;
	}
	bool operator()(const ReconstructedPoint *point) const {
		return point->_id == _point._id;
	}
};

struct Compare4DPointsForClusterByDistance { // partitioning in order to detect reference tag (3 LEDs) 
	double threshold;
	Compare4DPointsForClusterByDistance(): threshold(pow(g_max_clusterdistance, 2)) {
	}
	template<typename M>
	bool operator()(const M left, const M right) const {
		double x[2] = {left(0), right(0)};
		double y[2] = {left(1), right(1)};
		double z[2] = {left(2), right(2)};

		return (pow(x[0] - x[1], 2) + pow(y[0] - y[1], 2) + pow(z[0] - z[1], 2)) < threshold;
	}
};

template<typename T>
double distance3d(const T& left, const T& right, T& vec) {
	double sum = 0;
	for(int j = 0; j < 3; ++j) { 
		vec(j) = right(j) - left(j);
		sum += pow(vec(j), 2);
	}
	return sqrt(sum);

}

template<typename T>
inline double distance3d(const T& left, const T& right) {
	T vec; 
	return distance3d(left, right, vec); 
}

struct ComparePairsForDistances {
	bool operator()(const std::vector<Mat_<double>*>& left, const std::vector<Mat_<double>*>& right) const {
		return distance3d(*(left[0]), *(left[1])) < distance3d(*(right[0]), *(right[1]));
	}
	bool operator()(const std::vector<ReconstructedPoint*>& left, const std::vector<ReconstructedPoint*>& right) const {
		return distance3d(*(left[0]), *(left[1])) < distance3d(*(right[0]), *(right[1]));
	}
};

Point3d cross3d(Point3d src[2]) {
	Point3d dst;
	dst.x = src[0].y*src[1].z - src[0].z*src[1].y;
	dst.y = src[0].z*src[1].x - src[0].x*src[1].z;
	dst.z = src[0].x*src[1].y - src[0].y*src[1].x;
	return dst;
}
Mat_<double> cross3d(const Mat_<double>& src0, const Mat_<double>& src1) {
	Mat_<double> dst(src0.rows, src0.cols);
	dst(0) = src0(1)*src1(2) - src0(2)*src1(1);
	dst(1) = src0(2)*src1(0) - src0(0)*src1(2);
	dst(2) = src0(0)*src1(1) - src0(1)*src1(0);
	return dst;
}
Matx41d cross3d(const Matx41d& src0, const Matx41d& src1) {
	Matx41d dst;
	dst(0) = src0(1)*src1(2) - src0(2)*src1(1);
	dst(1) = src0(2)*src1(0) - src0(0)*src1(2);
	dst(2) = src0(0)*src1(1) - src0(1)*src1(0);
	dst(3) = 1;
	return dst;
}

double dot3d(const Point3d& src0, const Point3d& src1) {
	return src0.x*src1.x + src0.y*src1.y + src0.z*src1.z;
}
double dot3d(Point3d src[2]) {
	return src[0].x*src[1].x + src[0].y*src[1].y + src[0].z*src[1].z;
}
template<typename M>
double dot3d(const M& src0, const M& src1) {
	return src0(0)*src1(0) + src0(1)*src1(1) + src0(2)*src1(2);
}

Mat_<double> normal3d(const std::vector<Mat_<double>>& one, const std::vector<Mat_<double>>& two) {
	Point3d src[2];
	src[0].x = one[1](0) - one[0](0);
	src[0].y = one[1](1) - one[0](1);
	src[0].z = one[1](2) - one[0](2);
	src[1].x = two[1](0) - two[0](0);
	src[1].y = two[1](1) - two[0](1);
	src[1].z = two[1](2) - two[0](2);

	Point3d dst = cross3d(src);

	Mat_<double> mdst(4, 1);

	mdst(0) = dst.x;
	mdst(1) = dst.y;
	mdst(2) = dst.z;
	mdst(3) = 0;

	return mdst;
}
template<typename M> M normal3d(const std::vector<M>& one, const std::vector<M>& two) {
	Point3d src[2];
	src[0].x = one[1](0) - one[0](0);
	src[0].y = one[1](1) - one[0](1);
	src[0].z = one[1](2) - one[0](2);
	src[1].x = two[1](0) - two[0](0);
	src[1].y = two[1](1) - two[0](1);
	src[1].z = two[1](2) - two[0](2);

	Point3d dst = cross3d(src);

	M mdst;

	mdst(0) = dst.x;
	mdst(1) = dst.y;
	mdst(2) = dst.z;
	mdst(3) = 0;

	return mdst;
}






/**
From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
*/
Mat_<double> LinearLSTriangulation(Point3d u,	//homogenous image point (u,v,1)
	Matx34d P,	//camera 1 matrix
	Point3d u1,	//homogenous image point in 2nd camera
	Matx34d P1	//camera 2 matrix
	) {

	Matx43d A(u.x*P(2, 0) - P(0, 0), u.x*P(2, 1) - P(0, 1), u.x*P(2, 2) - P(0, 2),
		u.y*P(2, 0) - P(1, 0), u.y*P(2, 1) - P(1, 1), u.y*P(2, 2) - P(1, 2),
		u1.x*P1(2, 0) - P1(0, 0), u1.x*P1(2, 1) - P1(0, 1), u1.x*P1(2, 2) - P1(0, 2),
		u1.y*P1(2, 0) - P1(1, 0), u1.y*P1(2, 1) - P1(1, 1), u1.y*P1(2, 2) - P1(1, 2)
		);
	Matx41d B(-(u.x*P(2, 3) - P(0, 3)),
		-(u.y*P(2, 3) - P(1, 3)),
		-(u1.x*P1(2, 3) - P1(0, 3)),
		-(u1.y*P1(2, 3) - P1(1, 3)));

	Mat_<double> X;
	solve(A, B, X, DECOMP_SVD);

	return X;
}

#define EPSILON 0.005

/**
From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
*/
Mat_<double> IterativeLinearLSTriangulation(Point3d u,	//homogenous image point (u,v,1)
	Matx34d P,	//camera 1 matrix
	Point3d u1,	//homogenous image point in 2nd camera
	Matx34d P1	//camera 2 matrix
	) {
	double wi = 1, wi1 = 1;
	Mat_<double> X(4, 1);
	Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);

	X(0) = X_(0);
	X(1) = X_(1);
	X(2) = X_(2);
	X(3) = 1.0;

	//return X; 

	int i;
	for(i = 0; i < 10; ++i) { //Hartley suggests 10 iterations at most
		//recalculate weights
		double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
		double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);

		//breaking point
		if(std::abs(wi - p2x) <= EPSILON && std::abs(wi1 - p2x1) <= EPSILON) break;

		wi = p2x;
		wi1 = p2x1;

		//reweight equations and solve
		Matx43d A((u.x*P(2, 0) - P(0, 0)) / wi, (u.x*P(2, 1) - P(0, 1)) / wi, (u.x*P(2, 2) - P(0, 2)) / wi,
			(u.y*P(2, 0) - P(1, 0)) / wi, (u.y*P(2, 1) - P(1, 1)) / wi, (u.y*P(2, 2) - P(1, 2)) / wi,
			(u1.x*P1(2, 0) - P1(0, 0)) / wi1, (u1.x*P1(2, 1) - P1(0, 1)) / wi1, (u1.x*P1(2, 2) - P1(0, 2)) / wi1,
			(u1.y*P1(2, 0) - P1(1, 0)) / wi1, (u1.y*P1(2, 1) - P1(1, 1)) / wi1, (u1.y*P1(2, 2) - P1(1, 2)) / wi1
			);
		Mat_<double> B = (Mat_<double>(4, 1) << -(u.x*P(2, 3) - P(0, 3)) / wi,
			-(u.y*P(2, 3) - P(1, 3)) / wi,
			-(u1.x*P1(2, 3) - P1(0, 3)) / wi1,
			-(u1.y*P1(2, 3) - P1(1, 3)) / wi1
			);

		solve(A, B, X_, DECOMP_SVD);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
	}
	return X;
}


void CheckTrianglePointsOrder(std::vector<std::vector<ReconstructedPoint*>>& pairs) {
	if(pairs[0][1] == pairs[1][1]) {
		std::swap(pairs[0][0], pairs[0][1]);
		std::swap(pairs[1][0], pairs[1][1]);
	}
	else
	if(pairs[0][0] != pairs[1][0]) {
		std::swap(pairs[0][0], pairs[0][1]);
	}
	if(pairs[0][0] != pairs[1][0]) {
		std::swap(pairs[0][0], pairs[0][1]);
		std::swap(pairs[1][0], pairs[1][1]);
	}
}

double Eval_ReprojectError(
	Point2d& pl, /*in*/
	Point2d& pr, /*in*/
	Mat& PMl, /*in*/
	Mat& PMr, /*in*/
	Mat_<double>& val /*in*/
	) {

	Mat_<double> xim = PMr * val;	//reproject 
	double reproject_err = norm(Point2d(xim(0) / xim(2), xim(1) / xim(2)) - pr);
	xim = PMl * val;	//reproject 
	reproject_err += norm(Point2d(xim(0) / xim(2), xim(1) / xim(2)) - pl);

	return reproject_err / 2.0; 
}

struct Set_CameraNumber {
	int _cameraNumber; 
	void operator() (ClusteredPoint& point) {
		point._camera = _cameraNumber; 
	}
	Set_CameraNumber(int cameraNumber): _cameraNumber(cameraNumber) {
	}
}; 






struct ComparePointsForEpipolarEquivalence {
	const double _max_Y_error;
	const double _max_Y_error_center;
	ComparePointsForEpipolarEquivalence(double max_Y_error) : _max_Y_error(max_Y_error), _max_Y_error_center(2 * max_Y_error) {
	}
	bool operator()(const ClusteredPoint& left, const ClusteredPoint& right) const {
		if (left._isACorner != right._isACorner) {
			return false;
		}
		if (left._isACenter != right._isACenter) {
			return false;
		}
		if (left._isACorner) {
			return std::abs(left.y - right.y) <= left._corners_max_Y_error;
		}
		if (left._isACenter) {
			return std::abs(left.y - right.y) <= _max_Y_error_center;
		}
		return std::abs(left.y - right.y) <= _max_Y_error;
	}
};

void PartitionPoints(std::vector<ClusteredPoint> cv_points[2], std::vector<std::vector<ClusteredPoint*>> &clusters /*out*/) {
	std::vector<ClusteredPoint> combinedpoints(cv_points[0]);
	combinedpoints.insert(combinedpoints.end(), cv_points[1].begin(), cv_points[1].end());

	if(combinedpoints.size()) {
		//std::vector<int> ylevels(combinedpoints.size(), -1);
		std::vector<int> ylevels;
		partitionEx(combinedpoints, ylevels, ComparePointsForEpipolarEquivalence(g_max_Y_error));

		int onepointgroups_count = 0;

		std::vector<int>::iterator it_lbl = ylevels.begin();
		for(int j = 0; j < 2; ++j) {
			for(auto& point : cv_points[j]) {
				int cluster = point._cluster = *(it_lbl++);
				if((int)clusters.size() < (cluster + 1)) {
					clusters.resize(cluster + 1);
				}
				clusters[cluster].push_back(&point);
				switch(clusters[cluster].size()) {
				case 1:
				++onepointgroups_count;
				break;
				case 2:
				--onepointgroups_count;
				break;
				}
			}
		}
		for(auto& cluster : clusters) {
			ClusteredPoint* point = cluster[0]; 
			for(auto& other : cluster) {
				if(other->aType() != point->aType()) {
					std::ostringstream ostr; 
					ostr << "PartitionPoints: point type " << point->aType() << " != other point type " << other->aType() << std::endl; 
					std::cout << ostr.str(); 
					break; 
				}
			}
		}
		if(onepointgroups_count > 0) {
			for(auto& cluster : clusters) {
				if(cluster.size() >= 2) {
					int clusterType = cluster[0]->aType(); 
					int cluster_number = -1;
					double Y = 0;
					for(auto point_ptr : cluster) {
						cluster_number = point_ptr->_cluster;
						Y += point_ptr->y;
					}
					Y /= cluster.size();
					double max_Y_error = g_max_Y_error * (clusterType != 0? 2: 1.0);
					for(auto& c : clusters) {
						if(c.size() == 1) {
							if(c[0]->aType() == clusterType && std::abs(c[0]->y - Y) <= max_Y_error) {
								c[0]->_cluster = cluster_number;
								cluster.push_back(c[0]);
								c.resize(0);
							}
						}
					}
				}
				clusters.erase(
					std::remove_if(clusters.begin(), clusters.end(), [](std::vector<ClusteredPoint*> &cluster) -> bool {
					return cluster.size() == 0;
				}),
				clusters.end());
			}
		}
	}
}






double reconstruct4DPoint(Mat_<double>& X, ClusteredPoint& p1, ClusteredPoint& p2, Mat& F, Mat& Pl, Mat& Pr) { // returns reproject error
	X = Mat_<double>(4, 1, 0.0);

	const double image_center_x = g_imageSize.width / 2.0;
	const double image_center_y = g_imageSize.height / 2.0;

	// Heigher weight modifies more because the point is further away from center, i.e. more esscentric ellipse. 
	double w[3] = {image_center_x - p1.x, image_center_x - p2.x};
	w[2] = std::abs(w[0]) + std::abs(w[1]);
	w[0] /= w[2];
	w[1] /= w[2];
	double dy = std::abs(p1.y - p2.y) / 2;
	double dx[2] = {dy * w[0], dy * w[1]}; // the sign of w[] correponds to the x-location of the point relatively to the center. 
	double my = (p1.y + p2.y) / 2;
	if(std::abs(my - image_center_y) < std::abs(p1.y - image_center_y)) { // toward the center
		dx[0] *= -1.0;
	}
	if(std::abs(my - image_center_y) < std::abs(p2.y - image_center_y)) { // toward the center
		dx[1] *= -1.0;
	}
	const size_t NP = 2/*9*//*10*//*5*//*1*/;
	Point3d points[NP][2] = { {Point3d(p1.x, p1.y, 1), Point3d(p2.x, p2.y, 1)}, {Point3d(p1.x - dx[0], my, 1), Point3d(p2.x - dx[1], my, 1)}
	};
	double sum_reproject_weights = 0;
	int min_index = 0;
	for(size_t i = 0, np = p1._isACorner? 1 : NP; i < np; ++i) {
		Point3d u[2] = {points[i][0], points[i][1]};
		Mat_<double> val[2];
		double reproject_err[2] = {10, 10};
		for(size_t q = 0; q < 2; ++q) {
			cv::Mat cam0pnts(1, 1, CV_64FC2, Scalar(u[0].x, u[0].y));
			cv::Mat cam1pnts(1, 1, CV_64FC2, Scalar(u[1].x, u[1].y));

			cv::Mat cam0pnts_corrected;
			cv::Mat cam1pnts_corrected;

			if(q == 0) {
				cam0pnts_corrected = cam0pnts;
				cam1pnts_corrected = cam1pnts;
			}
			else {
				correctMatches(F, cam0pnts, cam1pnts, cam0pnts_corrected, cam1pnts_corrected);
			}

			Point3d U[3];
			U[0].x = ((Scalar*)(cam0pnts_corrected.data))->val[0];
			U[0].y = ((Scalar*)(cam0pnts_corrected.data))->val[1];
			U[0].z = 1;
			U[1].x = ((Scalar*)(cam1pnts_corrected.data))->val[0];
			U[1].y = ((Scalar*)(cam1pnts_corrected.data))->val[1];
			U[1].z = 1;

			val[q] = IterativeLinearLSTriangulation(U[0], Pl, U[1], Pr);
			reproject_err[q] = Eval_ReprojectError(/*Point2d(u[0].x, u[0].y), Point2d(u[1].x, u[1].y), */p1, p2, Pl, Pr, val[q]);
			if(reproject_err[q] < 0.1) {
				break;
			}
		}

		size_t idx = (reproject_err[0] < reproject_err[1])? 0: 1;

		if(NP == 1) {
			X = val[idx];
			sum_reproject_weights = 1;
		}
		else {
			if(reproject_err[idx] < 0.1) {
				reproject_err[idx] = 0.1;
			}
			double reproject_weight = 1.0 / pow(reproject_err[idx], 2.0);
			sum_reproject_weights += reproject_weight;

			X += val[idx] * reproject_weight;
		}
	}

	X *= 1.0 / sum_reproject_weights;

	return Eval_ReprojectError(p1, p2, Pl, Pr, X);
}




struct ComparePointsByClusterNumber {
	bool operator()(const ClusteredPoint& left, const ClusteredPoint& right) const {
		if (left._cluster == right._cluster)
			return left.x < right.x;
		if (left._cluster == -1 && right._cluster > -1)
			return false;
		if (left._cluster > -1 && right._cluster == -1)
			return true;
		return left._cluster < right._cluster;
	}
};

void reconstruct4DPoints(
	std::vector<ClusteredPoint> points2D[2] /*in/out*/, // gets sorted and modified. The _cluster is set to the index of epipolar cluster. 
	Mat& F, /*in*/
	Mat& camMatl, /*in*/
	Mat& camMatr, /*in*/
	Mat& Pl, /*in*/
	Mat& Pr, /*in*/
	std::vector<ReconstructedPoint>& points4D /*out*/ // the _id member of each point links to the point in cv_point array. 
	) {

	std::for_each(points2D[0].begin(), points2D[0].end(), Set_CameraNumber(-1));
	std::for_each(points2D[1].begin(), points2D[1].end(), Set_CameraNumber(+1));

	std::vector<std::vector<ClusteredPoint*>> clusters; // is built on epipolar lines (here it is a strip of configurable width). cluster must have even number of points; otherwise do mark all points in the cluster as not clustered. 

	PartitionPoints(points2D, clusters);

	for(auto& cluster : clusters) {
		int camera;
		do {
			camera = 0;
			for(auto point : cluster) { // identify which camera has more images. 
				if(point->_cluster >= 0) {
					camera += point->_camera;
				}
			}
			if(camera != 0) {
				int min_intensity = std::numeric_limits<int>::max();
				ClusteredPoint* aPoint = 0;
				for(auto point : cluster) {
					if(point->_cluster >= 0 && (camera * point->_camera) > 0) {
						if(point->_intensity < min_intensity) {
							min_intensity = point->_intensity;
							aPoint = point;
						}
					}
				}
				if(aPoint) {
					aPoint->_cluster = -1;
				}
			}
		} while(camera != 0);
	}

	std::sort(points2D[0].begin(), points2D[0].end(), ComparePointsByClusterNumber());
	std::sort(points2D[1].begin(), points2D[1].end(), ComparePointsByClusterNumber());

	if(points2D[0].size() > 0) {
		for(size_t j = 0; j < points2D[0].size() && j < points2D[1].size(); ++j) {
			ClusteredPoint& p1 = points2D[0][j];
			ClusteredPoint& p2 = points2D[1][j];
			if(p1._cluster == -1 || p2._cluster == -1) {
				break;
			}

			Mat_<double> X(4, 1, 0.0);

			double min_reproject_err = reconstruct4DPoint(X, p1, p2, F, Pl, Pr);

			points4D.push_back(ReconstructedPoint(Matx41d(X(0), X(1), X(2), 1), (int)j, p1.aType(), min_reproject_err));
		}
	}

	// invert Z-coordinate

	for(auto& point : points4D) {
		point(2) = -point(2);
	}
}


