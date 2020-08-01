// RasterPolygonBuilder.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "stdafx.h"

#include "XConfiguration.h"
#include "XClasses.h"

#include "FrameMain.h"

#include "resource.h"

#include "OleDate.h" 
#include "LoGSeedPoint.h"

//#ifdef _DEBUG
//#include "vld.h"
//#endif


#define STROther_Instance_running 150019 


const Scalar RED(0, 0, 255), GREEN(0, 255, 0), BLUE(0, 0, 255), WHITE(255, 255, 255);
const char ESC_KEY = 27;



const char* g_path_vsconfiguration = ".\\VSWorkConfiguration.txt";
const char* g_path_defaultvsconfiguration = "VSConfiguration.txt";
const char* g_path_calib_images_dir = ".\\nwpu_images\\";

const char* g_path_calibrate_file = ".\\stereo_calibrate.xml";
const char* g_path_calibrate_file_backup = ".\\stereo_calibrate_backup.xml";

const char* g_path_features_file = ".\\stereo_features.xml";
const char* g_path_features_file_backup = ".\\stereo_features_backup.xml";

bool g_bTerminated;
bool g_bUserTerminated;
bool g_bCalibrationExists;
bool g_bRestart;

uint32_t g_actionDeviceKey;
uint32_t g_actionGroupKey;


size_t g_min_images;
double g_aposteriory_minsdistance;
double g_pattern_distance;



StereoConfiguration g_configuration;




BOOL IsChildDir(WIN32_FIND_DATAA* lpFindData) {
	return
		((lpFindData->dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
			(lstrcmp(lpFindData->cFileName, ".") != 0) &&
			(lstrcmp(lpFindData->cFileName, "..") != 0));
}

void Find_SubDirectories(const std::string& parent_dir, std::vector<std::string>& v_sub_dirs) {
	v_sub_dirs.clear();

	WIN32_FIND_DATAA stFindData;
	std::string strFileName = parent_dir + "*";
	HANDLE hFind = FindFirstFileA(strFileName.c_str(), &stFindData);
	BOOL fOk = (hFind != INVALID_HANDLE_VALUE);
	while (!g_bTerminated && fOk) {
		if (IsChildDir(&stFindData)) {
			v_sub_dirs.push_back(stFindData.cFileName);
		}

		fOk = FindNextFileA(hFind, &stFindData);
	}

	if (hFind != INVALID_HANDLE_VALUE) {
		FindClose(hFind);
	}
}






bool VS_SaveConfiguration(StereoConfiguration& configuration) {
	bool ok = false;
	std::string config_str;
	XSerialize(config_str, configuration, true);
	ok = WriteTextFile(g_path_vsconfiguration, config_str) != 0;
	if (!ok) {
		ok = WriteTextFile(g_path_defaultvsconfiguration, config_str) != 0;
	}
	return ok;
}

bool VS_ReadConfiguration(StereoConfiguration& configuration) {
	bool ok = false;
	std::string& config = ReadTextFile(g_path_vsconfiguration);
	if (config.size() == 0) {
		if (strlen(g_path_defaultvsconfiguration)) {
			config = ReadTextFile(g_path_defaultvsconfiguration);
		}
		if (config.size()) {
			WriteTextFile(g_path_vsconfiguration, config);
		}
	}
	if (config.size() != 0) {
		XConfigure(configuration, config);
		bool doSave = false;
		if (configuration._image_height <= 0) {
			configuration._image_height = 483;
		}
		if (configuration._version_number <= 0) {
			configuration._version_number = 1;
		}
		if (configuration._version_number < 2) {
			configuration._version_number = 2;
		}
		if (configuration._version_number < 3) {
			configuration._version_number = 3;
			configuration._percent_maxintensity = 0.1333;
			doSave = true;
		}
		if (configuration._version_number < 4) {
			configuration._version_number = 4;
			configuration._calib_auto_image_capture = 1;
			doSave = true;
		}
		if (configuration._version_number < 5) {
			configuration._version_number = 5;
			configuration._max_boxsize_pixels = 25;
			doSave = true;
		}
		if (configuration._version_number < 6) {
			configuration._version_number = 6;
			doSave = true;
		}
		if (configuration._version_number < 7) {
			configuration._version_number = 7;
			configuration._calib_rectify_alpha_param = 1;
			doSave = true;
		}
		if (configuration._version_number < 8) {
			configuration._version_number = 8;
			configuration._calib_rectify_alpha_param = 0;
			doSave = true;
		}
		if (configuration._version_number < 9) {
			configuration._version_number = 9;
			configuration._calib_min_confidence = 0.4;
			doSave = true;
		}
		if (configuration._version_number < 10) {
			configuration._version_number = 10;
			configuration._distance_to_target = 4;
			configuration._scalefactor2distance_to_target = 0;
			doSave = true;
		}
		if (configuration._version_number < 11) {
			configuration._version_number = 11;
			configuration._visual_diagnostics = 1;
			doSave = true;
		}
		if (doSave) {
			VS_SaveConfiguration(configuration);
		}
		ok = true;
	}
	return ok;
}




bool ProcessWinMessages(DWORD dwMilliseconds) {
	bool rc = true;
	MSG msg;
	try {
		if (PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE)) {
			while (PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE)/* == TRUE*/) {
				if (GetMessage(&msg, NULL, 0, 0)) {
					TranslateMessage(&msg);
					DispatchMessage(&msg);
				}
				else {
					g_bTerminated = true;
				}
			}
		}
		else {
			if (dwMilliseconds) {
				Sleep(dwMilliseconds);
			}
			rc = false;
		}
	}
	catch (...) {
		rc = false;
	}
	return rc;
}

void AcceptNewGlobalConfiguration(StereoConfiguration& configuration/*out - local config*/,
	SImageAcquisitionCtl& image_acquisition_ctl, SPointsReconstructionCtl* reconstruction_ctl, vs_callback_launch_workerthreads* launch_workerthreads) {
	if (g_bTerminated) {
		return;
	}

	StereoConfiguration new_configuration;
	new_configuration << g_configuration;

	bool do_restart_cameras = false;
	bool do_resend_cameras_params = false;

	if (configuration._is_active) {
		size_t j = 0;
		for (auto& name : new_configuration._camera_names) {
			if (j < configuration._camera_names.size()) {
				if (name != configuration._camera_names[j++]) {
					do_restart_cameras = true;
				}
			}
		}
		if (configuration._image_height != new_configuration._image_height) {
			do_restart_cameras = true;
		}

		if (configuration._trigger_source_hardware != new_configuration._trigger_source_hardware) {
			do_resend_cameras_params = true;
		}
	}

	if (do_restart_cameras) {
		image_acquisition_ctl._terminated = 1;
		while (image_acquisition_ctl._status != 0) {
			ProcessWinMessages(10);
		}
		if (image_acquisition_ctl._imagepoints_status > -1) {
			while (image_acquisition_ctl._imagepoints_status != 0) {
				ProcessWinMessages(10);
			}
		}

		if (reconstruction_ctl) {
			reconstruction_ctl->_terminated = 1;
			while (reconstruction_ctl->_status != 0) {
				ProcessWinMessages(10);
			}
		}
	}

	if (new_configuration._calib_rectify_alpha_param > 1) {
		new_configuration._calib_rectify_alpha_param = 1;
	}
	if (new_configuration._calib_rectify_alpha_param < -1) {
		new_configuration._calib_rectify_alpha_param = -1;
	}

	if (new_configuration._calib_min_confidence > 1) {
		new_configuration._calib_min_confidence = 1;
	}
	if (new_configuration._calib_min_confidence < 0) {
		new_configuration._calib_min_confidence = 0;
	}

	configuration = new_configuration;

	g_pattern_distance = configuration._pattern_distance > 0 ? configuration._pattern_distance : 2.5;
	g_max_clusterdistance = configuration._max_clusterdistance > 0 ? configuration._max_clusterdistance : 4;
	g_axises_ratio = configuration._axises_ratio > 0 ? configuration._axises_ratio : 7.0 / 9.0;
	g_percent_maxintensity = configuration._percent_maxintensity > 0 ? configuration._percent_maxintensity : 1.0 / 3.0;

	g_max_boxsize_pixels = configuration._max_boxsize_pixels > 0 ? configuration._max_boxsize_pixels : 25;

	g_max_Y_error = configuration._max_Y_error > 0 ? configuration._max_Y_error : 4.0;

	g_min_images = configuration._min_images > 0 ? configuration._min_images : 12;
	g_aposteriory_minsdistance = configuration._min_RMSE_images > 0 ? configuration._min_RMSE_images : 100;

	if (configuration._image_height <= 0) {
		configuration._image_height = 483;
	}
	image_acquisition_ctl._image_height = configuration._image_height;

	image_acquisition_ctl._trigger_source_software = configuration._trigger_source_hardware == 0;

	image_acquisition_ctl._images_from_files = configuration._images_from_files != 0;
	image_acquisition_ctl._save_all_calibration_images = configuration._save_all_calibration_images != 0;

	image_acquisition_ctl._pattern_is_whiteOnBlack = configuration._pattern_is_whiteOnBlack != 0;
	image_acquisition_ctl._pattern_is_gridOfSquares = configuration._pattern_is_gridOfSquares != 0;
	image_acquisition_ctl._pattern_is_chessBoard = configuration._pattern_is_chessBoard != 0;

	while (configuration._scene_translate.size() < 4) configuration._scene_translate.push_back(0);
	while (configuration._localscene_translate.size() < 4) configuration._localscene_translate.push_back(0);

	if (reconstruction_ctl) {
		reconstruction_ctl->_pixel_threshold = configuration._pixel_threshold;
	}

	configuration._is_active = true;
}


return_t __stdcall ZeroLogHandler(LPVOID lp) {
	try {
		IPCSetLogHandler((HWND)0);
	}
	catch (...) {
	}
	return 0;
}

HANDLE g_event_SFrameIsAvailable = INVALID_HANDLE_VALUE;
HANDLE g_event_SeedPointIsAvailable = INVALID_HANDLE_VALUE;
HANDLE g_event_ContourIsConfirmed = INVALID_HANDLE_VALUE;
LoGSeedPoint g_LoG_seedPoint;

void OnMouseCallback(int event, int x, int y, int flags, void* userdata) {
	MouseCallbackParameters* params = (MouseCallbackParameters*)userdata;
	if (event == 0) {
		return;
	}
	if (flags == 1) {
		return;
	}
	/*
	event - 1: left button
	event - 2: right button
	event - 3: central button
	*/
	/*
	std::cout << "flags" << flags << std::endl;
	*/

	g_LoG_seedPoint.params = *params;
	g_LoG_seedPoint.x = x;
	g_LoG_seedPoint.y = y;

	switch (params->windowNumber) {
	case 1: // window 1
	case 2: // window 2
		SetEvent(g_event_SeedPointIsAvailable);
		break;
	case 3:
	case 4:
		SetEvent(g_event_ContourIsConfirmed);
		break;
	default:
		break;
	}

}



bool DisplayReconstructionData(SPointsReconstructionCtl& reconstruction_ctl, SImageAcquisitionCtl& image_acquisition_ctl, std::string imagewin_names[4], int& time_average) {
	bool stereodata_statistics_changed = false;

	bool data_isok = reconstruction_ctl._data_isvalid;
	bool image_isok = reconstruction_ctl._image_isvalid;

	__int64 time_start = OSDayTimeInMilliseconds();

	static Mat cv_image[2]; // gets destroyed each time around
	static Mat cv_edges[2];
	static Mat cv_background[4];

	int nrows = 0;
	int ncols = 0;

	static std::vector<ABox> boxes[2];
	static std::vector<ClusteredPoint> cv_points[2];

	static std::vector<int> labels; // labels for points4D; a label represents a cluster of points.It is based on distance, i.e.has to be closer than threshold distance.
	static int coordsystem_label = 0; // it is the label of the cluster that represents the coordsystem. 
	static std::vector<ReconstructedPoint> points4D; // the _id member of each point links to the point in cv_point array. 
	static std::vector<std::vector<ReconstructedPoint>> coordlines4D;
	static std::vector<ReconstructedPoint> points4Dtransformed;
	static std::vector<std::vector<ReconstructedPoint>> coordlines4Dtransformed;

	static wsi_gate gate;

	cv::Rect roi[2];

	double fx = 1;
	double fy = 1;

	static int s_windows_painted = 0;
	static Mat s_windows_default_image;

	static bool onMouse_isActive[4] = { 0, 0, 0, 0 };
	static MouseCallbackParameters mouse_callbackParams[4];


	gate.lock();

	if (data_isok || image_isok) { // retrieve data
		reconstruction_ctl._gate.lock();

		data_isok = reconstruction_ctl._data_isvalid;
		image_isok = reconstruction_ctl._image_isvalid;

		roi[0] = reconstruction_ctl._roi[0];
		roi[1] = reconstruction_ctl._roi[1];

		if (data_isok) {
			boxes[0] = (reconstruction_ctl._boxes[0]);
			boxes[1] = (reconstruction_ctl._boxes[1]);
			cv_points[0] = (reconstruction_ctl._cv_points[0]);
			cv_points[1] = (reconstruction_ctl._cv_points[1]);

			labels = (reconstruction_ctl._labels);
			coordsystem_label = reconstruction_ctl._coordsystem_label; // it is the label of the cluster that represents the coordsystem. 
			points4D = (reconstruction_ctl._points4D); // the _id member of each point links to the point in cv_point array. 
			coordlines4D = (reconstruction_ctl._coordlines4D);
			points4Dtransformed = (reconstruction_ctl._points4Dtransformed);
			coordlines4Dtransformed = (reconstruction_ctl._coordlines4Dtransformed);

			reconstruction_ctl._data_isvalid = false;
		}

		if (image_isok) {
			matCV_16UC1_memcpy(cv_image[0], reconstruction_ctl._cv_image[0]);
			matCV_16UC1_memcpy(cv_image[1], reconstruction_ctl._cv_image[1]);
			matCV_16UC1_memcpy(cv_edges[0], reconstruction_ctl._cv_edges[0]);
			matCV_16UC1_memcpy(cv_edges[1], reconstruction_ctl._cv_edges[1]);

			nrows = cv_edges[0].rows;
			ncols = cv_edges[0].cols;

			reconstruction_ctl._image_isvalid = false;
		}

		stereodata_statistics_changed = true;

		reconstruction_ctl._gate.unlock();
	}



	/*Actual display starts here*/

	if (g_configuration._visual_diagnostics) {
		if (image_isok) {
			int fs = std::max(256 / (int)g_bytedepth_scalefactor, 1);
			if (fs != 1) {
				cv_edges[0] *= fs; // Mar.4 2015.
				cv_edges[1] *= fs; // Mar.4 2015.
			}

			cvtColor(cv_edges[0], cv_image[0] = Mat(), CV_GRAY2RGB);
			cvtColor(cv_edges[1], cv_image[1] = Mat(), CV_GRAY2RGB);

			fx = 700.0 / cv_image[0].cols;
			fy = 400.0 / cv_image[0].rows;
			//fy = fx; 

			HWND hwnd = (HWND)cvGetWindowHandle(imagewin_names[0].c_str());
			RECT clrect;
			if (GetWindowRect(GetParent(hwnd), &clrect)) {
				fx = (double)(clrect.right - clrect.left) / (double)cv_image[0].cols;
				fy = (double)(clrect.bottom - clrect.top) / (double)cv_image[0].rows;
			}

			cv::resize(cv_image[0], cv_edges[0] = Mat(), cv::Size(0, 0), fx, fy, INTER_AREA);
			cv::resize(cv_image[1], cv_edges[1] = Mat(), cv::Size(0, 0), fx, fy, INTER_AREA);

			cv_image[0] = cv_edges[0];
			cv_image[1] = cv_edges[1];
		}


		//if(image_isok && g_background_image[0].rows > 0) {
		//	g_background_image_lock.lock();
		//	for(int j = 0; j < 2; ++j) {
		//		if(g_background_image[j].rows > 0) {
		//			matCV_16UC1_memcpy(cv_background[j], g_background_image[j]);
		//			resize(cv_background[j], cv_background[j + 2] = Mat(), cv::Size(0, 0), fx, fy, INTER_AREA);
		//		}
		//	}
		//	g_background_image_lock.unlock();
		//}


		const int cdf = 256; // color depth factor. Mar.4 2015. 

		if (data_isok && image_isok) { // draw rectified images with detected objects and epipolar lines.
			for (int j = 0; j < 2; ++j) {
				if (roi[j].height > 0 && roi[j].width > 0) {
					rectangle(cv_image[j], Point((int)(roi[j].x * fx), (int)(roi[j].y * fx)), Point((int)((roi[j].x + roi[j].width) * fx), (int)((roi[j].y + roi[j].height) * fx)), Scalar(0, 0, 255 * cdf));
				}

				for (auto& box : boxes[j]) {
					if ((box.x[0] > 0 || box.x[1] < cv_edges[j].cols) && (box.y[0] > 0 || box.y[1] < cv_edges[j].rows))
						rectangle(cv_image[j], Point((int)(box.x[0] * fx), (int)(box.y[0] * fy)), Point((int)(box.x[1] * fx), (int)(box.y[1] * fy)), Scalar(0, 0, 255 * cdf));
				}

				if (!g_configuration._supervised_LoG) {
					for (int y = 0; y < cv_image[j].rows; y += 15) {
						line(cv_image[j], Point(0, y), Point(cv_image[j].cols, y), Scalar(0, 255 * cdf, 0));
					}
				}

				if (imagewin_names[j].size() > 0) {
					cv::imshow(imagewin_names[j], cv_image[j]);
				}

				if (cv_background[j].rows > 0) {
					cv::imshow(imagewin_names[j], cv_background[j + 2]);
				}
			}
		}

		if (data_isok && image_isok) { // draw detected shapes. 
			for (int j = 0; j < 2; ++j) {
				std::vector<ClusteredPoint*> sorted_bycircularity;
				sorted_bycircularity.reserve(cv_points[j].size());

				int crop_center_line = 0;
				int crop_maxrows = 0;
				int crop_maxcols = 0;
				int crop_center_maxoffset = 0;
				int crop_center_x = 0;

				for (auto& point : cv_points[j]) {
					if (point._cluster != -1) {
						point.x = std::floor(point.x * fx + 0.5);
						point.y = std::floor((nrows - point.y) * fy + 0.5);
					}

					//point._crop_center.x += range_rand(10);

					sorted_bycircularity.push_back(&point);
					if (point._crop_center.y > crop_center_line) {
						crop_center_line = (int)(point._crop_center.y + 0.5);
					}
					if (point._crop.rows > crop_maxrows) {
						crop_maxrows = point._crop.rows;
					}
					if (point._crop.cols > crop_maxcols) {
						crop_maxcols = point._crop.cols;
					}
					int point_center_offset = std::abs((int)(point._crop_center.y + 0.5) - (point._crop.rows >> 1) + 1);
					if (point_center_offset > crop_center_maxoffset) {
						crop_center_maxoffset = point_center_offset;
					}
					if (point._crop_center.x > crop_center_x) {
						crop_center_x = (int)(point._crop_center.x + 0.5);
					}

					for (int j = ((int)sorted_bycircularity.size() - 1); j > 0; --j) {
						if (sorted_bycircularity[j]->_display_weight_factor < sorted_bycircularity[j - 1]->_display_weight_factor) {
							if (sorted_bycircularity[j]->_cluster < 0 || (sorted_bycircularity[j]->_cluster >= 0 && sorted_bycircularity[j - 1]->_cluster >= 0)) {
								break;
							}
						}
						if (sorted_bycircularity[j]->_cluster < 0 && sorted_bycircularity[j - 1]->_cluster >= 0) {
							break;
						}
						std::swap(sorted_bycircularity[j], sorted_bycircularity[j - 1]);
					}
				}

				if (sorted_bycircularity.size()) {
					static int crop_center_line_avg[2] = { 0, 0 };
					static int crop_rows_max[2] = { 0, 0 };
					static int crop_max_center_x[2] = { 0, 0 };
					if (crop_center_line_avg[j] == 0) {
						crop_center_line_avg[j] = crop_center_line;
					}
					else {
						crop_center_line_avg[j] = (int)floor((crop_center_line_avg[j] * 100.0 + crop_center_line) / 101.0 + 0.5);
					}
					crop_maxrows += (crop_center_maxoffset << 1) + (std::abs(crop_center_line_avg[j] - crop_center_line) << 1);
					if (crop_rows_max[j] < crop_maxrows) {
						crop_rows_max[j] = crop_maxrows;
					}
					else {
						crop_rows_max[j] = (int)floor((crop_rows_max[j] * 100.0 + crop_maxrows) / 101.0 + 0.5);
					}
					if (crop_max_center_x[j] < crop_center_x) {
						crop_max_center_x[j] = crop_center_x;
					}
					else {
						crop_max_center_x[j] = (int)floor((crop_max_center_x[j] * 100.0 + crop_center_x) / 101.0 + 0.5);
					}

					Mat crop;
					int image_count = 0;
					int i = 0;
					while (image_count < 5 && i < sorted_bycircularity.size()) {
						if (sorted_bycircularity[i]->_crop.rows > 0) {
							Mat& newcrop = sorted_bycircularity[i]->_crop;
							int crop_x_offset = crop_max_center_x[j] - (int)(sorted_bycircularity[i]->_crop_center.x + 0.5);
							Size newsize(crop.cols + newcrop.cols + crop_x_offset, crop_rows_max[j]);
							Mat out = cv::Mat::zeros(newsize, newcrop.type());
							if (crop.cols) {
								crop.copyTo(out(cv::Rect(0, 0, crop.cols, crop.rows)));
							}
							int crop_offset = crop_center_line_avg[j] - (int)(sorted_bycircularity[i]->_crop_center.y + 0.5);
							if (crop_offset < 0) {
								crop_offset = 0;
							}
							newcrop.copyTo(out(cv::Rect(crop.cols + crop_x_offset, crop_offset, newcrop.cols, newcrop.rows)));
							out.copyTo(crop = Mat());
							if (++image_count < 3) {
								circle(cv_image[j], Point(*sorted_bycircularity[i]), 3, Scalar(0, 255 * cdf, 0), -1);
							}
							newcrop = Mat();
						}
						++i;
					}
					if (image_count > 0 && imagewin_names[j + 2].size() > 0) {
						int q = j + 2;
						HWND hwnd = (HWND)cvGetWindowHandle(imagewin_names[q].c_str());
						RECT clrect;
						if (GetWindowRect(GetParent(hwnd), &clrect)) {
							Mat crop_scaled;
							Size dsize = Size(clrect.right - clrect.left, clrect.bottom - clrect.top);
							cv::resize(crop, crop_scaled, dsize, 0, 0, INTER_AREA);
							crop = crop_scaled.clone();
							crop_scaled = Mat();
						}
						else {
							bool err = true;
						}
						cv::imshow(imagewin_names[q], crop);
						if (!onMouse_isActive[q]) {
							mouse_callbackParams[q].scaleFactors.fx = fx;
							mouse_callbackParams[q].scaleFactors.fy = fy;
							mouse_callbackParams[q].windowNumber = q + 1;
							setMouseCallback(imagewin_names[q], OnMouseCallback, (void*)&mouse_callbackParams[q]);
							onMouse_isActive[q] = true;
						}
					}
				}
			}
		}

		if (!data_isok && image_isok) {
			for (int j = 0; j < 2; ++j) {
				if (imagewin_names[j].size() > 0) {
					if (roi[j].height > 0 && roi[j].width > 0) {
						rectangle(cv_image[j], Point((int)(roi[j].x * fx), (int)(roi[j].y * fx)), Point((int)((roi[j].x + roi[j].width) * fx), (int)((roi[j].y + roi[j].height) * fx)), Scalar(0, 0, 255 * 256));
					}
					cv::imshow(imagewin_names[j], cv_image[j]);
				}
				if (!onMouse_isActive[j]) {
					mouse_callbackParams[j].scaleFactors.fx = fx;
					mouse_callbackParams[j].scaleFactors.fy = fy;
					mouse_callbackParams[j].windowNumber = j + 1;
					setMouseCallback(imagewin_names[j], OnMouseCallback, (void*)&mouse_callbackParams[j]);
					onMouse_isActive[j] = true;
				}
			}
		}

		s_windows_painted = 1;
	}



	if (!g_configuration._visual_diagnostics && s_windows_painted) {
		for (int j = 0; j < 4; ++j) {
			if (imagewin_names[j].size() > 0) {
				if (s_windows_default_image.rows == 0) {
					s_windows_default_image = cv::imread(IMG_DELETEDOCUMENT_H, CV_LOAD_IMAGE_COLOR);
				}
				if (s_windows_default_image.rows != 0) {
					cv::imshow(imagewin_names[j], s_windows_default_image);
				}

				//HWND hwnd = (HWND)cvGetWindowHandle(imagewin_names[j].c_str());
				//InvalidateRect(hwnd, NULL, TRUE);

				//RECT clrect;
				//if(GetWindowRect(hwnd, &clrect)) {
				//	HWND hparent = GetParent(hwnd);
				//	POINT actual_origin = {clrect.left, clrect.top};
				//	ScreenToClient(hparent, &actual_origin);

				//	MoveWindow(hwnd, actual_origin.x, actual_origin.y, clrect.right + 1 - clrect.left, clrect.bottom + 1 - clrect.top, TRUE);
				//	MoveWindow(hwnd, actual_origin.x, actual_origin.y, clrect.right - clrect.left, clrect.bottom - clrect.top, TRUE);
				//}
			}
		}

		_g_main_frame->Invalidate();

		s_windows_painted = 0;
	}




	if (data_isok || image_isok) {
		time_average = (time_average * 20 + (int)(OSDayTimeInMilliseconds() - time_start)) / 21;
		stereodata_statistics_changed = true;
	}


	gate.unlock();

	for (auto& boxes : boxes) boxes.clear();
	for (auto& cv_points : cv_points) cv_points.clear();
	labels.clear();
	points4D.clear();
	coordlines4D.clear();
	points4Dtransformed.clear();
	coordlines4Dtransformed.clear();

	return stereodata_statistics_changed;
}




void show_image(Mat& image, const std::string& window_name) {
	HWND hwnd = (HWND)cvGetWindowHandle(window_name.c_str());
	RECT clrect;
	if (GetWindowRect(GetParent(hwnd), &clrect)) {
		double fx = (double)(clrect.right - clrect.left) / (double)image.cols;
		double fy = (double)(clrect.bottom - clrect.top) / (double)image.rows;
		Mat image2;
		cv::resize(image, image2 = Mat(), cv::Size(0, 0), fx, fy, INTER_AREA);
		cv::imshow(window_name, image2);
	}
	while (ProcessWinMessages());
}





int main() {
	srand((unsigned)time(NULL));
	std::mt19937 rand_gen(070764);

	SImageAcquisitionCtl image_acquisition_ctl;
	SPointsReconstructionCtl reconstruction_ctl;

	StereoConfiguration configuration;

	int number_of_threads = cv::getNumberOfCPUs() / 2;
	if (number_of_threads == 0) {
		number_of_threads = 1;
	}
	cv::setNumThreads(number_of_threads);

	std::string config;
	if (VS_ReadConfiguration(g_configuration)) {
		AcceptNewGlobalConfiguration(configuration/*out - local config*/, image_acquisition_ctl, &reconstruction_ctl);
	}


	HINSTANCE hInstance = GetModuleHandle(NULL);

	HICON hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON1));

	FrameMain* mywin = new FrameMain(hInstance, hIcon);
	if (mywin->OtherInstanceRunning()) {
		Sleep(4000);
		if (mywin->OtherInstanceRunning()) {
			std::string str;
			GetStringentity(STROther_Instance_running, str);
			MessageBox(0, str.c_str(), "Information", MB_OK);
			delete mywin;
			return 0;
		}
	}

	_g_main_frame = mywin;

	_g_main_frame->Instantiate();

	while (ProcessWinMessages());

	g_event_SFrameIsAvailable = CreateEvent(0, 0, 0, 0);
	g_event_SeedPointIsAvailable = CreateEvent(0, 0, 0, 0);
	g_event_ContourIsConfirmed = CreateEvent(0, 0, 0, 0);

	if (_g_images_frame) {
		_g_images_frame->NEW_StereoConfiguration(g_configuration);
	}


	int time_average = 0;

	//VLDEnable();
	//ReconstructPoints(&reconstruction_ctl);
	//return 0;

	if (!g_bTerminated) {
		launch_reconstruction(image_acquisition_ctl, &reconstruction_ctl);

		if (_g_images_frame) {
			_g_images_frame->NEW_StereoConfiguration(g_configuration);
		}

		std::string imagewin_names[4];
		rootCVWindows(_g_images_frame, ARRAY_NUM_ELEMENTS(imagewin_names), 1, imagewin_names);

		while (!g_bTerminated) {
			if (g_configuration._changes_pending) {
				AcceptNewGlobalConfiguration(configuration/*out - local config*/, image_acquisition_ctl, &reconstruction_ctl, launch_reconstruction);
				if (!g_bTerminated) {
					VS_SaveConfiguration(configuration);
				}
				if (!g_bTerminated && !g_bCalibrationExists) {
					g_bRestart = true;
				}
				if (!g_bCalibrationExists) {
					g_bTerminated = true;
				}
				continue;
			}

			WaitForSingleObject(g_event_SFrameIsAvailable, 10);

			bool stereodata_statistics_changed = DisplayReconstructionData(reconstruction_ctl, image_acquisition_ctl, imagewin_names, time_average);
			if (stereodata_statistics_changed) {
				if (_g_images_frame) {
				}
			}


			if (ProcessWinMessages()) {
			}
		}
	}

	QueueWorkItem(ZeroLogHandler);

	image_acquisition_ctl._terminated = 1;
	while (image_acquisition_ctl._status != 0) {
		while (ProcessWinMessages(10));
	}
	image_acquisition_ctl._terminated = 0;

	reconstruction_ctl._terminated = 1;
	while (reconstruction_ctl._status != 0) {
		while (ProcessWinMessages(10));
	}
	reconstruction_ctl._terminated = 0;


	while (ProcessWinMessages());

	g_bTerminated = true;

	do {
	} while (ProcessWinMessages(50));

	SignalWorkItems(9);
	RestoreStandardOutput();

	Sleep(500);
	std::cout << "quit" << std::endl;
	WaitforWorkItems();

	if (!g_bUserTerminated) {
		g_bRestart = true;
	}

	if (g_bRestart) {
		STARTUPINFO si;
		memset(&si, 0, sizeof(si));
		si.cb = sizeof(si);

		PROCESS_INFORMATION pi;
		memset(&pi, 0, sizeof(pi));

		char myName[MAX_PATH + 16];
		GetModuleFileNameA(GetModuleHandle(0), myName, _MAX_PATH);
		CreateProcessA(myName, NULL, NULL, NULL, FALSE, NULL, NULL, NULL, &si, &pi);
		if (pi.hProcess != NULL) {
			CloseHandle(pi.hProcess);
		}
	}

	//_CrtDumpMemoryLeaks();

	TerminateProcess(GetCurrentProcess(), 0);

	return 0;
}



int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
	return main();
}



VSPrincipalDirection_FitLine* VSFitLine(VSPrincipalDirection_FitLine* obj) {
	obj->_ok = 0;
	if (obj->_input_points.size() < 2) {
		return obj;
	}

	Mat_<float> Swarm((int)obj->_input_points.size(), 3);
	int j = 0;
	for (auto& point : obj->_input_points) {
		for (int i = 0; i < 3; ++i) {
			Swarm(j, i) = (float)point._xyz[i];
		}
		++j;
	}
	obj->_input_points.resize(0);

	Vec6f aLine;
	fitLine(Swarm, aLine, CV_DIST_L2, 0, 0.001, 0.001);

	for (int i = 0; i < 3; ++i) {
		obj->_output_direction._xyz[i] = (aLine[i]);
	}

	obj->_ok = 1;

	return obj;
}



