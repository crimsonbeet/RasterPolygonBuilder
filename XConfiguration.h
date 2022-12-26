
#ifndef StereoVision_wsiclassesH
#define StereoVision_wsiclassesH




#ifdef WSI_NAMESPACE
#undef WSI_NAMESPACE
#endif
#ifdef autocreateserialization_not_necessaryH
#undef autocreateserialization_not_necessaryH
#endif

#define WSI_NAMESPACE XStereo



#include "WSIClassFactory.h"



struct StereoConfiguration { 
	double _pattern_distance; 
	double _max_clusterdistance; 
	double _axises_ratio; 
	int _min_images; 
	double _min_RMSE_images; 
	double _max_Y_error;

	int _pixel_threshold; // is replaced with Otsu optimal threshold

	int _trigger_source_hardware; 

	std::vector<double> _scene_translate; // (not used) is dinamically calculated 
	std::vector<double> _localscene_translate; // (not used) is dinamically calculated 

	std::vector<std::string> _camera_names; 

	int _image_height; 

	int _12bit_format; 

	double _percent_maxintensity; // is used only in BlobDetector() that implements center of gravity (not used)

	int _calib_auto_image_capture;
	int _two_step_calibration;
	int _calib_images_from_files; // is used for re-calibration from previously captured images in "Calibimages" folder. 
	int _save_all_calibration_images;

	double _calib_min_confidence; 
	double _calib_rectify_alpha_param; // alpha=0 means that the rectified images are zoomed and shifted so that only valid pixels are visible(no black areas after rectification).alpha = 1 means that the rectified image is decimated and shifted so that all the pixels from the original images from the cameras are retained in the rectified images 
	int _calib_use_homography; 

	int _pattern_is_chessBoard; // assymetric 7 rows by 8 columns; each row has 4 white squares; 28 (4 by 7) centers, 42 corners ((4*2 - 1) by (7 - 1))
	int _pattern_is_gridOfSquares;
	int _pattern_is_whiteOnBlack;

	int _max_boxsize_pixels; 

	int _version_number; 

	int _file_log; 

	int _use_ellipse_fit; 

	double _distance_to_target;
	double _scalefactor2distance_to_target;
	int _pca_changeOfSign; // tracks the fact of change of sign of computed principal direction. 

	Mat_<double> _principalDirection; // tracks principal direction; it is a work variable 
	int64_t _principalDirection_expiration_time; // defines time limit of validity of _principalDirection; it is a work variable 
	double _principalDirection_reprojerr;

	int _use_center_of_gravity; 

	int _supervised_LoG = 1; 

	int _frames_acquisition_mode; // is used to input from files the frames for reconstruction; value of >1 means read the first image and then re-use it in N threads; -N means read from cameras with N threads. 
	int _evaluate_contours = 0;

	int _visual_diagnostics; 

	int _continuous_capture; // on capture request the images are created continuously; each request will create new folder with file names starting with 1.

	int _engine_features_requested; 

	int _stereoimage_capture_requested; 

	// access control variables; should not be copied. 

	bool _changes_pending;

	bool _is_active; 

	StereoConfiguration() {
		_pattern_distance = 0;
		_max_clusterdistance = 0;
		_axises_ratio = 0;
		_min_images = 0;
		_min_RMSE_images = 0; 

		_max_Y_error = 4.0; 

		_trigger_source_hardware = 0;

		_pixel_threshold = 0;

		_image_height = 483; 

		_12bit_format = 0; 

		_percent_maxintensity = 1.0 / 3.0; 

		_calib_auto_image_capture = 1; 
		_two_step_calibration = 1;
		_calib_images_from_files = 0;
		_save_all_calibration_images = 0;

		_max_boxsize_pixels = 25; 

		_version_number = 0; 

		_file_log = 0; 

		_use_ellipse_fit = 0; 

		_calib_min_confidence = 0; 
		_calib_rectify_alpha_param = 0;
		_calib_use_homography = 0; 

		_pattern_is_chessBoard = 0; 
		_pattern_is_gridOfSquares = 1;
		_pattern_is_whiteOnBlack = 1;

		_distance_to_target = 4; 
		_scalefactor2distance_to_target = 0; 
		_pca_changeOfSign = 0;

		_principalDirection_expiration_time = 0; 
		_principalDirection_reprojerr = 0; 

		_use_center_of_gravity = 0; 

		_frames_acquisition_mode = 0;

		_visual_diagnostics = 1; 

		_continuous_capture = 0; 

		_engine_features_requested = 0; 

		_stereoimage_capture_requested = 0; 

		_supervised_LoG = 1; 

		_changes_pending = false; 
		_is_active = false; 
	} 

	StereoConfiguration& operator=(const StereoConfiguration& other) {
		accept(other); 
		return *this;
	}

	StereoConfiguration& operator<<(StereoConfiguration& other) { // do use operator<<() when synchronization is needed. 
		_gate.lock();
		other._gate.lock(); 
		accept(other);
		_changes_pending = true;
		other._changes_pending = false;
		other._gate.unlock();
		_gate.unlock();
		return *this; 
	}

private:
	wsi_gate _gate;
	void accept(const StereoConfiguration& other) {
		_pattern_distance = other._pattern_distance;
		_max_clusterdistance = other._max_clusterdistance;
		_axises_ratio = other._axises_ratio;
		_min_images = other._min_images;
		_min_RMSE_images = other._min_RMSE_images;

		_max_Y_error = other._max_Y_error;

		_pixel_threshold = other._pixel_threshold;

		_trigger_source_hardware = other._trigger_source_hardware;

		_scene_translate = other._scene_translate;
		_localscene_translate = other._localscene_translate;

		_camera_names = other._camera_names;

		_image_height = other._image_height;

		_12bit_format = other._12bit_format;

		_percent_maxintensity = other._percent_maxintensity; 

		_calib_auto_image_capture = other._calib_auto_image_capture;
		_two_step_calibration = other._two_step_calibration;
		_calib_images_from_files = other._calib_images_from_files;
		_save_all_calibration_images = other._save_all_calibration_images; 

		_calib_min_confidence = other._calib_min_confidence; 
		_calib_rectify_alpha_param = other._calib_rectify_alpha_param;
		_calib_use_homography = other._calib_use_homography;

		_pattern_is_chessBoard = other._pattern_is_chessBoard; 
		_pattern_is_gridOfSquares = other._pattern_is_gridOfSquares;
		_pattern_is_whiteOnBlack = other._pattern_is_whiteOnBlack;

		_max_boxsize_pixels = other._max_boxsize_pixels;

		_distance_to_target = other._distance_to_target;
		_scalefactor2distance_to_target = other._scalefactor2distance_to_target; 
		_pca_changeOfSign = other._pca_changeOfSign; 

		_principalDirection = Mat_<double>();
		_principalDirection_expiration_time = 0;
		_principalDirection_reprojerr = 0;

		_use_center_of_gravity = other._use_center_of_gravity; 

		_frames_acquisition_mode = other._frames_acquisition_mode;

		_use_ellipse_fit = other._use_ellipse_fit; 
		_file_log = other._file_log; 

		_visual_diagnostics = other._visual_diagnostics; 

		_continuous_capture = other._continuous_capture; 

		_engine_features_requested = other._engine_features_requested; 

		_version_number = other._version_number; 
	}
};

BEGIN_WSI_SERIALIZATION_OBJECT(StereoConfiguration)
CONTAINS_FLAT_MEMBER(_pattern_distance, PatternDistance)
CONTAINS_FLAT_MEMBER(_max_clusterdistance, MaxClusterDistance)
CONTAINS_FLAT_MEMBER(_axises_ratio, AxisesRatio)
CONTAINS_FLAT_MEMBER(_min_images, MinImages)
CONTAINS_FLAT_MEMBER(_min_RMSE_images, MinRMSEImages)
CONTAINS_FLAT_MEMBER(_max_Y_error, MaxYError)
CONTAINS_FLAT_MEMBER(_camera_names, CameraNames)
CONTAINS_FLAT_MEMBER(_trigger_source_hardware, TriggerHardware)
CONTAINS_FLAT_MEMBER(_pixel_threshold, PixelThreshold) // is replaced with Otsu optimal threshold
CONTAINS_FLAT_MEMBER(_scene_translate, SceneTranslate)
CONTAINS_FLAT_MEMBER(_localscene_translate, LocalSceneTranslate)
CONTAINS_FLAT_MEMBER(_image_height, ImageHeight)
CONTAINS_FLAT_MEMBER(_12bit_format, Format12Bit)
CONTAINS_FLAT_MEMBER(_percent_maxintensity, PercentMaxintensity) // is used only in BlobDetector() that implements center of gravity (not used)
CONTAINS_FLAT_MEMBER(_calib_auto_image_capture, CalibAutoImageCapture)
CONTAINS_FLAT_MEMBER(_two_step_calibration, TwoStepCalibration)
CONTAINS_FLAT_MEMBER(_calib_images_from_files, CalibImagesFromFiles)
CONTAINS_FLAT_MEMBER(_save_all_calibration_images, SaveAll)
CONTAINS_FLAT_MEMBER(_calib_min_confidence, CalibMinConfidence)
CONTAINS_FLAT_MEMBER(_calib_rectify_alpha_param, CalibRectifyAlphaParam)
CONTAINS_FLAT_MEMBER(_calib_use_homography, CalibUseHomography)
CONTAINS_FLAT_MEMBER(_pattern_is_chessBoard, PatternIsChessBoard)
CONTAINS_FLAT_MEMBER(_pattern_is_gridOfSquares, PatternIsGridOfSquares)
CONTAINS_FLAT_MEMBER(_pattern_is_whiteOnBlack, PatternIsWhiteOnBlack)
CONTAINS_FLAT_MEMBER(_max_boxsize_pixels, MaxBoxsizePixels)
CONTAINS_FLAT_MEMBER(_distance_to_target, DistanceToTarget)
CONTAINS_FLAT_MEMBER(_scalefactor2distance_to_target, Scalefactor2DistanceToTarget)
CONTAINS_FLAT_MEMBER(_pca_changeOfSign, PcaChangeOfSign)
CONTAINS_FLAT_MEMBER(_version_number, VersionNumber)
CONTAINS_FLAT_MEMBER(_file_log, FileLog)
CONTAINS_FLAT_MEMBER(_use_ellipse_fit, UseEllipseFit)
CONTAINS_FLAT_MEMBER(_use_center_of_gravity, UseCenterOfGravity)
CONTAINS_FLAT_MEMBER(_supervised_LoG, SupervisedLoG)
CONTAINS_FLAT_MEMBER(_frames_acquisition_mode, FramesAcquisitionMode)
CONTAINS_FLAT_MEMBER(_evaluate_contours, EvaluateContours)
CONTAINS_FLAT_MEMBER(_visual_diagnostics, VisualDiagnostics)
CONTAINS_FLAT_MEMBER(_continuous_capture, ContinuousCapture)
CONTAINS_FLAT_MEMBER(_engine_features_requested, FeaturesRequested)
END_WSI_SERIALIZATION_OBJECT()

AUTOCREATE_WSI_SERIALIZATION_OBJECT(StereoConfiguration)


extern StereoConfiguration g_configuration;






BEGIN_WSI_SERIALIZATION_OBJECT(Point2f)
CONTAINS_FLAT_MEMBER(x, X)
CONTAINS_FLAT_MEMBER(y, Y)
END_WSI_SERIALIZATION_OBJECT()

BEGIN_WSI_SERIALIZATION_OBJECT(Point3d)
CONTAINS_FLAT_MEMBER(x, X)
CONTAINS_FLAT_MEMBER(y, Y)
CONTAINS_FLAT_MEMBER(z, Z)
END_WSI_SERIALIZATION_OBJECT()


BEGIN_WSI_SERIALIZATION_OBJECT(KeyPoint)
CONTAINS_FLAT_MEMBER(size, S)
CONTAINS_OBJECT_MEMBER(pt, P)
END_WSI_SERIALIZATION_OBJECT()







void AcceptNewGlobalConfiguration(StereoConfiguration& configuration/*out - local config*/, SImageAcquisitionCtl& image_acquisition_ctl, SPointsReconstructionCtl *reconstruction_ctl = 0, vs_callback_launch_workerthreads *launch_workerthreads = 0);
bool VS_SaveConfiguration(StereoConfiguration& configuration); 

bool CalibrationFileExists();


void CalibrateCameras(StereoConfiguration& configuration, SImageAcquisitionCtl& image_acquisition_ctl);


#endif //StereoVision_wsiclassesH
