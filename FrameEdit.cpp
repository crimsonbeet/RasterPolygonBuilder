
#include "stdafx.h"
#include "IPCInterface.h" 

#include "FrameEdit.h" 
#include "FrameMain.h" 


FrameEdit *_g_edit_frame; 


const std::string _g_save_images[1] = { 
	IMG_SAVEDOCUMENT 
}; 

const std::string _g_save_images_disabled[1] = { 
	IMG_SAVEDOCUMENT_D 
}; 

const std::string _g_save_images_hot[1] = { 
	IMG_SAVEDOCUMENT_H 
}; 

extern bool g_bCalibrationExists;


#define The_form_can_not_be_saved 100045
#define Some_fields_are_not_valid 100046
#define Save_and_reconnect 100063
#define Save_and_run 100066


FrameEdit::FrameEdit(HINSTANCE hInstance, HWND hParent): KFrame(hInstance), _user_input(0) { 
	Configure(ReadTextFile("FrameEdit.txt")); 
	Create(hInstance, hParent); 
} 

void FrameEdit::ProcessToolBarCommand(SReBar& rebar) { 
	if(rebar._lparam == (LPARAM)_toolbar->_hwndtoolbar) { 
		switch(rebar._wparam) { 
			case 303/*save*/: 
			if(_user_input) { 
				if(_user_input->_pHList->EntryFieldsAreValid()) { 
					_user_input->PostValues(); 
					_g_main_frame->StepBackHistory(); 
				} 
				else { 
					_user_input->_pHList->Invalidate();
					std::string msg; 
					std::string str; 
					if(GetStringentity(The_form_can_not_be_saved, str)) { 
						msg += str; 
						msg += ".\r\n..."; 
					} 
					if(GetStringentity(Some_fields_are_not_valid, str)) { 
						msg += str; 
						msg += "."; 
					} 
					MessageBoxA(_hwnd, msg.c_str(), "Information", MB_OK|MB_ICONINFORMATION); 
				} 
			} 
			break; 
		} 
	} 
} 

void FrameEdit::ProcessToolBarNotify(SReBar& rebar) {
	if(rebar._lparam) { 
		if(((LPNMHDR)rebar._lparam)->hwndFrom == _toolbar->_hwndtoolbar) { 
			switch(((LPNMHDR)rebar._lparam)->code) { 

				case TBN_GETINFOTIPA: { 
					LPNMTBGETINFOTIPA tipinfo = (LPNMTBGETINFOTIPA)rebar._lparam; 
					switch(tipinfo->iItem) { 
						case 303: 
							std::string str; 
							GetStringentity(Save_and_run, str);
							strncpy(tipinfo->pszText, str.c_str(), tipinfo->cchTextMax);
						break;
					} 
				}
				break; 
			} 
		} 
	} 
} 

void FrameEdit::OnChangeFrameState(SReBar& rebar, int state) {
	switch(state) { 
		case 1: /*active*/
			if(!_toolbar) { 
//				rebar._fbandstyle &= ~RBBS_NOGRIPPER; 
				ActivateToolBar(rebar, 303, _g_save_images, _g_save_images_hot, _g_save_images_disabled, ARRAY_NUM_ELEMENTS(_g_save_images)); 
			} 
			else 
			if(_toolbar->_rebarbandid != -1) { 
				SendMessage(rebar._hwndrebar, RB_SHOWBAND, _toolbar->_rebarbandid, 1); 
			} 
			break; 
		default: 
			KFrame::OnChangeFrameState(rebar, state); 
			break;
	} 
} 


BOOL RegExp_CheckNaturalNumber(const std::string& str, int *validNumber = 0);

BOOL RegExp_CheckNaturalNumber(const std::string& str, int *validNumber) {
	std::regex base_regex("[0-9]+");
	std::smatch base_match;

	BOOL ok = FALSE;
	if(str.size() && std::regex_match(str, base_match, base_regex)) { 
		if(base_match.size() == 1) {
			ok = TRUE;
			if(validNumber) {
				*validNumber = atoi(str.c_str());
			}
		}
	}
	
	return ok; 
}

BOOL RegExp_CheckFloatNumber(const std::string& str, double *validNumber = 0);

BOOL RegExp_CheckFloatNumber(const std::string& str, double *validNumber) {
	std::string str_regex(R"(([0-9]*\.[0-9]+)|([0-9]+\.?))"); 
	std::regex base_regex(str_regex);
	std::smatch base_match;

	BOOL ok = FALSE;
	if(str.size() && std::regex_match(str, base_match, base_regex)) {
		if(base_match.size() <= 3) {
			ok = TRUE;
			if(validNumber) {
				*validNumber = atof(str.c_str());
			}
		}
	}

	return ok;
} 

template <typename T> inline
BOOL RegExp_CheckStrings(const std::string& str, std::vector<T>& strarray, size_t N = 0, const std::string& str_regex = R"(^\w+( +(\w+))*$)");

void RegExt_Conv(cv::String& ret, const std::string& str) {
	ret = str; 
}
void RegExt_Conv(int& ret, const std::string& str) {
	ret = atoi(str.c_str());
}
void RegExt_Conv(double& ret, const std::string& str) {
	ret = atof(str.c_str());
}

template <typename T> inline
BOOL RegExp_CheckStrings(const std::string& str, std::vector<T>& strarray, size_t N, const std::string& str_regex) {
	std::regex base_regex(str_regex);
	std::smatch base_match;

	BOOL ok = str.size()? FALSE: TRUE;
	if(str.size() && std::regex_match(str, base_match, base_regex, std::regex_constants::match_any)) {
		ok = TRUE;
		std::string s = str;
		size_t n = 0; 
		static int z = 0; 
		++z; 
		for(size_t j = s.find_first_of(' '); j < s.size(); j = s.find_first_of(' ')) { 
			if(n < N) {
				RegExt_Conv(strarray[n++], s.substr(0, j));
			}
			s = s.substr(j + 1);
		}
		if(n < N && s.size()) {
			RegExt_Conv(strarray[n++], s);
		}
	}

	return ok;
}

int StereoConfiguration_UIControl::Validate_pattern_distance(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckFloatNumber(value, &_val._pattern_distance)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._pattern_distance > 0) {
			value += _val._pattern_distance;
		}
	}
	return err;
}
int StereoConfiguration_UIControl::Validate_max_clusterdistance(std::string& value) {
	int err = 0; 
	value = trim2stdstring(value); 
	if(value.size() && !RegExp_CheckFloatNumber(value, &_val._max_clusterdistance)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._max_clusterdistance) {
			value += _val._max_clusterdistance;
		}
	}
	return err;
} 
int StereoConfiguration_UIControl::Validate_axises_ratio(std::string& value) {
	int err = 0; 
	value = trim2stdstring(value); 
	if(value.size() && !RegExp_CheckFloatNumber(value, &_val._axises_ratio)) {
		err = 1; 
	} 
	else {
		value.resize(0);
		if(_val._axises_ratio) {
			value += _val._axises_ratio;
		}
	}
	return err;
} 
int StereoConfiguration_UIControl::Validate_min_images(std::string& value) {
	int err = 0; 
	value = trim2stdstring(value); 
	if(value.size() && !RegExp_CheckNaturalNumber(value, &_val._min_images)) {
		err = 1; 
	} 
	else {
		value.resize(0);
		if(_val._min_images) {
			value += _val._min_images;
		}
	}
	return err;
} 

int StereoConfiguration_UIControl::Validate_min_RMSE_images(std::string& value) {
	int err = 0; 
	value = trim2stdstring(value); 
	if(value.size() && !RegExp_CheckFloatNumber(value, &_val._min_RMSE_images)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._min_RMSE_images) {
			value += _val._min_RMSE_images;
		}
	}
	return err;
} 

int StereoConfiguration_UIControl::Validate_camera_names(std::string& value) {
	int err = 0; 
	value = trim2stdstring(value); 
	if(value.size() && !RegExp_CheckStrings(value, _val._camera_names, _val._camera_names.size())) {
		err = 1; 
	}
	else {
		value.resize(0);
		value += _val._camera_names;
	}
	return err;
}


int StereoConfiguration_UIControl::Validate_trigger_source_hardware(std::string& value) {
	int err = 0; 
	value = trim2stdstring(value); 
	if(value.size() && !RegExp_CheckNaturalNumber(value, &_val._trigger_source_hardware)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._trigger_source_hardware) {
			_val._trigger_source_hardware = 1;
			value += _val._trigger_source_hardware;
		}
	}
	return err;
} 

int StereoConfiguration_UIControl::Validate_pixel_threshold(std::string& value) {
	int err = 0; 
	value = trim2stdstring(value); 
	if(value.size() && !RegExp_CheckNaturalNumber(value, &_val._pixel_threshold)) {
		err = 1; 
	} 
	else {
		value.resize(0);
		if(_val._pixel_threshold) {
			value += _val._pixel_threshold;
		}
	}
	return err;
} 

int StereoConfiguration_UIControl::Validate_max_boxsize_pixels(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckNaturalNumber(value, &_val._max_boxsize_pixels)) {
		err = 1;
	}
	else
	if(value.size() && (_val._max_boxsize_pixels > 100 || _val._max_boxsize_pixels < 10)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._max_boxsize_pixels) {
			value += _val._max_boxsize_pixels;
		}
	}
	return err;
}

int StereoConfiguration_UIControl::Validate_percent_maxintensity(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckFloatNumber(value, &_val._percent_maxintensity)) {
		err = 1;
	}
	else
	if(value.size() && (_val._percent_maxintensity > 1 || _val._percent_maxintensity < 0)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._percent_maxintensity) {
			value += _val._percent_maxintensity;
		}
	}
	return err;
}

int StereoConfiguration_UIControl::Validate_calib_auto_image_capture(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckNaturalNumber(value, &_val._calib_auto_image_capture)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._calib_auto_image_capture) {
			_val._calib_auto_image_capture = 1;
			value += _val._calib_auto_image_capture;
		}
	}
	return err;
}

int StereoConfiguration_UIControl::Validate_calib_use_homography(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckNaturalNumber(value, &_val._calib_use_homography)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._calib_use_homography) {
			_val._calib_use_homography = 1;
			value += _val._calib_use_homography;
		}
	}
	return err;
}

int StereoConfiguration_UIControl::Validate_use_ellipse_fit(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckNaturalNumber(value, &_val._use_ellipse_fit)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._use_ellipse_fit) {
			_val._use_ellipse_fit = 1;
			value += _val._use_ellipse_fit;
		}
	}
	return err;
}

int StereoConfiguration_UIControl::Validate_pattern_is_whiteOnBlack(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckNaturalNumber(value, &_val._pattern_is_whiteOnBlack)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._pattern_is_whiteOnBlack) {
			_val._pattern_is_whiteOnBlack = 1;
			value += _val._pattern_is_whiteOnBlack;
		}
	}
	return err;
}

int StereoConfiguration_UIControl::Validate_pattern_is_gridOfSquares(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckNaturalNumber(value, &_val._pattern_is_gridOfSquares)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._pattern_is_gridOfSquares) {
			_val._pattern_is_gridOfSquares = 1;
			value += _val._pattern_is_gridOfSquares;
		}
		if(!_val._pattern_is_gridOfSquares) {
			if(_val._pattern_is_chessBoard) {
				err = 2;
				_crossCheck_error = true; 
			}
		}
		if(!err && _crossCheck_error) {
			_crossCheck_error = false;
			_pHList->EntryFieldsAreValid();
			_pHList->Invalidate();
		}
	}
	return err;
}

int StereoConfiguration_UIControl::Validate_pattern_is_chessBoard(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckNaturalNumber(value, &_val._pattern_is_chessBoard)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._pattern_is_chessBoard) {
			_val._pattern_is_chessBoard = 1;
			value += _val._pattern_is_chessBoard;
		}
		if(_val._pattern_is_chessBoard) {
			if(!_val._pattern_is_gridOfSquares) {
				err = 2;
				_crossCheck_error = true;
			}
		}
		if(!err && _crossCheck_error) {
			_crossCheck_error = false;
			_pHList->EntryFieldsAreValid();
			_pHList->Invalidate();
		}
	}
	return err;
}

int StereoConfiguration_UIControl::Validate_calib_rectify_alpha_param(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckFloatNumber(value, &_val._calib_rectify_alpha_param)) {
		err = 1;
	}
	else
	if(value.size() && (_val._calib_rectify_alpha_param > 1 || _val._calib_rectify_alpha_param < 0)) {
		err = 1;
	}
	else {
		value.resize(0);
		value += _val._calib_rectify_alpha_param;
	}
	return err;
}

int StereoConfiguration_UIControl::Validate_images_from_files(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckNaturalNumber(value, &_val._images_from_files)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._images_from_files) {
			_val._images_from_files = 1;
			value += _val._images_from_files;
		}
	}
	return err;
}

int StereoConfiguration_UIControl::Validate_image_height(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckNaturalNumber(value, &_val._image_height)) {
		err = 1;
	}
	else 
	if(_val._image_height > 1024 || _val._image_height < 10) {
		err = 1;
	}
	else {
		value.resize(0);
		value += _val._image_height;
	}
	return err;
}

int StereoConfiguration_UIControl::Validate_max_Y_error(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckFloatNumber(value, &_val._max_Y_error)) {
		err = 1;
	}
	else
	if(value.size() && (_val._max_Y_error >= 10 || _val._max_Y_error < 1)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._max_Y_error) {
			value += _val._max_Y_error;
		}
	}
	return err;
}

int StereoConfiguration_UIControl::Validate_12bit_format(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckNaturalNumber(value, &_val._12bit_format)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._12bit_format) {
			_val._12bit_format = 1; 
			value += _val._12bit_format;
		}
	}
	return err;
}

int StereoConfiguration_UIControl::Validate_distance_to_target(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckFloatNumber(value, &_val._distance_to_target)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._distance_to_target > 0) {
			value += _val._distance_to_target;
		}
	}
	return err;
}

int StereoConfiguration_UIControl::Validate_use_center_of_gravity(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckNaturalNumber(value, &_val._use_center_of_gravity)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._use_center_of_gravity) {
			_val._use_center_of_gravity = 1;
			value += _val._use_center_of_gravity;
		}
	}
	return err;
}

int StereoConfiguration_UIControl::Validate_visual_diagnostics(std::string& value) {
	int err = 0;
	value = trim2stdstring(value);
	if(value.size() && !RegExp_CheckNaturalNumber(value, &_val._visual_diagnostics)) {
		err = 1;
	}
	else {
		value.resize(0);
		if(_val._visual_diagnostics) {
			_val._visual_diagnostics = 1;
			value += _val._visual_diagnostics;
		}
	}
	return err;
}

void StereoConfiguration_UIControl::PushValues(const StereoConfiguration& config, const std::string&, KHyperlist *pHList) {
	_pHList = pHList; 
	_val = config; 

	HListPushEntryField(pHList, _val, _val._pattern_is_gridOfSquares, &StereoConfiguration_UIControl::Validate_pattern_is_gridOfSquares, this);
	HListPushEntryField(pHList, _val, _val._pattern_is_chessBoard, &StereoConfiguration_UIControl::Validate_pattern_is_chessBoard, this);
	HListPushEntryField(pHList, _val, _val._pattern_is_whiteOnBlack, &StereoConfiguration_UIControl::Validate_pattern_is_whiteOnBlack, this);
	HListPushEntryField(pHList, _val, _val._pattern_distance, &StereoConfiguration_UIControl::Validate_pattern_distance, this);
	HListPushEntryField(pHList, _val, _val._max_clusterdistance, &StereoConfiguration_UIControl::Validate_max_clusterdistance, this);
	HListPushEntryField(pHList, _val, _val._axises_ratio, &StereoConfiguration_UIControl::Validate_axises_ratio, this);
	HListPushEntryField(pHList, _val, _val._min_images, &StereoConfiguration_UIControl::Validate_min_images, this);
	HListPushEntryField(pHList, _val, _val._min_RMSE_images, &StereoConfiguration_UIControl::Validate_min_RMSE_images, this);
	HListPushEntryField(pHList, _val, _val._camera_names, &StereoConfiguration_UIControl::Validate_camera_names, this);
	HListPushEntryField(pHList, _val, _val._trigger_source_hardware, &StereoConfiguration_UIControl::Validate_trigger_source_hardware, this);
	HListPushEntryField(pHList, _val, _val._pixel_threshold, &StereoConfiguration_UIControl::Validate_pixel_threshold, this);
	HListPushEntryField(pHList, _val, _val._max_boxsize_pixels, &StereoConfiguration_UIControl::Validate_max_boxsize_pixels, this);
	HListPushEntryField(pHList, _val, _val._percent_maxintensity, &StereoConfiguration_UIControl::Validate_percent_maxintensity, this);
	HListPushEntryField(pHList, _val, _val._calib_auto_image_capture, &StereoConfiguration_UIControl::Validate_calib_auto_image_capture, this);
	HListPushEntryField(pHList, _val, _val._calib_rectify_alpha_param, &StereoConfiguration_UIControl::Validate_calib_rectify_alpha_param, this);
	HListPushEntryField(pHList, _val, _val._calib_use_homography, &StereoConfiguration_UIControl::Validate_calib_use_homography, this);
	HListPushEntryField(pHList, _val, _val._use_ellipse_fit, &StereoConfiguration_UIControl::Validate_use_ellipse_fit, this);
	HListPushEntryField(pHList, _val, _val._image_height, &StereoConfiguration_UIControl::Validate_image_height, this);
	HListPushEntryField(pHList, _val, _val._max_Y_error, &StereoConfiguration_UIControl::Validate_max_Y_error, this);
	HListPushEntryField(pHList, _val, _val._12bit_format, &StereoConfiguration_UIControl::Validate_12bit_format, this); 
	HListPushEntryField(pHList, _val, _val._images_from_files, &StereoConfiguration_UIControl::Validate_images_from_files, this);
	HListPushEntryField(pHList, _val, _val._distance_to_target, &StereoConfiguration_UIControl::Validate_distance_to_target, this);
	HListPushEntryField(pHList, _val, _val._use_center_of_gravity, &StereoConfiguration_UIControl::Validate_use_center_of_gravity, this);
	HListPushEntryField(pHList, _val, _val._visual_diagnostics, &StereoConfiguration_UIControl::Validate_visual_diagnostics, this);
}

void StereoConfiguration_UIControl::PostValues() { 
	if(_pHList) { 
		_g_images_frame->NEW_StereoConfiguration(g_configuration << this->_val);
	} 
} 




