#ifndef vision_userinterfaceH
#define vision_userinterfaceH

#include "stdafx.h"

#include "FrameWnd.h" 
#include "XConfiguration.h" 


struct IUIValidator { 
	PVOID _validator_key; 
	virtual int Validate(const std::string& value, const std::string& wsi_id) = 0; 
}; 


struct IUIControl { 
	KHyperlist *_pHList; 
	IUIValidator *_pValidateExtra; 

	virtual void PostValues() = 0; 
}; 



class StereoConfiguration_UIControl: public IUIControl {
public: 
	StereoConfiguration _val;

	int Validate_pattern_distance(std::string& value);
	int Validate_max_clusterdistance(std::string& value);
	int Validate_axises_ratio(std::string& value);
	int Validate_min_images(std::string& value);
	int Validate_min_RMSE_images(std::string& value);
	int Validate_camera_names(std::string& value);
	int Validate_trigger_source_hardware(std::string& value);
	int Validate_pixel_threshold(std::string& value); 
	int Validate_max_boxsize_pixels(std::string& value);
	int Validate_percent_maxintensity(std::string& value);
	int Validate_calib_auto_image_capture(std::string& value); 
	int Validate_pattern_is_whiteOnBlack(std::string& value);
	int Validate_pattern_is_gridOfSquares(std::string& value);
	int Validate_pattern_is_chessBoard(std::string& value);
	int Validate_calib_use_homography(std::string& value);
	int Validate_use_ellipse_fit(std::string& value);
	int Validate_calib_rectify_alpha_param(std::string& value);
	int Validate_images_from_files(std::string& value);
	int Validate_image_height(std::string& value);
	int Validate_max_Y_error(std::string& value);
	int Validate_12bit_format(std::string& value);
	int Validate_distance_to_target(std::string& value); 
	int Validate_use_center_of_gravity(std::string& value);
	int Validate_visual_diagnostics(std::string& value);

	void PushValues(const StereoConfiguration&, const std::string& uuid, KHyperlist*);
	void PostValues(); 

	bool _crossCheck_error; 
}; 




class FrameEdit: public KFrame { 
public: 
	FrameEdit(HINSTANCE hInstance, HWND hParent); 

	void ProcessToolBarCommand(SReBar& rebar); 
	void ProcessToolBarNotify(SReBar& rebar); 
	void OnChangeFrameState(SReBar& rebar, int state); 


	IUIControl *_user_input; 

	StereoConfiguration_UIControl _stereo_configuration;

	template <typename T, typename UIControl> 
	BOOL ShowUserInput(const T& row, UIControl& user_input_ctl, const std::string& uuid, IUIValidator *pvalidate_extra = 0) { 
		BOOL ok = FALSE; 
		KHyperlist *pHList = FindHListObject(GetRootSerializer((T*)0)._wsi_id); 
		if(pHList) { 
			user_input_ctl.PushValues(row, uuid, pHList); 
			SetTopMostTreeObject(*pHList); 
			SetTopMost(); 
			ok = TRUE; 

			_user_input = &user_input_ctl; 
			_user_input->_pValidateExtra = pvalidate_extra; 
		} 
		return ok; 
	} 
}; 


extern FrameEdit *_g_edit_frame; 




#endif //vision_userinterfaceH

