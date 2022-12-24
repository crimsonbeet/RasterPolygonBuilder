#ifndef FrameReconstructedImageH
#define FrameReconstructedImageH

#include "FrameWnd.h" 
#include "XConfiguration.h" 

#include "FrameEdit.h" 
#include "FrameCalibrationImages.h"





class StereoConfigurationUIElement: public CHListItem {
public: 
	StereoConfiguration _configuration;

	BOOL OnClick(HWND hWndCntxt, int nCell/*1 - Based*/); 

	BOOL GetText(int nColumn, std::string& text);  // 0 - Based 

	void GetIsHyperlink(BYTE bIsHyperlink[32]); 
}; 



class FrameReconstructedImage: public KFrame {

public: 
	FrameReconstructedImage(HINSTANCE hInstance, HWND hParent);

	StereoConfigurationUIElement _config_item;

	void ProcessToolBarCommand(SReBar& rebar); 
	void ProcessToolBarNotify(SReBar& rebar); 
	void OnChangeFrameState(SReBar& rebar, int state); 

	void OnPaint(SOnDrawParams& params);

	void NEW_StereoConfiguration(StereoConfiguration& config);

	void ONDisconnect(); 

	FrameEdit *_pframeedit; 
	FrameCalibrationImages* _pframecalibration;

	bool _ready; 
}; 




extern FrameReconstructedImage *_g_images_frame;
extern wsi_reenterantgate _g_descriptorsLOCKER;



#endif //FrameReconstructedImageH

