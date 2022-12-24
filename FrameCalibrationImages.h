#ifndef visionFrameCalibrationImagesH
#define visionFrameCalibrationImagesH

#include "FrameWnd.h" 
#include "XConfiguration.h" 

#include "FrameEdit.h" 





class FrameCalibrationImages: public KFrame {

public: 
	void ProcessToolBarCommand(SReBar& rebar);
	void ProcessToolBarNotify(SReBar& rebar);
	void OnChangeFrameState(SReBar& rebar, int state);

	FrameCalibrationImages(HINSTANCE hInstance, HWND hParent);

	bool _stop_capturing; 
}; 




extern FrameCalibrationImages *_g_calibrationimages_frame;



#endif //visionFrameCalibrationImagesH

