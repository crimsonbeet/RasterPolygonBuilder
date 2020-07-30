#ifndef visionFrameMainH
#define visionFrameMainH

#include "FrameWnd.h" 

#include "FrameReconstructedImage.h"
#include "FrameEdit.h"



class FrameMain: public KFrame { 
public: 
	FrameMain(HINSTANCE hInstance, HICON hIcon = NULL); 

	void Instantiate(); 

	SImageList _iml; 

	std::auto_ptr<SReBar> _prebar; 

	void ProcessToolBarCommand(SReBar& rebar); 

	FrameReconstructedImage *_images_frame;

	void OnMessage(SOnMessageParams& params); 
	void OnCanvasMessage(SOnMessageParams& params);
	void OnPaint(SOnDrawParams& params);
	void OnCanvasPaint(SOnDrawParams& params);

	void InsertHistory(KFrame*); 

	BOOL StepBackHistory(); 
	BOOL StepForthHistory(); 

	BOOL OtherInstanceRunning(); 
}; 

class MYKFrameHistory: public KFrameHistory {
public: 
	MYKFrameHistory(std::auto_ptr<SReBar>& prebar, SToolBar *toolbar): KFrameHistory(prebar, toolbar) {
	}	
	KFrame * getCurrentFrame() {
		return _frameshistory[_frameshistory_pos]; 
	}
};


extern FrameMain *_g_main_frame; 
extern MYKFrameHistory *_g_frame_history;


void rootCVWindows(KFrame *frame/*in*/, size_t cv_windows_count/*in*/, size_t veccanvas_offset/*in*/, std::string *cv_window_names/*out*/); 


#endif //visionFrameMainH

