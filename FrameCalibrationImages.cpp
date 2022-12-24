
#include "stdafx.h"
#include "IPCInterface.h" 

#include "FrameCalibrationImages.h"
#include "FrameMain.h"




const std::string _g_document_images[1] = {
	IMG_SAVEDOCUMENT
};

const std::string _g_document_images_disabled[1] = {
	IMG_SAVEDOCUMENT_D
};

const std::string _g_document_images_hot[1] = {
	IMG_SAVEDOCUMENT_H
};


#define Stop_capturing_images 120001 



FrameCalibrationImages *_g_calibrationimages_frame = 0;


FrameCalibrationImages::FrameCalibrationImages(HINSTANCE hInstance, HWND hParent): KFrame(hInstance), _stop_capturing(false) {
	Configure(ReadTextFile("FrameCalibrationImages.txt")); 
	Create(hInstance, hParent); 
} 


void FrameCalibrationImages::ProcessToolBarCommand(SReBar& rebar) {
	if(rebar._lparam == (LPARAM)_toolbar->_hwndtoolbar) {
		switch(rebar._wparam) {
		case 203/*save document*/:
		_stop_capturing = true; 
		break;
		}
	}
}

void FrameCalibrationImages::ProcessToolBarNotify(SReBar& rebar) {
	NMHDR *pnmhdr = (NMHDR*)(rebar._lparam);
	if(pnmhdr) {
		if(pnmhdr->hwndFrom == _toolbar->_hwndtoolbar) {
			switch(pnmhdr->code) {
			case TBN_GETINFOTIPA: {
				LPNMTBGETINFOTIPA tipinfo = (LPNMTBGETINFOTIPA)rebar._lparam;
				std::string str;
				switch(tipinfo->iItem) {
					case 203:
					GetStringentity(Stop_capturing_images, str);
					strncpy(tipinfo->pszText, str.c_str(), tipinfo->cchTextMax);
					break;
					}
				}
				break;
			}
		}
	}
}

void FrameCalibrationImages::OnChangeFrameState(SReBar& rebar, int state) {
	switch(state) {
	case 1: /*active*/
	if(!_toolbar) {
		ActivateToolBar(rebar, 203, _g_document_images, _g_document_images_hot, _g_document_images_disabled, ARRAY_NUM_ELEMENTS(_g_document_images));
		_toolbar->SetButtonStateByindex(TBSTATE_ENABLED, 0/*btn - save document*/, true/*remove enabled*/);
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


