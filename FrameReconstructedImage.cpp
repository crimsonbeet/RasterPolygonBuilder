
#include "stdafx.h"
#include "IPCInterface.h" 

#include "FrameReconstructedImage.h"
#include "FrameMain.h"

//#pragma comment(lib, "comctl32.lib")


const std::string _g_document_images[2] = { 
	IMG_NEWDOCUMENT, 
	IMG_DELETEDOCUMENT
}; 

const std::string _g_document_images_disabled[2] = { 
	IMG_NEWDOCUMENT_D, 
	IMG_DELETEDOCUMENT_D 
}; 

const std::string _g_document_images_hot[2] = { 
	IMG_NEWDOCUMENT_H, 
	IMG_DELETEDOCUMENT_H 
}; 



const std::string _g_altdocument_images[2] = {
	IMG_NEWDOCUMENT,
	IMG_UNDODELETEDOCUMENT
};

const std::string _g_altdocument_images_disabled[2] = {
	IMG_NEWDOCUMENT_D,
	IMG_DELETEDOCUMENT_D
};

const std::string _g_altdocument_images_hot[2] = {
	IMG_NEWDOCUMENT_H,
	IMG_UNDODELETEDOCUMENT_H
};



FrameReconstructedImage *_g_images_frame = 0;

wsi_reenterantgate _g_descriptorsLOCKER; 


#define RunTime_Stats 110001 
#define Appl_Setting 110002 
#define New_Reserved 110003 
#define Delete_Calibration 110004 
#define Restore_Calibration 110005 
#define CaptureImage_Action 110006 
#define Delete_Features 110007 
#define Restore_Features 110008 
#define Create_Features 110009 
#define Show_Features 110010 


extern bool g_bTerminated;
extern bool g_bRestart;


FrameReconstructedImage::FrameReconstructedImage(HINSTANCE hInstance, HWND hParent): KFrame(hInstance), _ready(false) {
	Configure(ReadTextFile("FrameReconstructedImage.txt")); 
	Create(hInstance, hParent); 

	KHyperlist *pHList = 0; 
	pHList = FindHListObject(GetRootSerializer((StereoConfiguration*)0)._wsi_id);
	if(pHList) { 
		pHList->AddRootItem(&_config_item); 
	} 

	_pframeedit = new FrameEdit(hInstance, hParent);
	_g_edit_frame = _pframeedit; 

	_pframecalibration = new FrameCalibrationImages(hInstance, hParent);
	_g_calibrationimages_frame = _pframecalibration;

	_delegates->_ondraw_delegates.Add(&FrameReconstructedImage::OnPaint, this);
} 

void FrameReconstructedImage::ONDisconnect() {
	KHyperlist *pHList = 0; 
	if(pHList) { 
	} 
} 


void FrameReconstructedImage::NEW_StereoConfiguration(StereoConfiguration& config) {
	_config_item._configuration = config; 

	KHyperlist *pHList = 0; 
	pHList = FindHListObject(GetRootSerializer((StereoConfiguration*)0)._wsi_id);
	if(pHList) { 
		pHList->RedrawItem(&_config_item); 
	} 

	bool is_disabled = false; 

	if(MyGetFileSize(CalibrationFileName())) {
		KWinChangeMaskedImageList(_g_document_images, *_toolbar->_piml, 2);
		KWinChangeMaskedImageList(_g_document_images_hot, *_toolbar->_pimlhot, 2);
	}
	else
	if(MyGetFileSize(g_path_calibrate_file_backup)) {
		KWinChangeMaskedImageList(_g_altdocument_images, *_toolbar->_piml, 2);
		KWinChangeMaskedImageList(_g_altdocument_images_hot, *_toolbar->_pimlhot, 2);
	}
	else {
		is_disabled = true;
	}

	_toolbar->SetButtonStateByindex(TBSTATE_ENABLED, 1/*delete document*/, is_disabled);
}




void FrameReconstructedImage::ProcessToolBarCommand(SReBar& rebar) {
	if(rebar._lparam == (LPARAM)_toolbar->_hwndtoolbar) { 
		std::string path_file_from = CalibrationFileName();
		std::string path_file_to = g_path_calibrate_file_backup;
		std::string str;
		switch(rebar._wparam) {
			case 203/*new document*/: 
			break; 
			case 204/*delete document*/: 
			if(_toolbar->_piml->_images[1] == _g_altdocument_images[1]) {
				path_file_from = g_path_calibrate_file_backup;
				path_file_to = CalibrationFileName();
			}
			ReadTextFile(path_file_from, str);
			if(str.size()) {
				WriteTextFile(path_file_to, str);
				str.resize(0); 
				WriteTextFile(path_file_from, str);

				g_bRestart = true;
				g_bTerminated = true;
			}

			break;
		} 
	} 
} 

void FrameReconstructedImage::ProcessToolBarNotify(SReBar& rebar) {
	NMHDR *pnmhdr = (NMHDR*)(rebar._lparam); 
	if(pnmhdr) { 
		if(pnmhdr->hwndFrom == _toolbar->_hwndtoolbar) { 
			const char *path_file_from = g_path_features_file;
			const char *path_file_to = g_path_features_file_backup;
			std::string features_str;

			switch(pnmhdr->code) {
				case TBN_DROPDOWN: { 
					LPNMTOOLBAR lpnmtb = (LPNMTOOLBAR)pnmhdr; 
					if(lpnmtb->iItem == 203/*new document*/) { 
						HMENU hmenu = CreatePopupMenu(); 
						std::string str;
						GetStringentity(Create_Features, str);
						AppendMenu(hmenu, MF_STRING |MF_ENABLED, 223, str.c_str());
						GetStringentity(New_Reserved, str);
						AppendMenu(hmenu, MF_STRING |MF_ENABLED, 225, str.c_str()); 

						UINT fuflags = TPM_LEFTALIGN|TPM_TOPALIGN|TPM_RIGHTBUTTON|TPM_NONOTIFY|TPM_RETURNCMD; 

						POINT point; 
						point.x = lpnmtb->rcButton.left; 
						point.y = lpnmtb->rcButton.bottom; 
						ClientToScreen(_toolbar->_hwndtoolbar, &point); 

						int rc = TrackPopupMenuEx(hmenu, fuflags, point.x, point.y, _hwnd, 0); 
						DestroyMenu(hmenu); 

						rebar._lresult = TBDDRET_DEFAULT; 

						switch(rc) {
							case 223: 
							ReadTextFile(path_file_from, features_str);
							if(features_str.size()) {
								WriteTextFile(path_file_to, features_str);
								features_str.resize(0);
								WriteTextFile(path_file_from, features_str);
							}
							else {
								g_configuration._engine_features_requested = 1; 
								VS_SaveConfiguration(g_configuration);
							}
							g_bRestart = true;
							g_bTerminated = true;
							break;

							case 224:
							break;

							case 225: 
							break; 

						} 
					} 
				} 
				break; 

				case TBN_GETINFOTIPA: { 
					LPNMTBGETINFOTIPA tipinfo = (LPNMTBGETINFOTIPA)rebar._lparam; 
					std::string str[3]; 
					switch(tipinfo->iItem) { 
						case 203: 
						GetStringentity(Delete_Features, str[0]);
						GetStringentity(Restore_Features, str[1]);
						GetStringentity(Create_Features, str[2]);
						strncpy(tipinfo->pszText, (str[0] + '|' + str[1] + '|' + str[2]).c_str(), tipinfo->cchTextMax);
						break; 
						case 204: 
						GetStringentity(_toolbar->_piml->_images[1] == _g_document_images[1] ? Delete_Calibration : Restore_Calibration, str[0]);
						strncpy(tipinfo->pszText, str[0].c_str(), tipinfo->cchTextMax); 
						break; 
					} 
				}
				break; 
			} 
		} 
	} 
} 

void FrameReconstructedImage::OnChangeFrameState(SReBar& rebar, int state) {
	switch(state) { 
		case 1: /*active*/
			if(!_toolbar) { 
//				rebar._fbandstyle &= ~RBBS_NOGRIPPER; 
				ActivateToolBar(rebar, 203, _g_document_images, _g_document_images_hot, _g_document_images_disabled, ARRAY_NUM_ELEMENTS(_g_document_images)); 

				_toolbar->SetButtonStyleByindex(/*BTNS_DROPDOWN*/BTNS_WHOLEDROPDOWN, 0/*btn - new document*/); 
				_toolbar->SetButtonStateByindex(TBSTATE_ENABLED, 1/*btn - delete document*/, true/*remove enabled*/); 

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

void FrameReconstructedImage::OnPaint(SOnDrawParams& params) {
	if(_g_images_frame->_ready) {
		if (!g_bTerminated) {
			IPCSetLogHandler(_g_images_frame->_hwnd);
		}
	}
}





BOOL StereoConfigurationUIElement::OnClick(HWND hWndCntxt, int nCell/*1 - Based*/) {
	_g_descriptorsLOCKER.lock();

	_g_edit_frame->ShowUserInput(_configuration, _g_edit_frame->_stereo_configuration, std::string());
	_g_main_frame->InsertHistory(_g_edit_frame);

	_g_descriptorsLOCKER.unlock();

	return FALSE; 
} 

BOOL StereoConfigurationUIElement::GetText(int nColumn, std::string& text) { // 0 - Based 
	std::ostringstream otext; 
	std::string str; 

	_g_descriptorsLOCKER.lock();

	switch(nColumn + 1) { 
		case 1: 
			GetStringentity(Appl_Setting, str);
			otext << str << ':'; 
		break; 
		case 3:  
			otext << "HTr " << itostdstring(_configuration._trigger_source_hardware);
			otext << ' ';
			otext << "THR " << itostdstring(_configuration._pixel_threshold);
		break; 
	} 

	_g_descriptorsLOCKER.unlock();

	otext.str().swap(text); 

	return text.size() != 0; 
} 

void StereoConfigurationUIElement::GetIsHyperlink(BYTE bIsHyperlink[32]) {
} 




