
#include "stdafx.h"
#include "IPCInterface.h" 

#include "FrameMain.h"

#include "opencv2\highgui\highgui_c.h"


const std::string _g_arrow_images[2] = { 
	IMG_ARROW_LEFT, 
	IMG_ARROW_RIGHT 
}; 

const std::string _g_arrow_images_disabled[2] = { 
	IMG_ARROW_LEFT_D, 
	IMG_ARROW_RIGHT_D 
}; 

const std::string _g_arrow_images_hot[2] = { 
	IMG_ARROW_LEFT_H, 
	IMG_ARROW_RIGHT_H 
}; 


FrameMain *_g_main_frame;
MYKFrameHistory *_g_frame_history;

extern bool g_bTerminated;
extern bool g_bUserTerminated; 
extern bool g_bCalibrationExists; 




void rootCVWindows(KFrame *frame/*in*/, size_t cv_windows_count/*in*/, size_t veccanvas_offset/*in*/, std::string *cv_window_names/*out*/) {
	if(frame && frame->_veccanvas.size() >= (veccanvas_offset + cv_windows_count)) {
		for(size_t j = 0; j < cv_windows_count; ++j) {
			if(frame->_veccanvas[j + veccanvas_offset]._hwnd) {
				if(frame->_veccanvas[j + veccanvas_offset]._ctl.name.size() > 0) {
					cv_window_names[j] = frame->_veccanvas[j + veccanvas_offset]._ctl.name;
					cv::namedWindow(cv_window_names[j]);
					HWND hwnd = (HWND)cvGetWindowHandle(cv_window_names[j].c_str());
					if(GetParent(hwnd) != frame->_veccanvas[j + veccanvas_offset]._hwnd) {
						HWND hparent = GetParent(hwnd);
						SetParent(hwnd, frame->_veccanvas[j + veccanvas_offset]._hwnd);
						ShowWindow(hparent, SW_HIDE);
					}
				}
			}
		}
	}
}




FrameMain::FrameMain(HINSTANCE hInstance, HICON hIcon): KFrame(hInstance, hIcon) { 
	_images_frame = 0;
} 

BOOL FrameMain::OtherInstanceRunning() {
	BOOL stat = FALSE; 
	Configure(ReadTextFile("FrameMain.txt")); 
	if(FindWindow(NULL, _ctl.name.c_str()) != NULL) { 
		stat = TRUE; 
	} 
	return stat; 
} 


void FrameMain::Instantiate() {
	Configure(ReadTextFile("FrameMain.txt")); 
	Create(_hinst, NULL); 

	SetTopMost(HWND_TOP); 

	int nstep = 1; 
	while(nstep) switch(nstep) { 
		case 1: // check if rebar can be created. 
			nstep = _veccanvas.size()? 2: 0/*end*/; 
			break; 
		case 2: // create rebar. 
		{ 
			KCanvas& canvas = _veccanvas[0]; 

			SCreateRebar cr_rebar; 
			cr_rebar._idrebar = 102; 
			cr_rebar._hwndparent = canvas._hwnd; 
//			cr_rebar._piml = &_iml; 

			nstep = KWinCreateRebar(cr_rebar, &_prebar)? 3: 0; 
			if(!nstep) { 
				std::cout << "FrameMain->can not create rebar\r\n"; 
			} 
		} 
		break; 
		case 3: // create back-forth toolbar. 
		{ 
			nstep = ActivateToolBar(*(_prebar.get()), 103, _g_arrow_images, _g_arrow_images_hot, _g_arrow_images_disabled, ARRAY_NUM_ELEMENTS(_g_arrow_images))? 4: 0; 
		} 
		break; 
		case 4: // create frame-history based on rebar. 
		{ 
			_g_frame_history = new MYKFrameHistory(_prebar, _toolbar); // it takes ownership of _prebar. 
			nstep = 5; 
		} 
		break; 
		case 5: // create reconstructed images frame. 
		if(_veccanvas.size() > 1) { 
			_images_frame = new FrameReconstructedImage(_hinst, _veccanvas[1]._hwnd);
			_g_images_frame = _images_frame;
			_g_images_frame->SetTopMost();

			if(_g_images_frame->_veccanvas.size() > 0) {
				_g_images_frame->_veccanvas[0]._delegates->_onmessage_delegates.Add(&FrameMain::OnMessage, this);
				_g_images_frame->_veccanvas[0]._delegates->_ondraw_delegates.Add(&FrameMain::OnCanvasPaint, this);
			}

			_g_frame_history->InsertHistory(_g_images_frame);
		} 
		nstep = 0; 
		break; 
	} 

	_delegates->_onmessage_delegates.Add(&FrameMain::OnMessage, this);
	_delegates->_ondraw_delegates.Add(&FrameMain::OnPaint, this);

	ShowWindow(); 
} 

void FrameMain::ProcessToolBarCommand(SReBar& rebar) {
	if(rebar._lparam == (LPARAM)_toolbar->_hwndtoolbar) { 
		switch(rebar._wparam) { 
			case 103/*_toolbar->_idfirstbutton*/: 
				StepBackHistory(); 
			break; 
			case 104/*_toolbar->_idfirstbutton + 1*/: 
				StepForthHistory(); 
			break; 
		} 
	} 
} 

DWORD WINAPI Decoupled_IPCSetLogHandler(LPVOID lpvoid) {
	IPCSetLogHandler((HWND)lpvoid);
	return 0;
}


void FrameMain::OnMessage(SOnMessageParams& params) {
//	static bool is_sizing = false; 
	switch(params._umsg) { 
		case WM_CLOSE: 
			g_bUserTerminated = true;
		case WM_DESTROY:
			g_bTerminated = true;
			QueueWorkItem(Decoupled_IPCSetLogHandler, 0);
		break;
		case WM_NOTIFY: 
			if(params._lparam) { 
				if(((LPNMHDR)params._lparam)->code == RBN_AUTOBREAK) { 
					((LPNMREBARAUTOBREAK)params._lparam)->fAutoBreak = FALSE; 
				} 
			} 
		break; 
		case WM_SIZE: 
		if(IsIconic(_hwnd)) { 
		} 
		else 
		if(IsSizing()) { 
		} 
		else { 
			RECT clrect; 
			if(_veccanvas.size() > 0) { 
				GetClientRect(_veccanvas[0]._hwnd, &clrect); 
//				LONG rebar_height = SendMessage(_g_frame_history->GetRebarPtr()->_hwndrebar, RB_GETBARHEIGHT, 0, 0); 
				RECT winrect; 
				GetWindowRect(_g_frame_history->GetRebarPtr()->_hwndrebar, &winrect); 
				LONG rebar_height = winrect.bottom - winrect.top; 
				if(rebar_height < clrect.bottom) { 
					clrect.bottom = rebar_height; // Do not change the height of rebar. 
				} 
				MoveWindow(_g_frame_history->GetRebarPtr()->_hwndrebar, 0, 0, clrect.right, clrect.bottom, TRUE); 
			} 
			if(_veccanvas.size() > 1) { 
				GetClientRect(_veccanvas[1]._hwnd, &clrect); 
				if(_images_frame) {
					MoveWindow(_images_frame->_hwnd, 0, 0, clrect.right, clrect.bottom, TRUE);
					if(_images_frame->_pframeedit) {
						MoveWindow(_images_frame->_pframeedit->_hwnd, 0, 0, clrect.right, clrect.bottom, TRUE);
					}
				}
			} 
		} 
		break; 
	} 
} 

void FrameMain::OnCanvasMessage(SOnMessageParams& params) {
	//	static bool is_sizing = false; 
	switch(params._umsg) {
		case WM_CLOSE:
		case WM_DESTROY:
		break;
	case WM_NOTIFY:
		if(params._lparam) {
			if(((LPNMHDR)params._lparam)->code == RBN_AUTOBREAK) {
				((LPNMREBARAUTOBREAK)params._lparam)->fAutoBreak = FALSE;
			}
		}
		break;
	case WM_SIZE:
		if(IsIconic(_hwnd)) {
		}
		else
		if(IsSizing()) {
		}
		else {
		}
		break;
	}
}

void FrameMain::OnPaint(SOnDrawParams& params) { // One time deal. 
	_g_frame_history->ShowCurrent(); 
	_delegates->_ondraw_delegates.Remove(&FrameMain::OnPaint, this);

	if(_g_images_frame) {
		IPCSetLogHandler(_g_images_frame->_hwnd);
		_g_images_frame->_ready = true;
	} 
} 

void FrameMain::OnCanvasPaint(SOnDrawParams& params) { // One time deal. 
	_g_images_frame->_veccanvas[0]._delegates->_ondraw_delegates.Remove(&FrameMain::OnCanvasPaint, this);
}


void FrameMain::InsertHistory(KFrame *pframe) {
	_g_frame_history->InsertHistory(pframe); 
} 

BOOL FrameMain::StepBackHistory() {
	BOOL rc = _g_frame_history->StepBackHistory();
	return rc;
} 

BOOL FrameMain::StepForthHistory() {
	return _g_frame_history->StepForthHistory(); 
} 












BOOL DecodeHtmlentities(std::string& szText); // from hyperlist.h



std::map<int, std::string> g_mapStringentities;
std::map<std::string, int> g_mapReferencedStringentities;
BOOL InitializeStringentities() {
	std::string strStringEntities;
	BOOL rc = FALSE;
	if(ReadTextFile("FrameStaticTexts.txt", strStringEntities)) {
		for(size_t j = 0; j < strStringEntities.size(); ++j) {
			//if(isdigit(strStringEntities[j])) { 
			if(strStringEntities[j] >= '0' && strStringEntities[j] <= '9') {
				size_t code_begin = j;
				size_t code_end = j;
				while(++j < strStringEntities.size() && ((unsigned char)strStringEntities[j]) >= 0x20) {
					if(code_begin == code_end) {
						if(strStringEntities[j] == 0x20) {
							code_end = j;
						}
					}
				}
				if(code_begin < code_end) {
					int code = atoi(&strStringEntities[code_begin]);
					int str_pos = (int)code_end + 1;
					std::string str = strStringEntities.substr(str_pos, j - str_pos);
					DecodeHtmlentities(str);
					g_mapStringentities.insert(std::map<int, std::string>::value_type(code, str));
					rc = TRUE;
				}
			}
		}
	}
	return rc;
}
BOOL InitializeReferencedStringentities() {
	if(g_mapReferencedStringentities.size() == 0) {
		g_mapReferencedStringentities.insert(std::map<std::string, int>::value_type(APPLICATION_INITIALIZED, 160020));
		g_mapReferencedStringentities.insert(std::map<std::string, int>::value_type(APPLICATION_ACTIVE, 160021));
		g_mapReferencedStringentities.insert(std::map<std::string, int>::value_type(APPLICATION_CLOSED, 160025));
	}
	return TRUE;
}

BOOL g_mapStringentitiesInitialized = InitializeStringentities();
BOOL g_mapReferencedStringentitiesInitialized = InitializeReferencedStringentities();



BOOL GetStringentity(int iKey, std::string& szText) {
	BOOL rc = FALSE;
	szText.resize(0);
	if(g_mapStringentitiesInitialized) {
		std::map<int, std::string>::iterator it = g_mapStringentities.find(iKey);
		if(it != g_mapStringentities.end()) {
			szText = (*it).second;
			rc = TRUE;
		}
	}
	return rc;
}

BOOL GetReferencedStringentity(const std::string& szKey, std::string& szText) {
	BOOL rc = FALSE;
	if(g_mapStringentitiesInitialized) {
		std::map<std::string, int>::iterator it = g_mapReferencedStringentities.find(szKey);
		if(it != g_mapReferencedStringentities.end()) {
			rc = GetStringentity((*it).second, szText);
		}
	}
	if(!rc) {
		szText = szKey;
	}
	return rc;
}




