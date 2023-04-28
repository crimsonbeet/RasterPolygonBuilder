// stdafx.cpp : source file that includes just the standard includes
// CalibrationStereoCamera.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.h"

#define WSICLASSFACTORY_IMPL
#define OLE_DATE_IMPL

#include "XConfiguration.h"
#include "XClasses.h"
#include "XAndroidCamera.h"

#include "FrameWnd.h" 

#include "OleDate.h" 

const char *IMG_ARROW_LEFT = "Bitmaps\\arrow_left_24.bmp";
const char *IMG_ARROW_LEFT_D = "Bitmaps\\arrow_left_24_d.bmp";
const char *IMG_ARROW_LEFT_H = "Bitmaps\\arrow_left_24_h.bmp";
const char *IMG_ARROW_RIGHT = "Bitmaps\\arrow_right_24.bmp";
const char *IMG_ARROW_RIGHT_D = "Bitmaps\\arrow_right_24_d.bmp";
const char *IMG_ARROW_RIGHT_H = "Bitmaps\\arrow_right_24_h.bmp";

const char *IMG_NEWDOCUMENT = "Bitmaps\\new_document_lined_24.bmp";
const char *IMG_NEWDOCUMENT_D = "Bitmaps\\new_document_lined_24_d.bmp";
const char *IMG_NEWDOCUMENT_H = "Bitmaps\\new_document_lined_24_h.bmp";
const char *IMG_DELETEDOCUMENT = "Bitmaps\\delete_24.bmp";
const char *IMG_DELETEDOCUMENT_D = "Bitmaps\\delete_24_d.bmp";
const char *IMG_DELETEDOCUMENT_H = "Bitmaps\\delete_24_h.bmp";
const char *IMG_SAVEDOCUMENT = "Bitmaps\\save_24.bmp";
const char *IMG_SAVEDOCUMENT_D = "Bitmaps\\save_24_d.bmp";
const char *IMG_SAVEDOCUMENT_H = "Bitmaps\\save_24_h.bmp";
const char *IMG_STOPDOCUMENT = "Bitmaps\\stop_24.bmp";
const char *IMG_STOPDOCUMENT_D = "Bitmaps\\stop_24_d.bmp";
const char *IMG_STOPDOCUMENT_H = "Bitmaps\\stop_24_h.bmp";
const char *IMG_FINISHDOCUMENT = "Bitmaps\\finish_document_24.bmp";
const char *IMG_FINISHDOCUMENT_D = "Bitmaps\\finish_document_24_d.bmp";
const char *IMG_FINISHDOCUMENT_H = "Bitmaps\\finish_document_24_h.bmp";

const char *IMG_UNDODELETEDOCUMENT = "Bitmaps\\undodelete_24.bmp";
const char *IMG_UNDODELETEDOCUMENT_D = "Bitmaps\\delete_24_d.bmp";
const char *IMG_UNDODELETEDOCUMENT_H = "Bitmaps\\undodelete_24_h.bmp";


const char* IMG_CAMERA = "Bitmaps\\camera_24.bmp";
const char* IMG_CAMERA_D = "Bitmaps\\camera_24_d.bmp";
const char* IMG_CAMERA_H = "Bitmaps\\camera_24_h.bmp";


std::string itostdstring(int j) {
	std::ostringstream ostr;
	ostr << j;
	return ostr.str();
}

std::string i64tostdstring(__int64 j) {
	std::ostringstream ostr;
	ostr << j;
	return ostr.str();
}

std::string trim2stdstring(const std::string& str) {
	size_t nsize = str.find_last_not_of(' ');
	if(nsize < str.size()) {
		return std::string(str, 0, nsize + 1);
	}
	return std::string();
	//	std::string::reverse_iterator last = std::find_if(str.rbegin(), str.rend(), std::not1(std::ptr_fun(isspace))); 
	//	int nsize = last.base() - str.begin(); 
	//	str.resize(nsize); 
}


void PrintLastError(DWORD dwErr) {
	if(dwErr == 0) {
		dwErr = GetLastError();
	}
	char szErr[4096];
	memset(szErr, 0, sizeof(szErr));

	FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, NULL, dwErr, 0, szErr, sizeof(szErr), NULL);
	std::cout << "Win Error : " << szErr << std::endl;
}


int MyCreateDirectory(const std::string& dir_name, const std::string& err_prefix) {
	std::cout << "Creating directory" << ' ' << dir_name << std::endl;
	int rc = 0;
	HANDLE hdir = CreateFile(dir_name.c_str(), FILE_LIST_DIRECTORY, FILE_SHARE_READ | FILE_SHARE_DELETE, NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
	if(hdir == INVALID_HANDLE_VALUE) {
		rc = 1;
		if(!CreateDirectory(dir_name.c_str(), NULL)) {
			std::cout << err_prefix << ' ' << "cannot create directory" << ' ' << dir_name << std::endl;
			rc = -1;
		}
	}
	else {
		CloseHandle(hdir);
	}
	return rc;
}

size_t MyGetFileSize(const std::string& filename) {
	size_t rc = 0;
	HANDLE h = OSRDONLYOpenFile(filename);
	if(h != (HANDLE)-1) {
		rc = OSSEEKFileSize(h);
		OSCloseFile(h);
	}
	return rc; 
}

int Delete_FilesInDirectory(const std::string& dir) {
	int err = 0;
	std::vector<std::string> v_sub_dirs;

	WIN32_FIND_DATAA stFindData;
	std::string strFileName = dir + "*";
	HANDLE hFind = FindFirstFileA(strFileName.c_str(), &stFindData);
	BOOL fOk = (hFind != INVALID_HANDLE_VALUE);
	while(!err && fOk) {
		if((stFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
			std::string strLocalFileName = dir + stFindData.cFileName;
			std::cout << "Deleting file" << ' ' << strLocalFileName << std::endl;
			if(!DeleteFileA(strLocalFileName.c_str())) {
				LPVOID lpMsgBuf;
				FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, 0, GetLastError(), 0, (LPSTR)&lpMsgBuf, 0, 0);
				std::cout << "DeleteFile(): " << (PCHAR)lpMsgBuf << std::endl;
				LocalFree(lpMsgBuf);
				err = 1;
			}
		}

		fOk = FindNextFileA(hFind, &stFindData);
	}

	if(hFind != INVALID_HANDLE_VALUE) {
		FindClose(hFind);
	}

	return err;
}


std::string stringify_currentTime(bool file_mode) {
	struct _timeb t;
	_ftime(&t);
	std::istringstream istrTime(ctime(&t.time));
	std::string sWeekDay, sMonth;
	int iDay, iHour, iMinutes, iSeconds, iYear;
	char cSColmn;
	istrTime >> sWeekDay >> sMonth >> iDay >> iHour >> cSColmn >> iMinutes >> cSColmn >> iSeconds >> iYear;
	std::ostringstream ostr;
	ostr.fill('0');
	if(file_mode) {
		ostr << std::setw(2) << iYear << sMonth << iDay << '_' << iHour << std::setw(2) << iMinutes << std::setw(2) << iSeconds << '_' << std::setw(3) << (int)t.millitm;
	}
	else {
		ostr << std::setw(2) << iHour << ':' << std::setw(2) << iMinutes << ':' << std::setw(2) << iSeconds << '_' << std::setw(3) << (int)t.millitm;
	}

	return ostr.str();
}



void VS_FileLog(const std::string& msg, bool do_close, bool do_ipclog) {
	if(do_ipclog) {
		IPCLog(msg);
	}

	if(g_configuration._file_log) {
		static wsi_gate gate;
		static HANDLE handle_file = (HANDLE)-1;
		gate.lock();
		if(!do_close && !g_bTerminated && handle_file == (HANDLE)-1) {
			handle_file = OSWRONLYOpenFile("VSStatistics" + stringify_currentTime(true) + ".csv");
		}
		if(handle_file != (HANDLE)-1) {
			OSAppendStringToFile(handle_file, msg);
		}
		if(do_close && handle_file != (HANDLE)-1) {
			OSCloseFile(handle_file);
			handle_file = (HANDLE)-1;
		}
		gate.unlock();
	}
}



void ARFF_FileLog(const std::string& msg, bool do_close, bool do_ipclog) {
	if(do_ipclog) {
		IPCLog(msg);
	}

	static wsi_gate gate;
	static HANDLE handle_file = (HANDLE)-1;

	gate.lock();
	if(!do_close && !g_bTerminated && handle_file == (HANDLE)-1) {
		handle_file = OSWRONLYOpenFile("Shapes_" + stringify_currentTime(true) + ".arff");
		if(handle_file != (HANDLE)-1) {
			std::ostringstream ostr; 
			ostr << "@relation shape" << std::endl;
			ostr << "@attribute isValid { 'Y', 'N'}" << std::endl;
			ostr << "@attribute isRectangle { 'Y', 'N'}" << std::endl;
			//ostr << "@attribute intensity numeric" << std::endl;
			//ostr << "@attribute intensity_atcenter numeric" << std::endl;
			ostr << "@attribute shapemeasure numeric" << std::endl;
			ostr << "@attribute effective_flattening numeric" << std::endl;
			ostr << "@attribute flattening numeric" << std::endl;
			//ostr << "@attribute skewness numeric" << std::endl;
			ostr << "@attribute covar numeric" << std::endl;
			ostr << "@attribute hull_circularity numeric" << std::endl;
			ostr << "@attribute intensity_upperquantile numeric" << std::endl;
			//ostr << "@attribute contour_area numeric" << std::endl;
			ostr << "@attribute contour_area2hull_area numeric" << std::endl;
			ostr << "@attribute centers_distance numeric" << std::endl;
			//ostr << "@attribute var1 numeric" << std::endl;
			ostr << "@attribute form_factor numeric" << std::endl;
			ostr << "@data" << std::endl;
			OSAppendStringToFile(handle_file, ostr.str());
		}
	}
	if(handle_file != (HANDLE)-1) {
		OSAppendStringToFile(handle_file, msg);
	}
	if(do_close && handle_file != (HANDLE)-1) {
		OSCloseFile(handle_file);
		handle_file = (HANDLE)-1;
	}
	gate.unlock();
}

void ClusteredPoint::ARFF_Output(bool isValid, bool rectangleDetected) {
	std::ostringstream ostr;
	ostr << (isValid? 'Y': 'N');
	ostr << ',' << (rectangleDetected? 'Y': 'N');
	//ostr << ',' << (_intensity / g_bytedepth_scalefactor);
	//ostr << ',' << (_intensity_atcenter / g_bytedepth_scalefactor);
	ostr << ',' << _shapemeasure;
	ostr << ',' << _effective_flattening;
	ostr << ',' << _flattening;
	//ostr << ',' << _skewness;
	ostr << ',' << _covar;
	ostr << ',' << _hull_circularity;
	ostr << ',' << _intensity_upperquantile;
	//ostr << ',' << _contour_area;
	ostr << ',' << _contour_area2hull_area;
	ostr << ',' << _centers_distance;
	//ostr << ',' << _skewness * (_intensity / g_bytedepth_scalefactor);
	ostr << ',' << _hull_circularity * (1 - _flattening) * _intensity_upperquantile; 
	ostr << std::endl; 

	ARFF_FileLog(ostr.str());
}

