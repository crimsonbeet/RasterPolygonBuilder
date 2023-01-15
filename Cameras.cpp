#include "stdafx.h"


#include "OleDate.h" 


void InitializeCameras(SImageAcquisitionCtl& ctl) {
	CBaslerUsbInstantCameraArray& cameras = *ctl._cameras;

	cameras.StopGrabbing();
	cameras.Close();

	g_actionDeviceKey = rand();
	g_actionGroupKey = rand();

	CTlFactory& tlFactory = CTlFactory::GetInstance();

	DeviceInfoList_t basler_list;
	tlFactory.EnumerateDevices(basler_list, true);

	int cameras_found = 0;

	for (auto it = basler_list.begin(); it != basler_list.end(); ++it) {
		auto full_name = std::string((*it).GetFullName());
		auto user_name = std::string((*it).GetSerialNumber());

		std::cout << "available device " << full_name << std::endl;
		std::cout << "Name " << user_name << std::endl;

		for (int j = 0; j < ARRAY_NUM_ELEMENTS(ctl._camera_serialnumbers); ++j) {
			if (user_name == ctl._camera_serialnumbers[j]) {
				auto dev = tlFactory.CreateDevice(*it); 
				cameras[j].Attach(dev);
				cameras[j].SetCameraContext(j); // CameraContext is attached to the GrabResult

				++cameras_found;
			}
		}
	}

	if (cameras_found < ARRAY_NUM_ELEMENTS(ctl._camera_serialnumbers)) {
		std::stringstream ostr;
		for (int j = 0; j < ARRAY_NUM_ELEMENTS(ctl._camera_serialnumbers); ++j) {
			if (std::string(cameras[j].GetDeviceInfo().GetSerialNumber()) != ctl._camera_serialnumbers[j]) {
				ostr << " No device " << ctl._camera_serialnumbers[j] << " found" << std::endl;
			}
		}
		g_bCamerasAreOk = false;
	}
	else {
		g_bCamerasAreOk = true;
		for (int j = 0; j < ARRAY_NUM_ELEMENTS(ctl._camera_serialnumbers); ++j) {
			std::cout << "Using device " << std::string(cameras[j].GetDeviceInfo().GetFullName()) << std::endl;
			std::cout << "S/N " << std::string(cameras[j].GetDeviceInfo().GetSerialNumber()) << std::endl;
		}
	}
}


void ParameterizeCameras(SImageAcquisitionCtl& ctl, bool setTriggerMode) {
	CBaslerUsbInstantCameraArray& cameras = *ctl._cameras;
	try {
		for (int j = 0; j < cameras.GetSize(); ++j) {
			if (ctl._exposure_times[j] <= 0) {
				cameras[j].ExposureAuto = Basler_UsbCameraParams::ExposureAuto_Continuous;
			}
			else {
				cameras[j].ExposureAuto = Basler_UsbCameraParams::ExposureAuto_Off;
				cameras[j].ExposureTime = ctl._exposure_times[j];
			}

			if (ctl._use_trigger) {
				cameras[j].TriggerSelector = Basler_UsbCameraParams::TriggerSelector_FrameStart;

				std::string model_name = cameras[j].GetDeviceInfo().GetModelName();

				cameras[j].TriggerSource = ctl._trigger_source_software ? Basler_UsbCameraParams::TriggerSource_Software : Basler_UsbCameraParams::TriggerSource_Line1;
				cameras[j].AcquisitionStatusSelector = Basler_UsbCameraParams::AcquisitionStatusSelector_FrameTriggerWait;
			}
			else {
				cameras[j].TriggerSelector = Basler_UsbCameraParams::TriggerSelector_FrameStart;
				cameras[j].TriggerSource = Basler_UsbCameraParams::TriggerSource_Software;
				cameras[j].AcquisitionStatusSelector = Basler_UsbCameraParams::AcquisitionStatusSelector_FrameTriggerWait;
			}

			if (setTriggerMode) {
				cameras[j].TriggerMode = ctl._use_trigger ? Basler_UsbCameraParams::TriggerMode_On : Basler_UsbCameraParams::TriggerMode_Off;
				if (ctl._use_trigger) {
				}
				else {
					cameras[j].AcquisitionStart();
				}
			}

			g_frameRate[j] = cameras[j].ResultingFrameRate.GetValue();
		}
		g_resultingFrameRate = 50;
		for (int j = 0; j < (int)cameras.GetSize(); ++j) {
			if (g_frameRate[j] < g_resultingFrameRate) {
				g_resultingFrameRate = g_frameRate[j];
			}
		}
	}
	catch (GenICam::GenericException& e) {
		std::cerr << "An exception occurred: " << e.GetDescription() << std::endl;
		g_bTerminated = true;
	}
}




void OpenCameras(SImageAcquisitionCtl& ctl) {
	CBaslerUsbInstantCameraArray& cameras = *ctl._cameras;
	cameras.StopGrabbing();
	cameras.Close();

	try {
		cameras.Open();
		for (int j = 0; j < cameras.GetSize(); ++j) {
			if (ctl._use_trigger && ctl._trigger_source_software) {
				cameras[j].OutputQueueSize = 1;
			}

			//cameras[j].BinningHorizontal = 1;
			//cameras[j].BinningVertical = 1;

			std::string model_name = cameras[j].GetDeviceInfo().GetModelName();

			int pixelsHoriz = 1280;
			int pixelsVert = 960;

			//cameras[j].OffsetX.SetValue(0);
			//cameras[j].Width.SetValue(pixelsHoriz);
			//cameras[j].CenterX = true;

			if (ctl._image_height > pixelsVert) {
				ctl._image_height = pixelsVert;
			}

			//cameras[j].OffsetY.SetValue(0);
			//cameras[j].Height.SetValue(ctl._image_height);
			//cameras[j].OffsetY.SetValue((pixelsVert - ctl._image_height) / 2);
			//cameras[j].CenterY = true;

			if (ctl._12bit_format) {
				cameras[j].PixelFormat = Basler_UsbCameraParams::PixelFormat_Mono12;
			}
			else {
				cameras[j].PixelFormat = Basler_UsbCameraParams::PixelFormat_BGR8;
			}

			cameras[j].ExposureMode.SetValue(Basler_UsbCameraParams::ExposureMode_Timed);

			cameras[j].BlackLevelSelector.SetValue(Basler_UsbCameraParams::BlackLevelSelector_All);
			cameras[j].BlackLevel.SetValue(1);

			cameras[j].BalanceWhiteAuto.SetValue(Basler_UsbCameraParams::BalanceWhiteAuto_Continuous);

			cameras[j].AcquisitionBurstFrameCount = 1;

			int framesRate = 60;

			cameras[j].AcquisitionFrameRateEnable = true;
			cameras[j].AcquisitionFrameRate = framesRate; // times a second

			cameras[j].AcquisitionMode = Basler_UsbCameraParams::AcquisitionMode_Continuous;
		}

		ParameterizeCameras(ctl, /*setTriggerMode*/true); // do the dynamic part of the parameters. 

		cameras.StartGrabbing(ctl._use_trigger ? GrabStrategy_LatestImageOnly : GrabStrategy_OneByOne, GrabLoop_ProvidedByUser);
	}
	catch (GenICam::GenericException& e) {
		std::cerr << "An exception occurred: " << e.GetDescription() << std::endl;
		g_bTerminated = true;
	}
}
