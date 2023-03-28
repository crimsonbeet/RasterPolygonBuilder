
#ifndef AndroidCamera_wsiclassesH
#define AndroidCamera_wsiclassesH

#if defined(_MSC_VER)
#include "stdafx.h"
#endif


#ifdef WSI_NAMESPACE
#undef WSI_NAMESPACE
#endif
#ifdef autocreateserialization_not_necessaryH
#undef autocreateserialization_not_necessaryH
#endif

#define WSI_NAMESPACE XAndroidCamera



#include "WSIClassFactory.h"


struct AndroidBytesContainer {
	std::vector<uint8_t> _buffer; // out
};

BEGIN_WSI_SERIALIZATION_OBJECT(AndroidBytesContainer)
CONTAINS_FLAT_MEMBER(_buffer, B)
END_WSI_SERIALIZATION_OBJECT()




struct AndroidCameraImageMetadata {
	int _ok; // out

    int _isFirst; // in

	int64_t _timestamp; // in
	std::string _cameraId; // in

    int _width;
    int _height;

	AndroidCameraImageMetadata() {
		_ok = 0;
        _isFirst = 0;
		_timestamp = 0;
        _width = 0;
        _height = 0;
	}
};

BEGIN_WSI_SERIALIZATION_OBJECT(AndroidCameraImageMetadata)
CONTAINS_FLAT_MEMBER(_isFirst, First)
CONTAINS_FLAT_MEMBER(_timestamp, T)
CONTAINS_FLAT_MEMBER(_cameraId, Camera)
CONTAINS_FLAT_MEMBER(_width, W)
CONTAINS_FLAT_MEMBER(_height, H)
END_WSI_SERIALIZATION_OBJECT()








struct AndroidBayerFilterImage : AndroidCameraImageMetadata {
public:
    BayerFilterContainer _image[2]; // in

    std::string _filterArrangment; // in

    void Init(const size_t N) {
        int j = 0;
        for(auto& buf: _image) {
            buf._buffer.resize(N/2);
            buffersData[j++] = buf._buffer.data();
        }
    }

    void Init() {
        int j = 0;
        for(auto& buf: _image) {
            buffersData[j++] = buf._buffer.data();
        }
    }



    void put_image(const int16_t* image, const size_t N) {
        Init(N);
        const int16_t* dataEnd = image + N;
        while (image < dataEnd) {
            for (auto& buf : buffersData) {
                *buf++ = *image++;
            }
        }
    }



    void put_image(std::vector<int16_t>& image) {
        put_image(image.data(), image.size());
    }



    void get_image(std::vector<int16_t>& image) {
        assert(_image[0]._buffer.size() == _image[1]._buffer.size());
        image.resize(_image[0]._buffer.size() * 2);
        Init();
        int16_t* dataPos = image.data();
        int16_t* dataEnd = dataPos + image.size();
        while (dataPos < dataEnd) {
            for (auto& buf : buffersData) {
                *dataPos++ = *buf++;
            }
        }
    }

protected:
    int16_t* buffersData[ARRAY_NUM_ELEMENTS(_image)];
};


BEGIN_WSI_INHERITED_SERIALIZATION_OBJECT(AndroidCameraImageMetadata, AndroidBayerFilterImage)
CONTAINS_FLAT_MEMBER(_filterArrangment, F)
CONTAINS_FLAT_MEMBER(_image, I)
END_WSI_SERIALIZATION_OBJECT()


AUTOCREATE_WSI_SERIALIZATION_OBJECT(AndroidBayerFilterImage)

AndroidBayerFilterImage* Process_BayerFilterImage(AndroidBayerFilterImage* obj);
void Process_CameraBayerFilterImage(AndroidBayerFilterImage* obj);











struct AndroidCameraJpegImage : AndroidCameraImageMetadata {
    JpegImageContainer _cont;

    void put_image(const uint8_t* image, const size_t N) {
        _cont._ptr = image;
        _cont._size = N;
    }
};


BEGIN_WSI_INHERITED_SERIALIZATION_OBJECT(AndroidCameraImageMetadata, AndroidCameraJpegImage)
CONTAINS_FLAT_MEMBER(_cont, C)
END_WSI_SERIALIZATION_OBJECT()


AUTOCREATE_WSI_SERIALIZATION_OBJECT(AndroidCameraJpegImage)

AndroidCameraJpegImage* Process_JpegImage(AndroidCameraJpegImage* obj);
void Process_CameraJpegImage(AndroidCameraJpegImage* obj);











struct AndroidCameraRaw10Image : AndroidCameraImageMetadata {
	uint8_t _diffs[5] = {0};

    std::string _filterArrangment; // in

    AndroidBytesContainer _buffers[5]; // in
};


BEGIN_WSI_INHERITED_SERIALIZATION_OBJECT(AndroidCameraImageMetadata, AndroidCameraRaw10Image)
CONTAINS_FLAT_MEMBER(_filterArrangment, F)
CONTAINS_OBJECT_MEMBER(_buffers, BB)
END_WSI_SERIALIZATION_OBJECT()


AUTOCREATE_WSI_SERIALIZATION_OBJECT(AndroidCameraRaw10Image)

AndroidCameraRaw10Image* Process_Raw10Image(AndroidCameraRaw10Image* obj);




struct AndroidCaptureStillImageRequest {
    int64_t _timestamp = 0;
};


BEGIN_WSI_SERIALIZATION_OBJECT(AndroidCaptureStillImageRequest)
    CONTAINS_FLAT_MEMBER(_timestamp, T)
END_WSI_SERIALIZATION_OBJECT()


AUTOCREATE_WSI_SERIALIZATION_OBJECT(AndroidCaptureStillImageRequest)

void Process_AndroidCaptureStillImageRequest(AndroidCaptureStillImageRequest* obj);




#endif //AndroidCamera_wsiclassesH
