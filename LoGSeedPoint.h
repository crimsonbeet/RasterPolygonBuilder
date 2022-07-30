
#ifndef LoGSeedPointH
#define LoGSeedPointH


struct ImageScaleFactors {
	double fx = 0;
	double fy = 0;
};

struct MouseCallbackParameters {
	ImageScaleFactors scaleFactors;
	int windowNumber = 0;
};

struct LoGSeedPoint {
	MouseCallbackParameters params;
	int eventValue = 0;
	int x;
	int y;
};

extern LoGSeedPoint g_LoG_seedPoint;
extern int g_LoG_imageWindowNumber;



#endif // LoGSeedPointH