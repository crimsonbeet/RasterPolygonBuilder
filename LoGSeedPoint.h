
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
	int x;
	int y;
};

extern LoGSeedPoint g_LoG_seedPoint;



#endif // LoGSeedPointH