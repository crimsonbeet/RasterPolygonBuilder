
#ifndef XClasses_wsiclassesH
#define XClasses_wsiclassesH

#include "stdafx.h"


#ifdef WSI_NAMESPACE
#undef WSI_NAMESPACE
#endif
#ifdef autocreateserialization_not_necessaryH
#undef autocreateserialization_not_necessaryH
#endif

#define WSI_NAMESPACE XClasses



#include "WSIClassFactory.h"





struct VSPoint { 
	double _xyz[3]; 
	VSPoint() {
		memset(_xyz, 0, sizeof(_xyz));
	}
	VSPoint(double x, double y, double z) {
		_xyz[0] = x;
		_xyz[1] = y;
		_xyz[2] = z;
	}
}; 

BEGIN_WSI_SERIALIZATION_OBJECT(VSPoint)
CONTAINS_FLAT_MEMBER(_xyz, P)
END_WSI_SERIALIZATION_OBJECT()





struct VSPrincipalDirection_FitLine {
	std::vector<VSPoint> _input_points; // in; empty in response 
	VSPoint _output_direction; // out; on return it is a normalized vector pointing along the line. 
	int _ok; // out

	VSPrincipalDirection_FitLine() {
		_ok = 0;
	}
};


BEGIN_WSI_SERIALIZATION_OBJECT(VSPrincipalDirection_FitLine)
CONTAINS_FLAT_MEMBER(_ok, Ok)
CONTAINS_OBJECT_MEMBER(_input_points, P)
CONTAINS_OBJECT_MEMBER(_output_direction, D)
END_WSI_SERIALIZATION_OBJECT()


AUTOCREATE_WSI_SERIALIZATION_OBJECT(VSPrincipalDirection_FitLine)

VSPrincipalDirection_FitLine* VSFitLine(VSPrincipalDirection_FitLine *obj);




#endif //XClasses_wsiclassesH
