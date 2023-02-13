#include "stdafx.h"

HWND  _g_hGLWindow = NULL;
HDC _g_hGLDConext;
HGLRC _g_hGLRContext;



GLdouble near_plane = 0.01;
GLdouble far_plane = 300;




BOOL glSetupPixelFormat(HDC hdc) {
	PIXELFORMATDESCRIPTOR pfd, *ppfd;
	int pixelformat;

	ppfd = &pfd;

	ppfd->nSize = sizeof(PIXELFORMATDESCRIPTOR);
	ppfd->nVersion = 1;
	ppfd->dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	ppfd->dwLayerMask = PFD_MAIN_PLANE;
	ppfd->iPixelType = PFD_TYPE_COLORINDEX;
	ppfd->cColorBits = 8;
	ppfd->cDepthBits = 16;
	ppfd->cAccumBits = 0;
	ppfd->cStencilBits = 0;

	pixelformat = ChoosePixelFormat(hdc, ppfd);

	if((pixelformat = ChoosePixelFormat(hdc, ppfd)) == 0) {
		MessageBox(NULL, "ChoosePixelFormat failed", "Error", MB_OK);
		return FALSE;
	}

	if(SetPixelFormat(hdc, pixelformat, ppfd) == FALSE) {
		MessageBox(NULL, "SetPixelFormat failed", "Error", MB_OK);
		return FALSE;
	}

	return TRUE;
}

/* OpenGL code */

GLvoid glResize(GLsizei width, GLsizei height) {
	GLfloat aspect;

	glViewport(0, 0, width, height);

	aspect = (GLfloat)width / height;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(75.0, aspect, near_plane, far_plane);
	glMatrixMode(GL_MODELVIEW);
}

int object_index = 0;

GLfloat light_ambient[] = {0.2f, 0.2f, 0.2f, 1.0f};
GLfloat light_diffuse[] = {1.0f, 1.0f, 1.0f, 1.0f};
GLfloat light_specular[] = {0.0f, 0.0f, 0.0f, 1.0f};

GLfloat mat_specular[] = {0.0f, 0.0f, 0.0f, 1.0f};
GLfloat mat_diffuse[] = {0.8f, 0.6f, 0.4f, 1.0f};
GLfloat mat_ambient[] = {0.8f, 0.6f, 0.4f, 1.0f};
GLfloat mat_shininess = {20.0f};

GLvoid initializeGL(GLsizei width, GLsizei height) {
	GLfloat maxObjectSize = 3.0;
	GLfloat aspect = (GLfloat)width / height;

	glClearDepth(1.0);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);

	glResize(width, height);

	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);

	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
	glMaterialf(GL_FRONT, GL_SHININESS, mat_shininess);
	glShadeModel(GL_SMOOTH); /*enable smooth shading */
	glEnable(GL_LIGHTING); /* enable lighting */
	glEnable(GL_LIGHT0); /* enable light 0 */
}





Point3d cross3d(Point3d src[2]); 
double dot3d(const Point3d& src0, const Point3d& src1);





GLvoid startScene() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

GLvoid drawScene(std::vector<Mat_<double>>& points4D, const std::vector<bool>& isACoordinatePoint, std::vector<int>& labels, const std::vector<std::vector<double>>& colors, std::vector<std::vector<Mat_<double>>>& coordlines4D) {
	std::vector<int>::iterator it = labels.begin();
	std::vector<std::vector<double>>::const_iterator it_colors = colors.begin();
	std::vector<bool>::const_iterator it_isACoordinatePoint = isACoordinatePoint.cbegin();

	GLdouble xAvg = 0;
	GLdouble yAvg = 0;
	GLdouble zAvg = 0;

	for(auto& point : points4D) {
		GLdouble x = point(0);
		GLdouble y = point(1);
		GLdouble z = point(2);

		GLfloat c[4] = {1, 1, 1, 1};

		float radius = 0.1f; 
		if(it_isACoordinatePoint != isACoordinatePoint.cend()) {
			if(*it_isACoordinatePoint) {
				radius = 0.15f; 
			}
			++it_isACoordinatePoint;
		}

		if(it != labels.end()) {
			++it;
		}

		if (it_colors != colors.end()) {
			for (int j = 0; j < 3; ++j) c[j] = (*it_colors)[j];
			++it_colors; 
		}

		GLUquadricObj *quadObj;

		glPushMatrix();

		glNewList(++object_index, GL_COMPILE);
		quadObj = gluNewQuadric();
		gluQuadricDrawStyle(quadObj, GLU_FILL);
		gluSphere(quadObj, radius, 10, 10);
		glEndList();

		glTranslated(x, y, z);
		glMaterialfv(GL_FRONT, GL_AMBIENT, c);
		glCallList(object_index);

		glPopMatrix();
		gluDeleteQuadric(quadObj);

		xAvg += x;
		yAvg += y;
		zAvg += z;
	}

	size_t nline = 0; 

	for(auto& axis : coordlines4D) {
		// rotation from z to the vector. 
		Point3d points[2] = {Point3d(0, 0, 1), Point3d(axis[1](0) - axis[0](0), axis[1](1) - axis[0](1), axis[1](2) - axis[0](2))};
		Point3d& z = points[0];
		Point3d& d = points[1];

		GLUquadricObj *quadObj;

		glPushMatrix();

		glNewList(++object_index, GL_COMPILE);

		glPushMatrix();
		z = cross3d(points);
		glRotated(acos(d.z / sqrt(dot3d(d, d))) * 180 / M_PI, z.x, z.y, z.z);
		quadObj = gluNewQuadric();
		gluQuadricDrawStyle(quadObj, GLU_FILL);
		gluCylinder(quadObj, 0.1, 0.1, sqrt(dot3d(d, d)), 12, 2);
		glPopMatrix();
		glEndList();

		glTranslated(axis[0](0), axis[0](1), axis[0](2));

		GLfloat c[4] = {1, 0, 0, 1};

		switch(nline) {
		case 0:
		c[0] = 1; c[1] = 0; c[2] = 0;
		break;
		case 1:
		c[0] = 0; c[1] = 1; c[2] = 0;
		break;
		case 2:
		c[0] = 0; c[1] = 0; c[2] = 1;
		break;
		default:
		for (int j = 2, k = 0; j >= 0; --j, ++k) c[k] = colors[nline - 3][j];
		break;
		}

		++nline;

		glMaterialfv(GL_FRONT, GL_AMBIENT, c);
		glCallList(object_index);

		glPopMatrix();
		gluDeleteQuadric(quadObj);
	}

	xAvg /= points4D.size();
	yAvg /= points4D.size();
	zAvg /= points4D.size();

	glLoadIdentity();
	gluLookAt(0, 0, 0, xAvg, yAvg, zAvg, 0, 1, 0);
}

GLvoid commitScene() {
	SwapBuffers(_g_hGLDConext);
	if(object_index) {
		glDeleteLists(1, object_index);
		GLenum glErr = glGetError();
		if(glErr != GL_INVALID_VALUE && glErr != GL_INVALID_OPERATION) {
			object_index = 0;
		}
	}
}


