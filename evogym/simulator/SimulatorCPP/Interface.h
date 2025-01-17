#ifndef INTERFACE_H
#define INTERFACE_H

#include "main.h"

//#include "GL/glew.h"
//#include "glfw3.h"
//#include "GL/gl.h"
//
//#include <vector>
//#include <Eigen/Dense>

#include "Sim.h"
#include "Environment.h"
#include "SimObject.h"
#include "Camera.h"
#include "VisualProcessor.h"

#include "Edge.h"

using namespace std;
using namespace Eigen;

class Interface
{
private:

	//RENDERING
	GLFWwindow* debug_window;
	bool debug_window_showing;
	Vector2d last;

	//DATA
	Matrix <double, 2, Dynamic>* pos;
	vector <Edge>* edges;
	vector <SimObject*>* objects;

	vector <bool>* point_is_colliding;

    // - vision perception
    vector <int>* vis1_cell_types;
    vector <Vector2d>* vis1_endpoint_a;
    vector <Vector2d>* vis1_endpoint_b;
    VisualProcessor* visualProcessor;
    bool _has_vis_proc;

	//COLORS
	struct color_byte {

		GLubyte r;
		GLubyte g;
		GLubyte b;

		color_byte(GLubyte ra, GLubyte ga, GLubyte ba) : r(ra), g(ga), b(ba) {}
	};

	color_byte get_encoded_color(int cell_type);
	
	void render_edges(Camera camera);	
	void render_edge_normals(Camera camera);
	void render_points(Camera camera);
	void render_object_points(Camera camera);
	void render_bounding_boxes(Camera camera);
	void render_boxels(Camera camera);
	void render_grid(Camera camera);
	void render_encoded_boxels(Camera camera);
    void render_vis_lines(Camera camera);

    vector<double> get_vis_color(int cell_type);

public:

	Interface(Sim* sim);
	~Interface();

	static void init();
	void render(Camera camera, bool hide_background = false, bool hide_grid = false, bool hide_edges = false, bool hide_boxels = false, bool dont_clear = false);

	void show_debug_window();
	void hide_debug_window();
	vector<int> get_debug_window_pos();

	GLFWwindow* get_debug_window_ref();

    void set_vis_proc(VisualProcessor* vis_proc);
    bool has_vis_proc();
};

#endif // !INTERFACE_H

