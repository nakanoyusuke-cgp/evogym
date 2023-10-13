//
// Created by Lycolet on 2023/07/12.
//

#ifndef SIMULATOR_CPP_VISUALPROCESSOR_H
#define SIMULATOR_CPP_VISUALPROCESSOR_H

#include "main.h"
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <Eigen/Core>

#include "SimObject.h"
#include "Edge.h"
#include "Camera.h"
#include "Sim.h"

using namespace std;
using namespace Eigen;


class VisualProcessor {
private:
    // hyper params
    int vis_type{};
    double vis_lim_len{};
    int vis2_resolution{};
//    bool is_enable_renderer;

    // environment reference
    vector<SimObject*>* objects{};
    vector<Edge>* edges{};
    Matrix<double, 2, Dynamic>* pos{};

    // common
    map<int, int> surfaces_to_type;
    int num_vis_surfaces{};
    vector<int> vis_surfaces_edge;
    vector<int> vis_surfaces_edge_a;
    vector<int> vis_surfaces_edge_b;
    vector<Vector4f> vis_voxel_points;
//    vector<int> vis_robot_own_idc;

    // vis1
    vector<int> vis1_types;
    vector<double> vis1_sqr_dists;
    vector<Vector2d> vis1_endpoints_a;
    vector<Vector2d> vis1_endpoints_b;
//    vector<vector<Vector2d>> vis1_endpoints;

    // vis2
    vector<VectorXi> vis2_types;
    vector<VectorXd> vis2_sqr_dists;

    // functions
    static double calc_determinant(Vector2d pi, Vector2d pj, Vector2d pk);

    // internal methods
    void init_surface_to_type();
    void update_vis_surfaces();
    void update_vis1();
    void update_vis2();

public:
    VisualProcessor(int _vis_type, Sim* sim, double _vis_lim_len, int _vis2_resolution);
    ~VisualProcessor();

    void update_configuration();
    void update_for_timestep();

    // getter
    int get_num_vis_surfaces();

    // vis1
    const int &get_vis_type();
    vector<int>* get_vis1_types();
    vector<double>* get_vis1_sqr_depths();
    vector<vector<Vector2d>*> get_vis1_endpoints();

    // vis2
    vector<VectorXi>* get_vis2_types();
    vector<VectorXd>* get_vis2_sqr_depths();
};


#endif //SIMULATOR_CPP_VISUALPROCESSOR_H
