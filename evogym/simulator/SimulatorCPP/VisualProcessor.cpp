//
// Created by Lycolet on 2023/07/12.
//

#include "VisualProcessor.h"


VisualProcessor::VisualProcessor(){

}

VisualProcessor::~VisualProcessor() {

}

void VisualProcessor::init(
        int vis_type,
        double vis_lim_len,
        vector<SimObject *> *objects,
        vector <Edge> *edges,
        Matrix<double, 2, Dynamic>* pos
//        ,bool is_enable_render
        ) {

    VisualProcessor::vis_type = vis_type;
    VisualProcessor::vis_lim_len = vis_lim_len;
    VisualProcessor::objects = objects;
    VisualProcessor::edges = edges;
    VisualProcessor::pos = pos;
//    VisualProcessor::is_enable_renderer = is_enable_render;
}


// functions
double VisualProcessor::calc_determinant(Vector2d pi, Vector2d pj, Vector2d pk) {
    return (pj(0) * pk(1)
            + pk(0) * pi(1)
            + pi(0) * pj(1)
            - pj(0) * pi(1)
            - pi(0) * pk(1)
            - pk(0) * pj(1));
}

// internal methods
void VisualProcessor::init_surface_to_type()
{
    surfaces_to_type.clear();

    for(const auto& s: *objects){
        for(const auto& v: s->boxels){
            for(int e_idx: v.edges){
                auto& e = (*edges)[e_idx];
                if (e.isOnSurface){
                    surfaces_to_type[e_idx] = v.cell_type;
                }
            }
        }
    }
}

void VisualProcessor::update_vis_surfaces()
{
    num_vis_surfaces = 0;
    vis_surfaces_edge.clear();
    vis_surfaces_edge_a.clear();
    vis_surfaces_edge_b.clear();
    vis_robot_own_idc.clear();

    for (int s_idx = 0; s_idx < objects->size(); s_idx++){
        auto& s = (*objects)[s_idx];
        for (const auto& v: s->boxels){
            if (v.cell_type == CELL_VIS){
                for (int e_idx: v.edges){
                    auto& e = (*edges)[e_idx];
                    if (e.isOnSurface){
                        auto a = pos->col(e.a_index);
                        auto b = pos->col(e.b_index);
                        auto c = (*pos)(Eigen::all, v.points).rowwise().sum() / 4.0f;  // 視覚ボクセル重心

                        if (calc_determinant(a, b, c) >= 0){
                            // left turn
                            vis_surfaces_edge_a.push_back(e.a_index);
                            vis_surfaces_edge_b.push_back(e.b_index);
                        }
                        else{
                            // right turn
                            vis_surfaces_edge_a.push_back(e.b_index);
                            vis_surfaces_edge_b.push_back(e.a_index);
                        }
                        vis_surfaces_edge.push_back(e_idx);
                        vis_robot_own_idc.push_back(s_idx);
                        num_vis_surfaces++;
                    }
                }
            }
        }
    }
}

void VisualProcessor::update_vis1()
{
    vis1_types.clear();
    vis1_sqr_dists.clear();
    vis1_endpoints_a.clear();
    vis1_endpoints_b.clear();

    for (int i = 0; i < num_vis_surfaces; i++){
        int min_d_e_idx = -1;
        Vector2d min_d_ep;
        double min_sq_depth = 1000;

        auto a = (*pos).col(vis_surfaces_edge_a[i]);
        auto b = (*pos).col(vis_surfaces_edge_b[i]);
        auto e_idx = vis_surfaces_edge[i];

        Vector2d n(b(1) - a(1), a(0) - b(0));  // 視覚ボクセル辺法線ベクトル
        Vector2d m = (a + b) / 2.0;  // 視覚ボクセル辺中点
        Vector2d v1 = m;  // 視線始点
        Vector2d v2 = n.normalized() * vis_lim_len + m;  // 視線終点

        for (int k = 0; k < edges->size(); k++){  // 自身以外のボクセル辺に対する処理
            if (k == e_idx) { continue; }
            auto& e = (*edges)[k];  // 観測対象の辺
            if (!e.isOnSurface) { continue; }

            auto e1 = (*pos).col(e.a_index);
            auto e2 = (*pos).col(e.b_index);
            if (calc_determinant(v1, v2, e1) * calc_determinant(v1, v2, e2) < 0 &&
                calc_determinant(e1, e2, v1) * calc_determinant(e1, e2, v2) < 0){

                Vector3d p1(1.0, v1(0), v1(1));
                Vector3d p2(1.0, v2(0), v2(1));
                Vector3d p3(1.0, e1(0), e1(1));
                Vector3d p4(1.0, e2(0), e2(1));

                auto cp = p1.cross(p2).cross(p3.cross(p4));
                auto cp_x = cp(1) / cp(0);
                auto cp_y = cp(2) / cp(0);

                auto tmp_sq_depth =
                        (v1(0) - cp_x) * (v1(0) - cp_x) +
                        (v1(1) - cp_y) * (v1(1) - cp_y);
                if (min_d_e_idx == -1 || (min_sq_depth > tmp_sq_depth)){
                    min_d_e_idx = k;
                    min_sq_depth = tmp_sq_depth;
                    min_d_ep << cp_x, cp_y;
                }
            }
        }

        if (min_d_e_idx != -1){
            vis1_types.push_back(surfaces_to_type.at(min_d_e_idx));
            vis1_sqr_dists.push_back(min_sq_depth);
            vis1_endpoints_a.push_back(v1);
            vis1_endpoints_b.push_back(min_d_ep);
        }
        else{
            vis1_types.push_back(-1);
            vis1_sqr_dists.push_back(vis_lim_len * vis_lim_len);
            vis1_endpoints_a.push_back(v1);
            vis1_endpoints_b.push_back(v2);
        }
    }
}

void VisualProcessor::update_vis2()
{
    // unimplemented
}

// public methods
void VisualProcessor::update_configuration()
{
    init_surface_to_type();
    update_vis_surfaces();
}

void VisualProcessor::update_for_timestep()
{
    if (vis_type == 1){
        update_vis1();
    }
    else if(vis_type == 2){
        update_vis2();
    }
    else{
        cout << "[Error] unexpected vis_type:" << vis_type << endl;
    }
}

//void VisualProcessor::render(Camera camera)
//{
//    if (is_enable_renderer){
//        if (vis_type == 1){
//            render_vis1(camera);
//        }
//        else if(vis_type == 2){
//            render_vis2(camera);
//        }
//        else{
//            cout << "[Error] unexpected vis_type:" << vis_type << endl;
//        }
//    }
//
//}

//void VisualProcessor::render_vis1(Camera camera) {
//    for (int i = 0; i < vis_type->size(); i++) {
//        glBegin(GL_LINES);
//
//        glColor3f(0.3, 0.3, 1);
//        auto start = camera.world_to_camera(vis_endpoint_a->at(i));
//        auto end = camera.world_to_camera(vis_endpoint_b->at(i));
//        glVertex2f(start.x(), start.y());
//        glVertex2f(end.x(), end.y());
//
//        glEnd();
//    }
//}

//void VisualProcessor::render_vis2(Camera camera) {
//    // unimplemented
//}

// getter
// vis1
vector<int>* VisualProcessor::get_vis1_types()
{
    return &vis1_types;
}

vector<double>* VisualProcessor::get_vis1_sqr_depths()
{
    return &vis1_sqr_dists;
}

vector<vector<Vector2d>*> VisualProcessor::get_vis1_endpoints() {
    vector<vector<Vector2d>*> vis1_endpoints;
    vis1_endpoints.push_back(&vis1_endpoints_a);
    vis1_endpoints.push_back(&vis1_endpoints_b);

    return vis1_endpoints;
}

const int &VisualProcessor::get_vis_type() {
    return VisualProcessor::vis_type;
}

