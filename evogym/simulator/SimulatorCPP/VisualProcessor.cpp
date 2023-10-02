//
// Created by Lycolet on 2023/07/12.
//

#include "VisualProcessor.h"


VisualProcessor::VisualProcessor(
        int _vis_type,
        Sim* sim,
        double _vis_lim_len,
        int _vis2_resolution
        ){
    VisualProcessor::vis_type = _vis_type;
    VisualProcessor::vis_lim_len = _vis_lim_len;
    VisualProcessor::vis2_resolution = _vis2_resolution;

    VisualProcessor::objects = sim->environment.get_objects();
    VisualProcessor::edges = sim->environment.get_edges();
    VisualProcessor::pos = sim->environment.get_pos();
}

VisualProcessor::~VisualProcessor() {

}

//void VisualProcessor::init(
//        int _vis_type,
//        double _vis_lim_len,
//        vector<SimObject *> *_objects,
//        vector <Edge> *_edges,
//        Matrix<double, 2, Dynamic>* _pos,
//        int _vis2_resolution
//        ) {
//
//    VisualProcessor::vis_type = _vis_type;
//    VisualProcessor::vis_lim_len = _vis_lim_len;
//    VisualProcessor::vis2_resolution = _vis2_resolution;
//    VisualProcessor::objects = _objects;
//    VisualProcessor::edges = _edges;
//    VisualProcessor::pos = _pos;
//}


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
    vis_voxel_points.clear();
//    vis_robot_own_idc.clear();

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
                        vis_voxel_points.push_back(v.points);
                        vis_surfaces_edge.push_back(e_idx);
//                        vis_robot_own_idc.push_back(s_idx);
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
    vis2_types.clear();
    vis2_sqr_dists.clear();

    // 視覚距離2乗
    double sqr_l = vis_lim_len * vis_lim_len;

    // 視覚ボクセル表面単位で視野を生成
    for (int i = 0; i < num_vis_surfaces; i++) {
        // 生成した視野を一次的に格納するバッファ
        VectorXi types_buf;
        VectorXd sqr_dists_buf;
        types_buf.resize(vis2_resolution);
        sqr_dists_buf.resize(vis2_resolution);

        // 視覚ボクセル辺端点など(a->b->cがleft turn)
        auto a = (*pos).col(vis_surfaces_edge_a[i]);
        auto b = (*pos).col(vis_surfaces_edge_b[i]);
        auto e_idx = vis_surfaces_edge[i];
        auto c = (*pos)(Eigen::all, vis_voxel_points[i]).rowwise().sum() / 4.0f;

        // 全てのボクセル辺に対する処理
        for (int k = 0; k < edges->size(); k++){
            // [E1]: 始点となる視覚ボクセル辺は除外
            if (k == e_idx) { continue; }

            auto& e = (*edges)[k];  // 観測対象の辺
            // ボクセル表面辺以外は除外
            if (!e.isOnSurface) { continue; }

            // 処理対象辺の端点
            auto e1 = (*pos).col(e.a_index);
            auto e2 = (*pos).col(e.b_index);

            // [E2]: 処理対象辺eの端点の内少なくとも片方が視野角の範囲に入っていることを確認
            if((calc_determinant(c, a, e1) >= 0 && calc_determinant(c, b, e1) <= 0) ||
                    (calc_determinant(c, a, e2) >= 0 && calc_determinant(c, b, e2) <= 0)){

                auto diff_e1_c = e1 - c;
                auto diff_e2_c = e2 - c;
                double sqr_re1 = diff_e1_c.squaredNorm();
                double sqr_re2 = diff_e2_c.squaredNorm();

                // [E3]: 処理対象辺eの端点の内片方が視覚距離範囲内
                if (sqr_re1 <= sqr_l || sqr_re2 <= sqr_l){

                    // [E4]: 各辺の端点をcを中心として極座標変換
                    //      {(sqr_re1^(1/2), theta_e1), (sqr_re2^(1/2), theta_e2)}
                    double theta_e1 = atan2(diff_e1_c.y(), diff_e1_c.x());
                    double theta_e2 = atan2(diff_e2_c.y(), diff_e2_c.x());

                    auto diff_ep1_c = a - c;
                    auto diff_ep2_c = b - c;
                    double theta_ep1 = atan2(diff_ep1_c.y(), diff_ep1_c.x());
                    double theta_ep2 = atan2(diff_ep2_c.y(), diff_ep2_c.x());

                    // [E5]: 処理対象辺e端点の角度を正規化：視覚ボクセル辺端点aと同じなら0, bと同じなら1
                    double theta_e1_nrm = (theta_e1 - theta_ep1) / (theta_ep2 - theta_ep1);
                    double theta_e2_nrm = (theta_e2 - theta_ep1) / (theta_ep2 - theta_ep1);

                    // sampling

                }

            }

        }
    }
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

// vis2
vector<VectorXi>* VisualProcessor::get_vis2_types(){
    return &vis2_types;
}

vector<VectorXd>* VisualProcessor::get_vis2_sqr_depths(){
    return &vis2_sqr_dists;
}
