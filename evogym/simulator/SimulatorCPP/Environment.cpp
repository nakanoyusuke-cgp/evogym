#include "Environment.h"

#include <fstream>
#include <iostream>
#include "Boxel.h"

Environment::Environment(){

}

void Environment::init(){

	//PHYSICS
	physics_handler = PhysicsEngine();
	physics_handler.init(
		&points_pos, &points_pos_last, &points_vel, &points_vel_true, &points_mass, &points_fixed,
		&a_index, &b_index, &length_eq, &length_eq_goal, &init_length_eq, &spring_const,
		&edges, &objects);
	physics_handler.point_is_colliding = &point_is_colliding;

	num_points = 0;
}

void Environment::set_surface_edge_color(int color) {
	
	for (int i = 0; i < objects.size(); i++) {
		for (int j = 0; j < objects[i]->surface_edges_index.cols(); j++) {
			edges[objects[i]->surface_edges_index[j]].color = color;
			edges[objects[i]->surface_edges_index[j]].isOnSurface = true;
		}
	}
}

//int Environment::createPoint(Vector2d_old pos, double mass) {
//
//	points_pos_old.push_back(pos);
//	points_vel_old.push_back(Vector2d_old(0, 0));
//	points_mass_old.push_back(mass);
//	points_fixed_old.push_back(false);
//
//	return points_pos_old.size() - 1;
//}

void Environment::create_points(vector <Vector2d>* pos, vector <Vector2d>* vel, vector <double>* mass, vector <bool>* fixed) {
	
	int old_size = num_points;
	int new_size = pos->size() + old_size;

	points_pos.conservativeResize(2, new_size);
	points_pos_last.conservativeResize(2, new_size);
	points_vel.conservativeResize(2, new_size);
	points_vel_true.conservativeResize(2, new_size);
	points_mass.conservativeResize(2, new_size);
	points_fixed.conservativeResize(2, new_size);
	
	for (int i = 0; i < pos->size(); i++) {
		points_pos.col(i + old_size) = pos->at(i);
		points_pos_last .col(i + old_size) = pos->at(i);
		points_vel.col(i + old_size) = vel->at(i);
		points_vel_true.col(i + old_size) = vel->at(i);
		points_mass.col(i + old_size) = Vector2d(mass->at(i), mass->at(i));
		points_fixed(0, i + old_size) = fixed->at(i);
		points_fixed(1, i + old_size) = fixed->at(i);
	}

	// for (int i = 0; i < points_pos.rows(); i++){
	// 	for(int j = 0; j < points_pos.cols(); j++){
	// 		if (points_fixed(i, j)){
	// 			continue;
	// 		}
		
	// 		double randn = ((double)rand() / (RAND_MAX))/ 1e7;
	// 		for (int f = 0; f < 12; f++){
	// 			randn = 2.0 * ((double)rand() / (RAND_MAX)) / 1e7;
	// 			randn = randn - 1e-7;
	// 		}
		
	// 		double old = points_pos(i, j); 
	// 		points_pos(i, j) = points_pos(i, j) + randn;
	// 		// cout << i << " " << j << " " << old << " " << randn << " " << points_pos(i, j) << "\n"; 
	// 	}
	// }
	

	num_points = new_size; 
}

void Environment::create_edges(vector <Edge>* new_edges) {

	int old_size = a_index.cols();
	int new_size = new_edges->size() + old_size;

	a_index.conservativeResize(1, new_size);
	b_index.conservativeResize(1, new_size);
	length_eq.conservativeResize(1, new_size);
	length_eq_goal.conservativeResize(1, new_size);
	init_length_eq.conservativeResize(1, new_size);
	spring_const.conservativeResize(1, new_size);

	for (int i = 0; i < new_edges->size(); i++) {
		edges.push_back(new_edges->at(i));
		a_index[i + old_size] = new_edges->at(i).a_index;
		b_index[i + old_size] = new_edges->at(i).b_index;
		length_eq[i + old_size] = new_edges->at(i).init_length_eq;
		length_eq_goal[i + old_size] = new_edges->at(i).init_length_eq;
		init_length_eq[i + old_size] = new_edges->at(i).init_length_eq;
		spring_const[i + old_size] = new_edges->at(i).spring_const;
	}
}

void Environment::swap_edge(int edge_index) {
	edges[edge_index].a_index = b_index[edge_index];
	edges[edge_index].b_index = a_index[edge_index];
	a_index[edge_index] = edges[edge_index].a_index;
	b_index[edge_index] = edges[edge_index].b_index;
}

//int Environment::create_edge(Edge edge) {
//
//	edges.push_back(edge);
//	return edges.size() - 1;
//}

void Environment::init_robot(string robot_name) {
	
	if (object_name_to_index.count(robot_name) <= 0)
	{
		cout << "Error: No robot named " << robot_name << ".\n";
		return;
	}
	if (!objects[object_name_to_index[robot_name]]->is_robot)
	{
		cout << "Error: " << robot_name << " is not robot.\n";
		return;
	}

	((Robot*)objects[object_name_to_index[robot_name]])->init();
}

bool Environment::add_object_name(string object_name, int index) {

	if (object_name_to_index.count(object_name) > 0)
		return false;

	object_name_to_index[object_name] = index;
	return true;
}

bool Environment::special_step() {
	physics_handler.resolve_edge_constraints();
	return false;
}
void Environment::print_poses(){
	cout << "pos\nnp.array([";
	for (int i = 0; i < points_pos.cols(); i++){
		cout << "[" << points_pos(0, i) << ", " << points_pos(1, i) << "],"; 
	}
	cout << "])\n\n";
	cout << "vel\nnp.array([";
	for (int i = 0; i < points_vel.cols(); i++){
		cout << "[" << points_vel(0, i) << ", " << points_vel(1, i) << "],"; 
	}
	cout << "])";
	cout << "\n\n\n\n";
}
bool Environment::step() {


	// for (int i = 0; i < points_pos.rows(); i++){
	// 	for(int j = 0; j < points_pos.cols(); j++){
	// 		if (points_fixed(i, j)){
	// 			continue;
	// 		}
			
	// 		double randn = ((double)rand() / (RAND_MAX))/ 1e7;
	// 		for (int f = 0; f < 28; f++)
	// 			randn = ((double)rand() / (RAND_MAX))/ 1e7; 
			
	// 		double old = points_pos(i, j); 
	// 		points_pos(i, j) = points_pos(i, j) + randn;
	// 		// cout << i << " " << j << " " << old << " " << randn << " " << points_pos(i, j) << "\n"; 
	// 	}
	// }
	// cout << "\n\n\n\n";

	// print_poses();
	physics_handler.resolve_collisions();
	physics_handler.update_actuators();
	// print_poses();
	physics_handler.step_rk4(0.0001);
	// print_poses();
	
	//physics_handler.step_rk4(0.00000005);
	//physics_handler.step_rk4(0.00001);
	physics_handler.resolve_edge_constraints();
	physics_handler.update_true_vel(0.0001);




	return physics_handler.is_any_robot_self_colliding();

	//physics_handler.step_euler(0.0000005);

}

void Environment::set_robot_action(string robot_name, Ref <Matrix <double, Dynamic, 2>> action) {

	if (object_name_to_index.count(robot_name) <= 0)
	{
		cout << "Error: No robot named " << robot_name << ".\n";
		return;
	}
	if (!objects[object_name_to_index[robot_name]]->is_robot)
	{
		cout << "Error: " << robot_name << " is not robot.\n";
		return;
	}

	int robot_index = object_name_to_index[robot_name];

	vector <Boxel*> actuators;
	vector <double> actuations;

	for (int i = 0; i < action.rows(); i++) {

		if (((Robot*)objects[robot_index])->get_actuator(action(i, 0)) == NULL) {
			cout << "Error: No actuator found at index " << action(i, 0) << " of robot " << robot_name << ".\n";
			return;
		}

		actuators.push_back(((Robot*)objects[robot_index])->get_actuator(action(i, 0)));
		actuations.push_back(action(i, 1));
	}

	physics_handler.update_actuator_goals(&actuators, &actuations);
}

void Environment::save_snapshot(long int sim_time) {

	/*if (history.count(sim_time) > 0)
		return;*/

	history[sim_time] = Snapshot(points_pos, points_vel, sim_time);
}

bool Environment::revert_to_snapshot(long int sim_time) {

	if (history.count(sim_time) <= 0) {
		cout << "Error: Could not revert simulation - no save found for time argument.\n";
		return false;
	}
		
	points_pos = history[sim_time].points_pos;
	points_vel = history[sim_time].points_vel;

	points_pos_last = points_pos;
	points_vel_true = points_vel;

	history.erase(next(history.find(sim_time), 1), history.end());

	return true;
}

Ref <MatrixXd> Environment::get_pos_at_time(long int sim_time) {
	
	if (history.count(sim_time) <= 0) {
		Matrix <double, 2, Dynamic> empty;
		empty.resize(2, 4);
		empty << 0,0,0,0,0,0,0,0;
		return empty;
	}

	return history[sim_time].points_pos;
}
Ref <MatrixXd> Environment::get_vel_at_time(long int sim_time) {

	if (history.count(sim_time) <= 0) {
		Matrix <double, 2, Dynamic> empty;
		empty.resize(2, 4);
		empty << 0, 0, 0, 0, 0, 0, 0, 0;
		return empty;
	}

	return history[sim_time].points_vel;
}

Ref <MatrixXd> Environment::object_pos_at_time(long int sim_time, string object_name) {

	if (history.count(sim_time) <= 0 || object_name_to_index.count(object_name) <= 0) {
		Matrix <double, 2, Dynamic> empty;
		empty.resize(2, 4);
		empty << 0, 0, 0, 0, 0, 0, 0, 0;
		return empty;
	}

	int object_index = object_name_to_index[object_name];
	int min_index = objects[object_index]->min_point_index;
	int max_index = objects[object_index]->max_point_index;
	return history[sim_time].points_pos(Eigen::all, Eigen::seq(min_index, max_index));

}
Ref <MatrixXd> Environment::object_vel_at_time(long int sim_time, string object_name) {

	if (history.count(sim_time) <= 0 || object_name_to_index.count(object_name) <= 0) {
		Matrix <double, 2, Dynamic> empty;
		empty.resize(2, 4);
		empty << 0, 0, 0, 0, 0, 0, 0, 0;
		return empty;
	}

	int object_index = object_name_to_index[object_name];
	int min_index = objects[object_index]->min_point_index;
	int max_index = objects[object_index]->max_point_index;
	return history[sim_time].points_vel(Eigen::all, Eigen::seq(min_index, max_index));

}double Environment::object_orientation_at_time(long int sim_time, string object_name) {

	if (history.count(sim_time) <= 0 || object_name_to_index.count(object_name) <= 0) {
		return 0;
	}

	int object_index = object_name_to_index[object_name];
	int min_index = objects[object_index]->min_point_index;
	int max_index = objects[object_index]->max_point_index;

	Matrix <double, 2, Dynamic> pos_new = history[sim_time].points_pos(Eigen::all, Eigen::seq(min_index, max_index));
	Matrix <double, 2, Dynamic> pos_old = history[0].points_pos(Eigen::all, Eigen::seq(min_index, max_index));

	//cout << "pos_new\n" << pos_new << "\n\n";
	//cout << "pos_old\n" << pos_old << "\n\n";

	Vector2d pos_new_com = pos_new.rowwise().sum() / pos_new.cols();
	Vector2d pos_old_com = pos_old.rowwise().sum() / pos_old.cols();

	//cout << "pos_new_com\n" << pos_new_com << "\n\n";
	//cout << "pos_old_com\n" << pos_old_com << "\n\n";

	pos_new = pos_new.colwise() - pos_new_com;
	pos_old = pos_old.colwise() - pos_old_com;

	//cout << "pos_new centered\n" << pos_new << "\n\n";
	//cout << "pos_old centered\n" << pos_old << "\n\n";

	Array <double, 1, Dynamic> mags_new = pos_new.colwise().norm();
	Array <double, 1, Dynamic> mags_old = pos_old.colwise().norm();

	pos_new = pos_new.colwise().normalized();
	pos_old = pos_old.colwise().normalized();

	//cout << "pos_new centered normalized\n" << pos_new << "\n\n";
	//cout << "pos_old centered normalized\n" << pos_old << "\n\n";

	Array <double, 1, Dynamic> angles = (pos_new.array() * pos_old.array()).colwise().sum();
	//cout << "dot\n" << angles << "\n\n";

	//cout << "full dot\n" << angles << "\n\n";
	angles = ((mags_new * mags_old) < 0.0000001).select(0.99999, angles);
	angles = (angles > 1.0).select(0.99999, angles);
	angles = (angles < -1.0).select(-0.99999, angles);

	//cout << "full dot after correction\n" << angles << "\n\n";

	//cout << "dot\n" << angles(Eigen::all, Eigen::seq(0, 16)) << "\n\n";
	angles = angles.acos();
	//cout << "acos\n" << angles << "\n\n";
	angles = angles * mags_old;

	double pi = 3.1415926;
	double average_angle = (angles.sum() / mags_old.sum());

	Array <double, 1, Dynamic> crosses = (pos_new(0, Eigen::all).array() * pos_old(1, Eigen::all).array()) - (pos_new(1, Eigen::all).array() * pos_old(0, Eigen::all).array());

	double cross = crosses.sum();

	if (cross > 0)
		average_angle = 2 * pi - average_angle;

	return average_angle;
}
void Environment::translate_object(double x, double y, string object_name) {
	
	if (object_name_to_index.count(object_name) <= 0)
		return;

	int object_index = object_name_to_index[object_name];
	int min_index = objects[object_index]->min_point_index;
	int max_index = objects[object_index]->max_point_index;

	Vector2d dx = Vector2d(x, y);
	points_pos(Eigen::all, Eigen::seq(min_index, max_index)).colwise() += dx;
}


Matrix<double, 2, Dynamic>* Environment::get_pos() {
	return &points_pos;
}
Matrix <double, 2, Dynamic>* Environment::get_vel() {
	return &points_vel;
}
Matrix <double, 2, Dynamic>* Environment::get_mass() {
	return &points_mass;
}
Matrix <bool, 2, Dynamic>* Environment::get_fixed() {
	return &points_fixed;
}

int Environment::get_num_points() {
	return num_points;
}
int Environment::get_num_edges() {
	return edges.size();
}

vector<Edge>* Environment::get_edges() {
	return &edges;
}
vector<SimObject*>* Environment::get_objects() {
	return &objects;
}
Robot* Environment::get_robot(string robot_name) {

	if (object_name_to_index.count(robot_name) <= 0)
	{
		cout << "Error: No robot named " << robot_name << ".\n";
		return NULL;
	}
	if (!objects[object_name_to_index[robot_name]]->is_robot)
	{
		cout << "Error: " << robot_name << " is not robot.\n";
		return NULL;
	}
	//return robots[robot_name_to_index[robot_name]];
	return (Robot*)(objects[object_name_to_index[robot_name]]);
}

//vector<Vector2d_old>* Environment::getPointsPos() {
//	return &points_pos_old;
//}
//vector<bool>* Environment::getPointsFixed() {
//	return &points_fixed_old;
//}
//vector<double>* Environment::getMasses() {
//	return &points_mass_old;
//}

SimObject* Environment::get_object(string object_name){
    if (object_name_to_index.count(object_name) <= 0){
        cout << "Error: No object named " << object_name << ".\n";
        return nullptr;
    }
    return objects[object_name_to_index[object_name]];
}

void Environment::add_object_velocity(double x, double y, string object_name) {
    if (object_name_to_index.count(object_name) <= 0)
        return;

    int object_index = object_name_to_index[object_name];
    int min_index = objects[object_index]->min_point_index;
    int max_index = objects[object_index]->max_point_index;

    Vector2d dx = Vector2d(x, y);
    points_vel(Eigen::all, Eigen::seq(min_index, max_index)).colwise() += dx;
}

void Environment::mul_object_velocity(double m, string object_name) {
    if (object_name_to_index.count(object_name) <= 0)
        return;

    int object_index = object_name_to_index[object_name];
    int min_index = objects[object_index]->min_point_index;
    int max_index = objects[object_index]->max_point_index;

//    Vector2d dx = Vector2d(m, m);

//    auto vel = points_vel(Eigen::all, Eigen::seq(min_index, max_index)).colwise();

//    std::cout << points_vel(Eigen::all, Eigen::seq(min_index, max_index)) << std::endl;
    points_vel(Eigen::all, Eigen::seq(min_index, max_index)) *= m;
}

void Environment::set_object_velocity(double x, double y, string object_name){
    if (object_name_to_index.count(object_name) <= 0)
        return;

    int object_index = object_name_to_index[object_name];
    int min_index = objects[object_index]->min_point_index;
    int max_index = objects[object_index]->max_point_index;

    Vector2d v = Vector2d(x, y);
    points_vel(Eigen::all, Eigen::seq(min_index, max_index)).colwise() = v;
}

Environment::~Environment(){

}

//Matrix<int, 2, Dynamic> Environment::get_surface_edges(string object_name) {
py::array_t<int> Environment::get_surface_edges(string object_name) {
    if (object_name_to_index.count(object_name) <= 0){
//        Matrix<int, 2, Dynamic> empty;
//        empty.resize(2, 1);
//        empty << -1, -1;
        py::array_t<int> empty({2, 1});
        *empty.mutable_data(0, 0) = 0;
        *empty.mutable_data(1, 0) = 0;
        return empty;
    }

    SimObject* simObject = get_object(object_name);
    Matrix<double, 1, Dynamic> surface_edge_idc = simObject->surface_edges_index;
    int n_surface_edges = (int)surface_edge_idc.size();

//    py::array_t<double> empty({2, 2, n_surface_edges});
    py::array_t<int> result({2, n_surface_edges});

    for (int i = 0; i < n_surface_edges; i++){
        Edge &e = edges.at(i);
        *result.mutable_data(0, i) = e.a_index;
        *result.mutable_data(1, i) = e.b_index;
    }

    return result;
}

py::tuple Environment::get_vis_type1(string object_name) {
    // return shape : (resolution, num_vis_surfaces)
    if (object_name_to_index.count(object_name) <= 0){
		std::cout << "object named: " << object_name << "dose not exist" << std::endl;
		return py::make_tuple(0, 0.0);
    }

    //
    int num_vis_surfaces = 0;
	int num_vis_voxels;
    int num_voxels = (int)obj->boxels.size();
    SimObject* obj = environment.get_object(std::move(object_name));

	for (int i = 0; i < num_voxels; i++){
		Boxel* voxel = obj->boxels[i];
		if (voxel.cell_type == CELL_VIS){
			for (int j = 0; j < 4; j++){
				Edge* e = edges(voxel->edges(j));
                if (e->isOnSurface){
                    auto bl = (*posptr).col(voxel.point_bot_left_index);
                    auto br = (*posptr).col(voxel.point_bot_right_index);
                    auto tl = (*posptr).col(voxel.point_top_left_index);
                    auto tr = (*posptr).col(voxel.point_top_right_index);
                    auto c = (bl + br + tl + tr) / 4;
                    auto p1 = (*points_pos).col(e->a_index);
                    auto p2 = (*points_pos).col(e->b_index);
                    // D(p1, p2, p3) > 0: left turn
                    if (is_left_turn(p1, p2, c) == false){
                        p1 = (*points_pos).col(e->b_index);
                        p2 = (*points_pos).col(e->a_index);
                    }

                }
			}
		}
	}

    for (int i = 0; i < num_vis_surfaces; i++){

    }
}

py::tuple Environment::get_vis_type2(string object_name, int resolution) {
    // return shape : (resolution, num_vis_surfaces)
    if (object_name_to_index.count(object_name) <= 0){
        py::array_t<int> empty({2, 1});
        *empty.mutable_data(0, 0) = 0;
        *empty.mutable_data(1, 0) = 0;
        return empty;
    }
	py::array_t<int> empty({2, 1});
	*empty.mutable_data(0, 0) = 0;
	*empty.mutable_data(1, 0) = 0;
	return empty;
}

double Environment::ground_on_robot(string above, string under) {
    if (object_name_to_index.count(above) <= 0 || object_name_to_index.count(under) <= 0){
        return -1.0;
    }

    SimObject* objAbove = get_object(above);
    SimObject* objUnder = get_object(under);

    Matrix<double, 1, Dynamic> &above_surface_edge_idc = objAbove->surface_edges_index;
//    map<int, int> &above_surface_edge_dir = objAbove->surface_edge_directions;
    Matrix<double, 1, Dynamic> &above_surface_point_idc = objAbove->surface_points_index;
    Matrix<double, 1, Dynamic> &under_surface_edge_idc = objUnder->surface_edges_index;
//    map<int, int> &under_surface_edge_dir = objUnder->surface_edge_directions;
//    Matrix<double, 1, Dynamic> &under_surface_point_idc = objUnder->surface_points_index;

//    int n_above_surface_edges = (int)above_surface_edge_idc.size();
    int n_above_surface_points = (int)above_surface_point_idc.size();
    int n_under_surface_edges = (int)under_surface_edge_idc.size();
//    int n_under_surface_points = (int)under_surface_point_idc.size();

    double result_min = 100.0;

    for (int i = 0; i < n_above_surface_points; i++){
        auto p = points_pos(all, (int)above_surface_point_idc(i));
        for (int j = 0; j < n_under_surface_edges; j++){
            auto &s = edges.at((int)under_surface_edge_idc(j));
            auto a = points_pos(all, s.a_index);
            auto b = points_pos(all, s.b_index);
            if (((a(0) - p(0)) * (b(0) - p(0))) <= 0.0) {
                // pxが線分ax-bxを内分する
                auto div = b(0) - a(0);
                if (div > 1e-8){
                    // 0割りにならない場合
                    auto hy = (((b(0)-p(0))*a(1)+(p(0)-a(0))*b(1))/div);
                    double len_pl = p(1) - hy;
                    if (len_pl >= 0.0 && len_pl < result_min){
                        result_min = len_pl;
                    }
                }
                else{
                    // 0割りになる場合
                    if (a(1) > b(1)){
                        double len_pl = p(1) - a(1);
                        if (len_pl >= 0.0 && len_pl < result_min){
                            result_min = len_pl;
                        }
                    }
                    else{
                        double len_pl = p(1) - b(1);
                        if (len_pl >= 0.0 && len_pl < result_min){
                            result_min = len_pl;
                        }
                    }
                }
            }
        }
    }
    return result_min;
}

bool Environment::is_left_turn(int pi, int pj, int pk) {
    return (pj(0) * pk(1)
          + pk(0) * pi(1)
          + pi(0) * pj(1)
          - pj(0) * pi(1)
          - pi(0) * pk(1)
          - pk(0) * pj(1)) > 0
}

