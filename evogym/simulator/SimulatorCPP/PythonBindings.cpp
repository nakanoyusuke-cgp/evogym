#include "main.h"

#include "Interface.h"
#include "Sim.h"
#include "Camera.h"
#include "VisualProcessor.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(simulator_cpp, m) {

	m.doc() = "A module for Robot API in c++";

	py::class_<Interface>(m, "Viewer")
		.def(py::init<Sim*>())
		.def("init", &Interface::init)
		.def("render", &Interface::render, py::arg("camera"), py::arg("hide_background") = false, py::arg("hide_grid") = false, py::arg("hide_edges") = false, py::arg("hide_boxels") = false, py::arg("dont_clear") = false)
		.def("show_debug_window", &Interface::show_debug_window)
		.def("hide_debug_window", &Interface::hide_debug_window)
		.def("get_debug_window_pos", &Interface::get_debug_window_pos, py::return_value_policy::copy)
		.def("set_vis_proc", &Interface::set_vis_proc, py::arg("vis_proc"));

	py::class_<Sim>(m, "Sim")
		.def(py::init<>())
		.def("init", &Sim::init)
		.def("get_version", &Sim::get_version)
		.def("read_object_from_file", &Sim::read_object_from_file)
		.def("read_robot_from_file", &Sim::read_robot_from_file)
		.def("read_object_from_array", &Sim::read_object_from_array)
		.def("read_robot_from_array", &Sim::read_robot_from_array)
		.def("step", &Sim::step)
		.def("set_action", &Sim::set_action)
		.def("revert", &Sim::revert)
		.def("get_time", &Sim::get_time)
		.def("pos_at_time", &Sim::pos_at_time, py::return_value_policy::reference_internal)
		.def("vel_at_time", &Sim::vel_at_time, py::return_value_policy::reference_internal)
		.def("object_pos_at_time", &Sim::object_pos_at_time, py::return_value_policy::reference_internal)
		.def("object_vel_at_time", &Sim::object_vel_at_time, py::return_value_policy::reference_internal)
		.def("object_orientation_at_time", &Sim::object_orientation_at_time)
		.def("translate_object", &Sim::translate_object)
		.def("move_object", &Sim::move_object)
		.def("get_indices_of_actuators", &Sim::get_actuator_indices, py::return_value_policy::reference_internal)
        .def("object_boxels_pos", &Sim::object_boxels_pos, py::return_value_policy::copy)
        .def("object_boxels_type", &Sim::object_boxels_type, py::return_value_policy::copy)
        .def("object_boxels_pos_eigen", &Sim::object_boxels_pos_eigen, "")
        .def("add_object_velocity", &Sim::add_object_velocity, "")
        .def("mul_object_velocity", &Sim::mul_object_velocity, "")
        .def("set_object_velocity", &Sim::set_object_velocity, "")
        .def("get_surface_edges", &Sim::get_surface_edges, "")
        .def("ground_on_robot", &Sim::ground_on_robot, "")
//        .def("get_vis1_types", &Sim::get_vis1_types, "")
//        .def("get_vis1_sqr_dists", &Sim::get_vis1_sqr_dists, "")
        ;

	py::class_<Camera>(m, "Camera")
		.def(py::init<bool>())
		.def("set_pos", static_cast<void (Camera::*)(double, double)>(&Camera::set_pos), py::arg("x"), py::arg("y"))
		.def("set_size", static_cast<void (Camera::*)(double, double)>(&Camera::set_size), py::arg("x"), py::arg("y"))
		.def("set_resolution", static_cast<void (Camera::*)(double, double)>(&Camera::set_resolution), py::arg("x"), py::arg("y"))
		.def("get_resolution_width", &Camera::get_resolution_width, py::return_value_policy::reference)
		.def("get_resolution_height", &Camera::get_resolution_height, py::return_value_policy::reference)
		.def("get_image", [](Camera& c) {
			return py::memoryview::from_memory(
				c.get_image_data_ptr(),						// buffer pointer
				c.get_image_data_size()						// buffer size
			);
		});

    py::class_<VisualProcessor>(m, "VisualProcessor")
        .def(py::init<int, Sim*, double, int>(), py::arg("vis_type"), py::arg("sim"), py::arg("vis_lim_len"), py::arg("vis2_resolution"))
        .def("update_configuration", &VisualProcessor::update_configuration, "")
        .def("update_for_timestep", &VisualProcessor::update_for_timestep, "")
        .def("get_vis_type", &VisualProcessor::get_vis_type, "")
        .def("get_vis1_types", &VisualProcessor::get_vis1_types, "")
        .def("get_vis1_sqr_depths", &VisualProcessor::get_vis1_sqr_depths, "")
        .def("get_vis2_types", &VisualProcessor::get_vis2_types, "")
        .def("get_vis2_sqr_depths", &VisualProcessor::get_vis2_sqr_depths, "")
        .def("get_num_vis_surfaces", &VisualProcessor::get_num_vis_surfaces, "")
        ;
}

