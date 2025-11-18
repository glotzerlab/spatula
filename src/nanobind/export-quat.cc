#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h> // For std::vector
#include <nanobind/stl/pair.h>   // For std::pair
#include <nanobind/make_iterator.h> // For nb::make_iterator
#include <sstream>
#include <string>

#include "../data/Quaternion.h"

namespace nb = nanobind;

namespace spatula { namespace data {

void bind_quaternion(nb::module_ &m) {
    nb::class_<Quaternion>(m, "Quaternion")
        .def(nb::init<double, double, double, double>())
        .def_rw("w", &Quaternion::w)
        .def_rw("x", &Quaternion::x)
        .def_rw("y", &Quaternion::y)
        .def_rw("z", &Quaternion::z)
        .def("__repr__",
             [](const Quaternion& q) {
                 auto repr = std::ostringstream();
                 repr << "Quaternion(" << std::to_string(q.w) << ", " << std::to_string(q.x) << ", "
                      << std::to_string(q.y) << ", " << std::to_string(q.z) << ")";
                 return repr.str();
             })
        .def("conjugate", &Quaternion::conjugate)
        .def("to_axis_angle", &Quaternion::to_axis_angle)
        .def("to_axis_angle_3D", &Quaternion::to_axis_angle_3D)
        .def("norm", &Quaternion::norm)
        .def("normalize", &Quaternion::normalize)
        .def("to_rotation_matrix", &Quaternion::to_rotation_matrix)
        .def(nb::self * nb::self)
        .def(nb::self *= nb::self)
        .def_static(
            "from_object",
            [](nb::object obj) {
                // Use nanobind's sequence protocol
                nb::sequence seq = nb::cast<nb::sequence>(obj);
                if (nb::len(seq) < 4) {
                    throw nb::type_error("Quaternion object requires a 4 length sequence like object.");
                }
                return Quaternion(nb::cast<double>(seq[0]), nb::cast<double>(seq[1]), nb::cast<double>(seq[2]), nb::cast<double>(seq[3]));
            },
            "Create a Quaternion from a 4-element sequence (w, x, y, z).");
}
}} // namespace spatula::data
