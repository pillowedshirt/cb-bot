#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
namespace py = pybind11;

static std::vector<double> finite_values(py::array_t<double, py::array::c_style | py::array::forcecast> a) {
    auto r = a.unchecked<1>(); std::vector<double> v; v.reserve(r.shape(0));
    for (ssize_t i=0;i<r.shape(0);++i) if (std::isfinite(r(i))) v.push_back(r(i));
    return v;
}

py::dict evaluate_agent_thresholds(py::array_t<double, py::array::c_style | py::array::forcecast> scores, py::array_t<double, py::array::c_style | py::array::forcecast> outcomes, py::array_t<double, py::array::c_style | py::array::forcecast> adverse, py::array_t<double, py::array::c_style | py::array::forcecast> thresholds) {
    auto s=scores.unchecked<1>(), o=outcomes.unchecked<1>(), a=adverse.unchecked<1>(), t=thresholds.unchecked<1>();
    py::list rows; ssize_t n = std::min(s.shape(0), o.shape(0));
    for (ssize_t j=0;j<t.shape(0);++j) {
        double th=t(j), sum=0.0, advsum=0.0; int count=0, wins=0; std::vector<double> vals;
        for (ssize_t i=0;i<n;++i) if (std::isfinite(s(i)) && s(i) >= th && std::isfinite(o(i))) { double v=o(i); vals.push_back(v); sum += v; wins += v > 0.0; if (i < a.shape(0) && std::isfinite(a(i))) advsum += std::abs(a(i)); count++; }
        py::dict row; row["threshold"]=th; row["selected"]=count;
        if (!count) { row["win_rate"]=0.0; row["avg_net"]=0.0; row["median"]=0.0; row["avg_adverse"]=0.0; }
        else { std::sort(vals.begin(), vals.end()); double med = vals[count/2]; if (count % 2 == 0) med = (vals[count/2-1] + vals[count/2]) / 2.0; row["win_rate"]=(double)wins/count; row["avg_net"]=sum/count; row["median"]=med; row["avg_adverse"]=advsum/count; }
        rows.append(row);
    }
    py::dict out; out["rows"] = rows; return out;
}

py::dict simulate_weighted_council(py::array_t<double, py::array::c_style | py::array::forcecast> score_matrix, py::array_t<double, py::array::c_style | py::array::forcecast> weights, py::array_t<double, py::array::c_style | py::array::forcecast> outcomes, double threshold) { py::dict d; d["selected"]=0; d["avg_net"]=0.0; return d; }
py::dict score_sell_paths(py::array_t<double> realized, py::array_t<double> max_fav, py::array_t<double> max_adv, py::array_t<double> move_after_sell, py::array_t<double> held_minutes) { py::dict d; d["rows"] = py::list(); return d; }
py::dict monte_carlo_equity_paths(py::array_t<double> returns_bps, double position_fraction, int paths, int horizon) { py::dict d; d["paths"]=paths; d["horizon"]=horizon; return d; }
PYBIND11_MODULE(fast_institutional_core, m) { m.def("evaluate_agent_thresholds", &evaluate_agent_thresholds); m.def("simulate_weighted_council", &simulate_weighted_council); m.def("score_sell_paths", &score_sell_paths); m.def("monte_carlo_equity_paths", &monte_carlo_equity_paths); }
