#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

static double median_from_values(std::vector<double> values) {
    values.erase(std::remove_if(values.begin(), values.end(), [](double v) { return !std::isfinite(v); }), values.end());
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    const size_t mid = n / 2;
    if (n % 2 == 1) return values[mid];
    return (values[mid - 1] + values[mid]) * 0.5;
}

template <typename Getter>
static double median_from_indices(const std::vector<int>& indices, Getter getter) {
    std::vector<double> values;
    values.reserve(indices.size());
    for (int idx : indices) {
        double value = getter(idx);
        if (std::isfinite(value)) values.push_back(value);
    }
    return median_from_values(values);
}

static py::dict evaluate_outcome_arrays(double entry_price, py::array_t<double, py::array::c_style | py::array::forcecast> highs, py::array_t<double, py::array::c_style | py::array::forcecast> lows, double target_bps, double cost_bps, double min_net_gain_bps, double bar_minutes, double max_adverse_before_profit_bps) {
    auto h = highs.unchecked<1>();
    auto l = lows.unchecked<1>();
    const ssize_t n = h.shape(0);
    if (entry_price <= 0.0 || n <= 0 || l.shape(0) != n) {
        py::dict out;
        out["max_favorable_bps"] = 0.0; out["max_adverse_bps"] = 0.0; out["reached_min_profit"] = false; out["reached_target"] = false; out["win_bps"] = 0.0; out["loss_bps"] = 0.0; out["time_to_min_profit_bars"] = -1; out["time_to_min_profit_minutes"] = 0.0; out["forward_window_minutes"] = 0.0; out["post_profit_max_favorable_bps"] = 0.0; out["post_profit_extra_gain_bps"] = 0.0; out["adverse_before_profit_bps"] = 0.0; out["survived_to_profit"] = false;
        return out;
    }
    const double required_profit_bps = cost_bps + min_net_gain_bps;
    double max_high = 0.0, min_low = std::numeric_limits<double>::infinity(), max_favorable_bps = 0.0, max_adverse_bps = 0.0;
    int time_to_min_profit_bars = -1, profit_hit_index = -1;
    double time_to_min_profit_minutes = 0.0, adverse_before_profit_bps = 0.0, low_before_profit = entry_price;
    for (ssize_t idx = 0; idx < n; ++idx) {
        const double high = h(idx), low = l(idx);
        if (high <= 0.0 || low <= 0.0 || !std::isfinite(high) || !std::isfinite(low)) continue;
        max_high = std::max(max_high, high); min_low = std::min(min_low, low);
        max_favorable_bps = std::max(max_favorable_bps, ((max_high / entry_price) - 1.0) * 10000.0);
        max_adverse_bps = std::max(max_adverse_bps, ((entry_price / min_low) - 1.0) * 10000.0);
        if (time_to_min_profit_bars < 0) {
            low_before_profit = std::min(low_before_profit, low);
            const double high_gain_bps = ((high / entry_price) - 1.0) * 10000.0;
            if (high_gain_bps >= required_profit_bps) { time_to_min_profit_bars = static_cast<int>(idx) + 1; time_to_min_profit_minutes = static_cast<double>(time_to_min_profit_bars) * bar_minutes; profit_hit_index = static_cast<int>(idx); adverse_before_profit_bps = ((entry_price / low_before_profit) - 1.0) * 10000.0; }
        }
    }
    const bool reached_min_profit = time_to_min_profit_bars >= 0;
    const bool reached_target = max_favorable_bps >= std::max(target_bps, required_profit_bps);
    double post_profit_max_favorable_bps = 0.0, post_profit_extra_gain_bps = 0.0;
    if (reached_min_profit && profit_hit_index >= 0) {
        double post_profit_high = 0.0;
        for (ssize_t idx = profit_hit_index; idx < n; ++idx) { const double high = h(idx); if (high > 0.0 && std::isfinite(high)) post_profit_high = std::max(post_profit_high, high); }
        if (post_profit_high > 0.0) { post_profit_max_favorable_bps = ((post_profit_high / entry_price) - 1.0) * 10000.0; post_profit_extra_gain_bps = std::max(0.0, post_profit_max_favorable_bps - required_profit_bps); }
    }
    py::dict out;
    out["max_favorable_bps"] = max_favorable_bps; out["max_adverse_bps"] = max_adverse_bps; out["reached_min_profit"] = reached_min_profit; out["reached_target"] = reached_target; out["win_bps"] = std::max(0.0, max_favorable_bps - cost_bps); out["loss_bps"] = std::max(0.0, max_adverse_bps); out["time_to_min_profit_bars"] = time_to_min_profit_bars; out["time_to_min_profit_minutes"] = time_to_min_profit_minutes; out["forward_window_minutes"] = static_cast<double>(n) * bar_minutes; out["post_profit_max_favorable_bps"] = post_profit_max_favorable_bps; out["post_profit_extra_gain_bps"] = post_profit_extra_gain_bps; out["adverse_before_profit_bps"] = adverse_before_profit_bps; out["survived_to_profit"] = reached_min_profit && adverse_before_profit_bps <= max_adverse_before_profit_bps;
    return out;
}

static py::object simulate_armed_exit_net_bps(double entry_price, py::array_t<double, py::array::c_style | py::array::forcecast> highs, py::array_t<double, py::array::c_style | py::array::forcecast> lows, py::array_t<double, py::array::c_style | py::array::forcecast> closes, double target_bps, double cost_bps, double pullback_pct) {
    auto h = highs.unchecked<1>(); auto l = lows.unchecked<1>(); auto c = closes.unchecked<1>(); const ssize_t n = h.shape(0);
    if (entry_price <= 0.0 || n <= 0 || l.shape(0) != n || c.shape(0) != n) return py::none();
    const double target_price = entry_price * (1.0 + target_bps / 10000.0);
    bool armed = false; double peak = 0.0;
    for (ssize_t idx = 0; idx < n; ++idx) {
        const double high = h(idx), low = l(idx), close = c(idx);
        if (!std::isfinite(high) || !std::isfinite(low) || !std::isfinite(close)) continue;
        if (!armed) { if (high >= target_price) { armed = true; peak = std::max(high, target_price); } continue; }
        peak = std::max(peak, high);
        const double trigger_price = peak * (1.0 - pullback_pct);
        if (low <= trigger_price) { const double gross_bps = ((trigger_price / entry_price) - 1.0) * 10000.0; return py::float_(gross_bps - cost_bps); }
        peak = std::max(peak, close);
    }
    if (armed && peak > 0.0) { const double gross_bps = ((c(n - 1) / entry_price) - 1.0) * 10000.0; return py::float_(gross_bps - cost_bps); }
    return py::none();
}

static py::dict find_best_threshold_profile(py::array_t<double, py::array::c_style | py::array::forcecast> scores, py::array_t<double, py::array::c_style | py::array::forcecast> probabilities, py::array_t<double, py::array::c_style | py::array::forcecast> expected_values, py::array_t<double, py::array::c_style | py::array::forcecast> costs, py::array_t<double, py::array::c_style | py::array::forcecast> spreads, py::array_t<int, py::array::c_style | py::array::forcecast> reached_min_profit, py::array_t<int, py::array::c_style | py::array::forcecast> survived_to_profit, py::array_t<double, py::array::c_style | py::array::forcecast> max_favorable_bps, py::array_t<double, py::array::c_style | py::array::forcecast> time_to_min_profit_minutes, py::array_t<double, py::array::c_style | py::array::forcecast> forward_window_minutes, py::array_t<double, py::array::c_style | py::array::forcecast> selected_forward_window_minutes, py::array_t<double, py::array::c_style | py::array::forcecast> post_profit_extra_gain_bps, py::array_t<double, py::array::c_style | py::array::forcecast> adverse_before_profit_bps, py::array_t<double, py::array::c_style | py::array::forcecast> score_candidates, py::array_t<double, py::array::c_style | py::array::forcecast> probability_candidates, int calib_exact_min_samples, double similar_score_band, double similar_prob_band, double similar_cost_band_bps, double similar_spread_band_bps, double calib_min_win_rate, double calib_min_expected_value_bps, double preferred_time_to_min_profit_minutes) {
    auto score=scores.unchecked<1>(), prob=probabilities.unchecked<1>(), evs=expected_values.unchecked<1>(), cost=costs.unchecked<1>(), spread=spreads.unchecked<1>();
    auto reached=reached_min_profit.unchecked<1>(), survived=survived_to_profit.unchecked<1>();
    auto favorable=max_favorable_bps.unchecked<1>(), ttmin=time_to_min_profit_minutes.unchecked<1>(), fwin=forward_window_minutes.unchecked<1>(), swin=selected_forward_window_minutes.unchecked<1>(), extra=post_profit_extra_gain_bps.unchecked<1>(), adverse=adverse_before_profit_bps.unchecked<1>();
    auto score_cands=score_candidates.unchecked<1>(), prob_cands=probability_candidates.unchecked<1>(); const ssize_t n=score.shape(0);
    if (prob.shape(0)!=n||evs.shape(0)!=n||cost.shape(0)!=n||spread.shape(0)!=n||reached.shape(0)!=n||survived.shape(0)!=n||favorable.shape(0)!=n||ttmin.shape(0)!=n||fwin.shape(0)!=n||swin.shape(0)!=n||extra.shape(0)!=n||adverse.shape(0)!=n) throw std::runtime_error("find_best_threshold_profile received arrays with mismatched lengths");
    bool found=false; double best_quality=-std::numeric_limits<double>::infinity(), best_score_threshold=0.0, best_probability_threshold=0.0, best_win_rate=0.0, best_avg_win=0.0, best_avg_loss=0.0, best_ev=0.0, best_projected_gross_bps=0.0, best_median_time_to_min_profit=0.0, best_median_forward_window=0.0, best_median_selected_window=0.0, best_median_post_profit_extra_gain=0.0, best_median_adverse_before_profit=0.0; int best_sample_count=0; std::vector<int> best_indices;
    for (ssize_t si=0; si<score_cands.shape(0); ++si) { const double score_threshold=score_cands(si); if (!std::isfinite(score_threshold)) continue; for (ssize_t pi=0; pi<prob_cands.shape(0); ++pi) { const double probability_threshold=prob_cands(pi); if (!std::isfinite(probability_threshold)) continue; std::vector<int> selected; selected.reserve(static_cast<size_t>(n));
        for (ssize_t i=0; i<n; ++i) if (std::isfinite(score(i))&&std::isfinite(prob(i))&&score(i)>=score_threshold&&prob(i)>=probability_threshold) selected.push_back(static_cast<int>(i));
        if (!selected.empty()) { const double reference_cost=median_from_indices(selected,[&](int idx){return cost(idx);}); const double reference_spread=median_from_indices(selected,[&](int idx){return spread(idx);}); std::vector<int> similar_selected; similar_selected.reserve(selected.size()); for (int idx:selected) if (std::abs(score(idx)-score_threshold)<=similar_score_band&&std::abs(prob(idx)-probability_threshold)<=similar_prob_band&&std::abs(cost(idx)-reference_cost)<=similar_cost_band_bps&&std::abs(spread(idx)-reference_spread)<=similar_spread_band_bps) similar_selected.push_back(idx); if (static_cast<int>(similar_selected.size())>=calib_exact_min_samples) selected.swap(similar_selected); }
        std::vector<int> selected_for_stats; for (int idx:selected) if ((reached(idx)==0)||survived(idx)!=0) selected_for_stats.push_back(idx); const int sample_count=static_cast<int>(selected_for_stats.size()); if (sample_count<calib_exact_min_samples) continue;
        int win_count=0, loss_count=0; double win_sum=0.0, loss_sum=0.0; for (int idx:selected_for_stats) { const double ev=evs(idx); if (!std::isfinite(ev)) continue; if (ev>0.0) { win_count++; win_sum+=std::max(0.0,ev); } else { loss_count++; loss_sum+=std::abs(std::min(0.0,ev)); } }
        const double win_rate=static_cast<double>(win_count)/std::max(1,sample_count); const double avg_win=win_count>0?win_sum/static_cast<double>(win_count):0.0; const double avg_loss=loss_count>0?loss_sum/static_cast<double>(loss_count):0.0; const double candidate_ev=win_rate*avg_win-(1.0-win_rate)*avg_loss; if (win_rate<calib_min_win_rate||candidate_ev<calib_min_expected_value_bps) continue;
        std::vector<int> winners; for (int idx:selected_for_stats) if ((reached(idx)!=0)&&(survived(idx)!=0)) winners.push_back(idx); const std::vector<int>& projection_source=winners.empty()?selected_for_stats:winners;
        const double projected_gross=median_from_indices(projection_source,[&](int idx){return favorable(idx);}); const double median_time=median_from_indices(winners,[&](int idx){return ttmin(idx);}); const double median_forward=median_from_indices(projection_source,[&](int idx){return fwin(idx);}); const double median_selected=median_from_indices(projection_source,[&](int idx){return swin(idx);}); const double median_extra=median_from_indices(projection_source,[&](int idx){return extra(idx);}); const double median_adverse=median_from_indices(projection_source,[&](int idx){return adverse(idx);});
        const double opportunity_bonus=std::min(10.0,static_cast<double>(sample_count)/25.0); const double quality_score=candidate_ev*1.00+win_rate*12.0+opportunity_bonus+median_extra*0.20-median_adverse*0.25-std::max(0.0,median_time-preferred_time_to_min_profit_minutes)*0.05-score_threshold*0.010-probability_threshold*1.00;
        if (!found||quality_score>best_quality) { found=true; best_quality=quality_score; best_score_threshold=score_threshold; best_probability_threshold=probability_threshold; best_win_rate=win_rate; best_avg_win=avg_win; best_avg_loss=avg_loss; best_ev=candidate_ev; best_projected_gross_bps=projected_gross; best_median_time_to_min_profit=median_time; best_median_forward_window=median_forward; best_median_selected_window=median_selected; best_median_post_profit_extra_gain=median_extra; best_median_adverse_before_profit=median_adverse; best_sample_count=sample_count; best_indices=selected_for_stats; }
    }}
    py::dict out; out["found"]=found; if(!found) return out; py::list py_indices; for(int idx:best_indices) py_indices.append(idx); out["score_threshold"]=best_score_threshold; out["prob_threshold"]=best_probability_threshold; out["win_rate"]=best_win_rate; out["avg_win"]=best_avg_win; out["avg_loss"]=best_avg_loss; out["ev"]=best_ev; out["projected_gross_bps"]=best_projected_gross_bps; out["median_time_to_min_profit"]=best_median_time_to_min_profit; out["median_forward_window"]=best_median_forward_window; out["median_selected_window"]=best_median_selected_window; out["median_post_profit_extra_gain"]=best_median_post_profit_extra_gain; out["median_adverse_before_profit"]=best_median_adverse_before_profit; out["n"]=best_sample_count; out["quality_score"]=best_quality; out["selected_indices"]=py_indices; return out;
}


static py::dict evaluate_best_window_from_arrays(
    double entry_price,
    py::array_t<double, py::array::c_style | py::array::forcecast> highs,
    py::array_t<double, py::array::c_style | py::array::forcecast> lows,
    int start_index,
    py::array_t<int, py::array::c_style | py::array::forcecast> forward_windows,
    double target_bps,
    double cost_bps,
    double min_net_gain_bps,
    double bar_minutes,
    double max_adverse_before_profit_bps,
    double preferred_time_to_min_profit_minutes
) {
    auto h = highs.unchecked<1>();
    auto l = lows.unchecked<1>();
    auto windows = forward_windows.unchecked<1>();

    const ssize_t n = h.shape(0);

    py::dict out;
    out["found"] = false;

    if (
        entry_price <= 0.0 ||
        n <= 0 ||
        l.shape(0) != n ||
        start_index < 0 ||
        windows.shape(0) <= 0
    ) {
        return out;
    }

    const double required_profit_bps = cost_bps + min_net_gain_bps;

    bool found = false;
    double best_quality = -std::numeric_limits<double>::infinity();

    int best_window_bars = 0;
    double best_max_favorable_bps = 0.0;
    double best_max_adverse_bps = 0.0;
    bool best_reached_min_profit = false;
    bool best_reached_target = false;
    double best_win_bps = 0.0;
    double best_loss_bps = 0.0;
    int best_time_to_min_profit_bars = -1;
    double best_time_to_min_profit_minutes = 0.0;
    double best_forward_window_minutes = 0.0;
    double best_post_profit_max_favorable_bps = 0.0;
    double best_post_profit_extra_gain_bps = 0.0;
    double best_adverse_before_profit_bps = 0.0;
    bool best_survived_to_profit = false;
    double best_expected_value_bps = 0.0;

    for (ssize_t wi = 0; wi < windows.shape(0); ++wi) {
        const int forward_bars = windows(wi);

        if (forward_bars <= 0) {
            continue;
        }

        const int end_index = start_index + forward_bars;

        if (end_index > n) {
            continue;
        }

        double max_high = 0.0;
        double min_low = std::numeric_limits<double>::infinity();
        double max_favorable_bps = 0.0;
        double max_adverse_bps = 0.0;

        int time_to_min_profit_bars = -1;
        int profit_hit_index = -1;
        double time_to_min_profit_minutes = 0.0;
        double adverse_before_profit_bps = 0.0;
        double low_before_profit = entry_price;

        for (int idx = start_index; idx < end_index; ++idx) {
            const double high = h(idx);
            const double low = l(idx);

            if (
                high <= 0.0 ||
                low <= 0.0 ||
                !std::isfinite(high) ||
                !std::isfinite(low)
            ) {
                continue;
            }

            max_high = std::max(max_high, high);
            min_low = std::min(min_low, low);

            max_favorable_bps = std::max(
                max_favorable_bps,
                ((max_high / entry_price) - 1.0) * 10000.0
            );

            max_adverse_bps = std::max(
                max_adverse_bps,
                ((entry_price / min_low) - 1.0) * 10000.0
            );

            if (time_to_min_profit_bars < 0) {
                low_before_profit = std::min(low_before_profit, low);

                const double high_gain_bps =
                    ((high / entry_price) - 1.0) * 10000.0;

                if (high_gain_bps >= required_profit_bps) {
                    time_to_min_profit_bars = idx - start_index + 1;
                    time_to_min_profit_minutes =
                        static_cast<double>(time_to_min_profit_bars) * bar_minutes;
                    profit_hit_index = idx;
                    adverse_before_profit_bps =
                        ((entry_price / low_before_profit) - 1.0) * 10000.0;
                }
            }
        }

        const bool reached_min_profit = time_to_min_profit_bars >= 0;
        const bool reached_target =
            max_favorable_bps >= std::max(target_bps, required_profit_bps);

        const double win_bps = std::max(0.0, max_favorable_bps - cost_bps);
        const double loss_bps = std::max(0.0, max_adverse_bps);

        const double forward_window_minutes =
            static_cast<double>(forward_bars) * bar_minutes;

        double post_profit_max_favorable_bps = 0.0;
        double post_profit_extra_gain_bps = 0.0;

        if (reached_min_profit && profit_hit_index >= start_index) {
            double post_profit_high = 0.0;

            for (int idx = profit_hit_index; idx < end_index; ++idx) {
                const double high = h(idx);
                if (high > 0.0 && std::isfinite(high)) {
                    post_profit_high = std::max(post_profit_high, high);
                }
            }

            if (post_profit_high > 0.0) {
                post_profit_max_favorable_bps =
                    ((post_profit_high / entry_price) - 1.0) * 10000.0;
                post_profit_extra_gain_bps =
                    std::max(0.0, post_profit_max_favorable_bps - required_profit_bps);
            }
        }

        const bool survived_to_profit =
            reached_min_profit &&
            adverse_before_profit_bps <= max_adverse_before_profit_bps;

        const double expected_value_bps =
            reached_min_profit ? win_bps : -loss_bps;

        const double time_penalty =
            reached_min_profit
                ? std::max(
                    0.0,
                    time_to_min_profit_minutes - preferred_time_to_min_profit_minutes
                ) * 0.05
                : 0.0;

        const double quality =
            expected_value_bps +
            post_profit_extra_gain_bps * 0.35 -
            adverse_before_profit_bps * 0.45 -
            time_penalty -
            (survived_to_profit ? 0.0 : 1000.0);

        if (!found || quality > best_quality) {
            found = true;
            best_quality = quality;
            best_window_bars = forward_bars;
            best_max_favorable_bps = max_favorable_bps;
            best_max_adverse_bps = max_adverse_bps;
            best_reached_min_profit = reached_min_profit;
            best_reached_target = reached_target;
            best_win_bps = win_bps;
            best_loss_bps = loss_bps;
            best_time_to_min_profit_bars = time_to_min_profit_bars;
            best_time_to_min_profit_minutes = time_to_min_profit_minutes;
            best_forward_window_minutes = forward_window_minutes;
            best_post_profit_max_favorable_bps = post_profit_max_favorable_bps;
            best_post_profit_extra_gain_bps = post_profit_extra_gain_bps;
            best_adverse_before_profit_bps = adverse_before_profit_bps;
            best_survived_to_profit = survived_to_profit;
            best_expected_value_bps = expected_value_bps;
        }
    }

    out["found"] = found;

    if (!found) {
        return out;
    }

    out["chosen_window_bars"] = best_window_bars;
    out["quality_score"] = best_quality;
    out["max_favorable_bps"] = best_max_favorable_bps;
    out["max_adverse_bps"] = best_max_adverse_bps;
    out["reached_min_profit"] = best_reached_min_profit;
    out["reached_target"] = best_reached_target;
    out["win_bps"] = best_win_bps;
    out["loss_bps"] = best_loss_bps;
    out["time_to_min_profit_bars"] = best_time_to_min_profit_bars;
    out["time_to_min_profit_minutes"] = best_time_to_min_profit_minutes;
    out["forward_window_minutes"] = best_forward_window_minutes;
    out["selected_forward_window_minutes"] = best_forward_window_minutes;
    out["post_profit_max_favorable_bps"] = best_post_profit_max_favorable_bps;
    out["post_profit_extra_gain_bps"] = best_post_profit_extra_gain_bps;
    out["adverse_before_profit_bps"] = best_adverse_before_profit_bps;
    out["survived_to_profit"] = best_survived_to_profit;
    out["expected_value_bps"] = best_expected_value_bps;

    return out;
}

PYBIND11_MODULE(fast_calibration_core, m) {
    m.doc() = "Fast C++ calibration helpers for the Binance.US trading bot";
    m.def("evaluate_outcome_arrays", &evaluate_outcome_arrays, py::arg("entry_price"), py::arg("highs"), py::arg("lows"), py::arg("target_bps"), py::arg("cost_bps"), py::arg("min_net_gain_bps"), py::arg("bar_minutes"), py::arg("max_adverse_before_profit_bps"));
    m.def("simulate_armed_exit_net_bps", &simulate_armed_exit_net_bps, py::arg("entry_price"), py::arg("highs"), py::arg("lows"), py::arg("closes"), py::arg("target_bps"), py::arg("cost_bps"), py::arg("pullback_pct"));
    m.def("find_best_threshold_profile", &find_best_threshold_profile, py::arg("scores"), py::arg("probabilities"), py::arg("expected_values"), py::arg("costs"), py::arg("spreads"), py::arg("reached_min_profit"), py::arg("survived_to_profit"), py::arg("max_favorable_bps"), py::arg("time_to_min_profit_minutes"), py::arg("forward_window_minutes"), py::arg("selected_forward_window_minutes"), py::arg("post_profit_extra_gain_bps"), py::arg("adverse_before_profit_bps"), py::arg("score_candidates"), py::arg("probability_candidates"), py::arg("calib_exact_min_samples"), py::arg("similar_score_band"), py::arg("similar_prob_band"), py::arg("similar_cost_band_bps"), py::arg("similar_spread_band_bps"), py::arg("calib_min_win_rate"), py::arg("calib_min_expected_value_bps"), py::arg("preferred_time_to_min_profit_minutes"));
    m.def(
        "evaluate_best_window_from_arrays",
        &evaluate_best_window_from_arrays,
        py::arg("entry_price"),
        py::arg("highs"),
        py::arg("lows"),
        py::arg("start_index"),
        py::arg("forward_windows"),
        py::arg("target_bps"),
        py::arg("cost_bps"),
        py::arg("min_net_gain_bps"),
        py::arg("bar_minutes"),
        py::arg("max_adverse_before_profit_bps"),
        py::arg("preferred_time_to_min_profit_minutes")
    );
}
