#pragma once

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>


class Timer
{
public:
    using clock = std::chrono::high_resolution_clock;

    void start()
    {
        running_ = true;
        start_time_ = clock::now();
    }

    void stop()
    {
        if (!running_)
            return;

        auto end_time = clock::now();
        double ms = std::chrono::duration<double, std::milli>(end_time - start_time_).count();
        history_.push_back(ms);
        running_ = false;
    }

    // Last elapsed in milliseconds
    double last_ms() const { return history_.empty() ? 0.0 : history_.back(); }
    // Last elapsed in seconds
    double last_s() const { return last_ms() / 1000.0; }

    double total_ms() const { return std::accumulate(history_.begin(), history_.end(), 0.0); }

    double total_s() const { return total_ms() / 1000.0; }

    std::vector<double> const& history() const { return history_; }
    void clear() { history_.clear(); }

private:
    clock::time_point start_time_;
    bool running_ = false;
    std::vector<double> history_; // stored in milliseconds
};

class TimerCollection
{
public:
    enum TimerId
    {
        Init = 0,
        Embedding,
        EvalObjective,
        Backpropagation,
        Update,
        Resample,
        PerIteration,
        Total,
        NumTimers
    };

    void start(TimerId id) { timers_[id].start(); }
    void stop(TimerId id) { timers_[id].stop(); }
    void start_all()
    {
        for (auto& t : timers_)
            t.start();
    }
    void stop_all()
    {
        for (auto& t : timers_)
            t.stop();
    }

    double last_s(TimerId id) const { return timers_[id].last_s(); }
    double last_ms(TimerId id) const { return timers_[id].last_ms(); }

    double total_s(TimerId id) const { return timers_[id].total_s(); }
    double total_ms(TimerId id) const { return timers_[id].total_ms(); }

    void print_stats(std::ostream& _out) const
    {
        constexpr char const* names[] = {"Init", "Embedding", "EvalObjective", "Backpropagation", "Update", "Resample", "PerIteration", "Total"};

        double const total_ms_all = total_ms(Total); // total time for percent calculation

        _out << std::fixed << std::setprecision(6);

        // Header
        _out << std::left << std::setw(20) << "Section" << std::right << std::setw(15) << "Total ms" << std::setw(15) << "Total s" << std::setw(15) << "Total min"
             << std::setw(15) << "Percent" << std::setw(15) << "Per-iter ms" << std::setw(15) << "Per-iter s" << std::setw(15) << "Per-iter min\n";

        for (int i = 0; i < NumTimers; ++i)
        {
            TimerId id = static_cast<TimerId>(i);

            double ms = total_ms(id);
            double s = total_s(id);
            double min = s / 60.0;
            double pct = (total_ms_all > 0.0) ? (ms / total_ms_all) * 100.0 : 0.0;

            std::string per_iter_ms = "-", per_iter_s = "-", per_iter_min = "-";

            // Only compute per-iteration values for selected timers
            if (id == Embedding || id == EvalObjective || id == Backpropagation || id == Update || id == PerIteration)
            {
                size_t num_iter = timers_[id].history().size();
                if (num_iter > 0)
                {
                    per_iter_ms = std::to_string(ms / num_iter);
                    per_iter_s = std::to_string(s / num_iter);
                    per_iter_min = std::to_string(min / num_iter);
                }
            }

            _out << std::left << std::setw(20) << names[i] << std::right << std::setw(15) << ms << std::setw(15) << s << std::setw(15) << min
                 << std::setw(14) << pct << "%" << std::setw(15) << per_iter_ms << std::setw(15) << per_iter_s << std::setw(15) << per_iter_min << "\n";
        }
    }

private:
    std::array<Timer, NumTimers> timers_;
};

extern TimerCollection timers; // declaration
