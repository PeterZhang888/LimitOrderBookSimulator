#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int D = 6;
constexpr int ROW_PARAMS = 7; // mu_i + six N_ij

const std::array<std::string, D> EVENT_LABELS = {
    "limit_buy", "limit_sell", "market_buy", "market_sell", "cancel_bid", "cancel_ask"
};

struct EventData {
    std::vector<double> times; // relative to t0, seconds
    std::vector<int> types;
    double T = 0.0;
    double t0_original = 0.0;
    double t1_original = 0.0;
};

struct RowData {
    std::vector<std::array<double, D>> R_at_events; // only events whose observed type is this row
};

struct RowFitResult {
    int row = -1;
    std::array<double, ROW_PARAMS> theta{}; // theta[0] = mu_i, theta[1+j] = N_ij
    double objective = std::numeric_limits<double>::quiet_NaN();
    double projected_grad_norm = std::numeric_limits<double>::quiet_NaN();
    int iterations = 0;
    int converged = 0;
    long long event_count = 0;
    double runtime_seconds = 0.0;
};

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(item);
    return out;
}

EventData load_events_csv(const std::string& path,
                          double start_seconds,
                          double end_seconds,
                          bool use_time_filter) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open input file: " + path);
    }

    std::string header;
    if (!std::getline(in, header)) {
        throw std::runtime_error("Input file is empty.");
    }

    auto cols = split_csv_line(header);
    int time_col = -1;
    int type_col = -1;
    for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
        if (cols[i] == "time_seconds") time_col = i;
        if (cols[i] == "event_type") type_col = i;
    }
    if (time_col < 0 || type_col < 0) {
        throw std::runtime_error("CSV must contain columns: time_seconds,event_type");
    }

    std::vector<std::pair<double, int>> rows;
    rows.reserve(5'000'000);

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto parts = split_csv_line(line);
        if (static_cast<int>(parts.size()) <= std::max(time_col, type_col)) continue;

        double t = std::stod(parts[time_col]);
        int u = std::stoi(parts[type_col]);
        if (u < 0 || u >= D) continue;

        if (use_time_filter) {
            if (t < start_seconds || t > end_seconds) continue;
        }
        rows.emplace_back(t, u);
    }

    if (rows.empty()) {
        throw std::runtime_error("No events after filtering.");
    }

    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    EventData data;
    data.t0_original = use_time_filter ? start_seconds : rows.front().first;
    data.t1_original = use_time_filter ? end_seconds : rows.back().first;
    data.T = data.t1_original - data.t0_original;

    if (data.T <= 0.0) {
        throw std::runtime_error("Observation window T must be positive.");
    }

    data.times.reserve(rows.size());
    data.types.reserve(rows.size());

    for (const auto& p : rows) {
        double rel_t = p.first - data.t0_original;
        if (rel_t < -1e-9 || rel_t > data.T + 1e-9) continue;
        data.times.push_back(std::max(0.0, std::min(data.T, rel_t)));
        data.types.push_back(p.second);
    }

    if (data.times.empty()) {
        throw std::runtime_error("No valid relative events after filtering.");
    }

    return data;
}

bool rank_owns_row(int row, int rank, int size) {
    return (row % size) == rank;
}

void precompute_for_owned_rows(const EventData& data,
                               double beta,
                               int rank,
                               int size,
                               std::array<double, D>& G,
                               std::array<RowData, D>& row_data) {
    G.fill(0.0);
    std::array<double, D> R{};
    R.fill(0.0);

    double prev_t = 0.0;
    const size_t n = data.times.size();

    for (size_t k = 0; k < n; ++k) {
        const double t = data.times[k];
        const int u = data.types[k];

        double dt = t - prev_t;
        if (dt < -1e-12) {
            throw std::runtime_error("Event times are not sorted.");
        }
        if (dt < 0.0) dt = 0.0;

        const double decay = std::exp(-beta * dt);
        for (int j = 0; j < D; ++j) R[j] *= decay;

        // Store R(t_k-) only if this rank fits the observed event row u.
        if (rank_owns_row(u, rank, size)) {
            row_data[u].R_at_events.push_back(R);
        }

        const double remaining = std::max(data.T - t, 0.0);
        G[u] += (1.0 - std::exp(-beta * remaining)) / beta;

        R[u] += 1.0;
        prev_t = t;
    }
}

double dot_N_R(const std::array<double, D>& N_row,
               const std::array<double, D>& R) {
    double s = 0.0;
    for (int j = 0; j < D; ++j) s += N_row[j] * R[j];
    return s;
}

struct ObjGrad {
    double value = 0.0;
    std::array<double, ROW_PARAMS> grad{};
};

ObjGrad row_objective_gradient(const std::array<double, ROW_PARAMS>& theta,
                               const RowData& row_data,
                               const std::array<double, D>& G,
                               double T,
                               double beta,
                               double eta) {
    constexpr double EPS_LAMBDA = 1e-12;

    const double mu = theta[0];
    std::array<double, D> Nrow{};
    for (int j = 0; j < D; ++j) Nrow[j] = theta[1 + j];

    const long long count = static_cast<long long>(row_data.R_at_events.size());
    const double scale = static_cast<double>(std::max<long long>(count, 1));

    double nll = mu * T;
    for (int j = 0; j < D; ++j) {
        nll += beta * Nrow[j] * G[j];
    }
    if (eta > 0.0) {
        for (int j = 0; j < D; ++j) nll += eta * Nrow[j];
    }

    ObjGrad out;
    out.grad.fill(0.0);
    out.grad[0] = T;
    for (int j = 0; j < D; ++j) {
        out.grad[1 + j] = beta * G[j] + (eta > 0.0 ? eta : 0.0);
    }

    for (const auto& R : row_data.R_at_events) {
        double lambda = mu + beta * dot_N_R(Nrow, R);
        if (lambda < EPS_LAMBDA) lambda = EPS_LAMBDA;

        nll -= std::log(lambda);

        const double inv_lambda = 1.0 / lambda;
        out.grad[0] -= inv_lambda;
        for (int j = 0; j < D; ++j) {
            out.grad[1 + j] -= beta * R[j] * inv_lambda;
        }
    }

    // Scaling does not change the optimum, but makes gradient descent numerically easier.
    out.value = nll / scale;
    for (double& g : out.grad) g /= scale;
    return out;
}

std::array<double, ROW_PARAMS> project_theta(const std::array<double, ROW_PARAMS>& theta) {
    std::array<double, ROW_PARAMS> out = theta;
    out[0] = std::max(out[0], 1e-12); // mu lower bound
    for (int j = 0; j < D; ++j) {
        out[1 + j] = std::max(0.0, std::min(0.95, out[1 + j]));
    }
    return out;
}

double projected_grad_norm(const std::array<double, ROW_PARAMS>& theta,
                           const std::array<double, ROW_PARAMS>& grad) {
    double s2 = 0.0;
    for (int p = 0; p < ROW_PARAMS; ++p) {
        double g = grad[p];
        if (p == 0) {
            if (theta[p] <= 1e-12 + 1e-14 && g > 0.0) g = 0.0;
        } else {
            if (theta[p] <= 1e-14 && g > 0.0) g = 0.0;
            if (theta[p] >= 0.95 - 1e-14 && g < 0.0) g = 0.0;
        }
        s2 += g * g;
    }
    return std::sqrt(s2);
}

RowFitResult fit_row_projected_gradient(int row,
                                         const RowData& row_data,
                                         const std::array<double, D>& G,
                                         double T,
                                         double beta,
                                         double eta,
                                         int max_iter) {
    auto t_start = std::chrono::high_resolution_clock::now();

    RowFitResult result;
    result.row = row;
    result.event_count = static_cast<long long>(row_data.R_at_events.size());

    const double empirical_rate = result.event_count > 0 ? result.event_count / T : 1e-12;

    std::array<double, ROW_PARAMS> theta{};
    theta.fill(0.0);
    theta[0] = std::max(0.5 * empirical_rate, 1e-9);
    for (int j = 0; j < D; ++j) theta[1 + j] = 0.005;
    theta[1 + row] = 0.05;
    theta = project_theta(theta);

    double step = 1.0;
    constexpr double TOL = 1e-6;
    constexpr double ARMIJO = 1e-4;

    ObjGrad og = row_objective_gradient(theta, row_data, G, T, beta, eta);
    double pg_norm = projected_grad_norm(theta, og.grad);

    int iter = 0;
    int converged = 0;

    for (iter = 0; iter < max_iter; ++iter) {
        if (pg_norm < TOL) {
            converged = 1;
            break;
        }

        bool accepted = false;
        double local_step = step;
        std::array<double, ROW_PARAMS> best_theta = theta;
        ObjGrad best_og = og;

        for (int ls = 0; ls < 40; ++ls) {
            std::array<double, ROW_PARAMS> trial{};
            for (int p = 0; p < ROW_PARAMS; ++p) {
                trial[p] = theta[p] - local_step * og.grad[p];
            }
            trial = project_theta(trial);

            ObjGrad trial_og = row_objective_gradient(trial, row_data, G, T, beta, eta);

            // A simple Armijo-like rule; if the objective decreases, accept.
            const double sufficient = og.value - ARMIJO * local_step * pg_norm * pg_norm;
            if (std::isfinite(trial_og.value) &&
                (trial_og.value <= sufficient || trial_og.value < og.value)) {
                accepted = true;
                best_theta = trial;
                best_og = trial_og;
                break;
            }
            local_step *= 0.5;
        }

        if (!accepted) {
            break;
        }

        theta = best_theta;
        og = best_og;
        pg_norm = projected_grad_norm(theta, og.grad);
        step = std::min(local_step * 1.25, 10.0);

        // Additional relative-improvement convergence check is intentionally conservative.
    }

    result.theta = theta;
    result.objective = og.value;
    result.projected_grad_norm = pg_norm;
    result.iterations = iter;
    result.converged = converged;

    auto t_end = std::chrono::high_resolution_clock::now();
    result.runtime_seconds = std::chrono::duration<double>(t_end - t_start).count();
    return result;
}

double spectral_radius_power(const std::array<std::array<double, D>, D>& N) {
    std::array<double, D> v{};
    for (int i = 0; i < D; ++i) v[i] = 1.0 / std::sqrt(static_cast<double>(D));

    double rho = 0.0;
    for (int iter = 0; iter < 1000; ++iter) {
        std::array<double, D> w{};
        w.fill(0.0);
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                w[i] += N[i][j] * v[j];
            }
        }
        double norm2 = 0.0;
        for (double x : w) norm2 += x * x;
        double norm = std::sqrt(norm2);
        if (norm < 1e-15) return 0.0;
        for (int i = 0; i < D; ++i) v[i] = w[i] / norm;
    }

    std::array<double, D> Nv{};
    Nv.fill(0.0);
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) Nv[i] += N[i][j] * v[j];
    }
    double num = 0.0, den = 0.0;
    for (int i = 0; i < D; ++i) {
        num += v[i] * Nv[i];
        den += v[i] * v[i];
    }
    rho = den > 0.0 ? num / den : 0.0;
    return std::abs(rho);
}

void make_dir(const std::string& dir) {
    std::string cmd = "mkdir -p \"" + dir + "\"";
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        throw std::runtime_error("Failed to create output directory: " + dir);
    }
}

void write_outputs(const std::string& out_dir,
                   double beta,
                   double eta,
                   double T,
                   double t0_original,
                   double t1_original,
                   const std::array<double, D>& mu,
                   const std::array<std::array<double, D>, D>& N,
                   double rho_before,
                   double rho_after,
                   int rescaled,
                   const std::array<RowFitResult, D>& row_results) {
    make_dir(out_dir);

    {
        std::ofstream out(out_dir + "/fixed_beta_mle_mu.csv");
        out << "event_type,label,mu\n";
        out << std::setprecision(17);
        for (int i = 0; i < D; ++i) {
            out << i << "," << EVENT_LABELS[i] << "," << mu[i] << "\n";
        }
    }

    {
        std::ofstream out(out_dir + "/fixed_beta_mle_branching_matrix_N.csv");
        out << "future_event_type_row";
        for (int j = 0; j < D; ++j) out << "," << EVENT_LABELS[j];
        out << "\n";
        out << std::setprecision(17);
        for (int i = 0; i < D; ++i) {
            out << EVENT_LABELS[i];
            for (int j = 0; j < D; ++j) out << "," << N[i][j];
            out << "\n";
        }
    }

    {
        std::ofstream out(out_dir + "/fixed_beta_mle_row_results.csv");
        out << "row,label,event_count,objective,projected_grad_norm,iterations,converged,runtime_seconds\n";
        out << std::setprecision(17);
        for (int i = 0; i < D; ++i) {
            const auto& r = row_results[i];
            out << i << "," << EVENT_LABELS[i] << "," << r.event_count << ","
                << r.objective << "," << r.projected_grad_norm << ","
                << r.iterations << "," << r.converged << "," << r.runtime_seconds << "\n";
        }
    }

    {
        std::ofstream out(out_dir + "/fixed_beta_mle_params_flat.csv");
        out << "beta,eta,T,start_time_seconds_original,end_time_seconds_original,"
            << "spectral_radius_before_rescale,spectral_radius_after_rescale,rescaled";
        for (int i = 0; i < D; ++i) out << ",mu_" << i;
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) out << ",N_" << i << j;
        }
        out << "\n";

        out << std::setprecision(17);
        out << beta << "," << eta << "," << T << "," << t0_original << "," << t1_original << ","
            << rho_before << "," << rho_after << "," << rescaled;
        for (int i = 0; i < D; ++i) out << "," << mu[i];
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) out << "," << N[i][j];
        }
        out << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        if (argc < 2) {
            if (rank == 0) {
                std::cerr << "Usage:\n"
                          << "  mpirun -np 6 ./hawkes_mle_fixed_beta <input_csv> [beta] [eta] [start] [end] [max_iter] [out_dir]\n\n"
                          << "Example:\n"
                          << "  mpirun -np 6 ./hawkes_mle_fixed_beta hawkes_events_QQQ.csv 10 0.001 34200 57600 300 hawkes_mle_fixed_beta_results\n";
            }
            MPI_Finalize();
            return 1;
        }

        const std::string input_csv = argv[1];
        const double beta = (argc > 2) ? std::stod(argv[2]) : 10.0;
        const double eta = (argc > 3) ? std::stod(argv[3]) : 0.0;
        const double start_seconds = (argc > 4) ? std::stod(argv[4]) : 34200.0;
        const double end_seconds = (argc > 5) ? std::stod(argv[5]) : 57600.0;
        const int max_iter = (argc > 6) ? std::stoi(argv[6]) : 300;
        const std::string out_dir = (argc > 7) ? argv[7] : "hawkes_mle_fixed_beta_results";
        const bool use_time_filter = true;

        if (rank == 0) {
            std::cout << "Fixed-beta full 6x6 Hawkes MLE\n"
                      << "input_csv = " << input_csv << "\n"
                      << "beta      = " << beta << "\n"
                      << "eta       = " << eta << "\n"
                      << "window    = [" << start_seconds << ", " << end_seconds << "]\n"
                      << "max_iter  = " << max_iter << "\n"
                      << "MPI size  = " << size << "\n";
        }

        EventData data = load_events_csv(input_csv, start_seconds, end_seconds, use_time_filter);

        std::array<long long, D> counts{};
        counts.fill(0);
        for (int u : data.types) counts[u]++;

        if (rank == 0) {
            std::cout << "events used = " << data.times.size() << "\n"
                      << "T           = " << data.T << " seconds\n";
            for (int i = 0; i < D; ++i) {
                std::cout << "  " << i << " " << std::setw(12) << EVENT_LABELS[i]
                          << ": " << counts[i] << "\n";
            }
            std::cout << "Precomputing row data...\n";
        }

        std::array<double, D> G{};
        std::array<RowData, D> row_data;
        precompute_for_owned_rows(data, beta, rank, size, G, row_data);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::cout << "Precompute finished.\n";

        // Local row results, empty rows have row=-1.
        std::array<RowFitResult, D> local_results;
        for (int i = 0; i < D; ++i) local_results[i].row = -1;

        for (int row = 0; row < D; ++row) {
            if (!rank_owns_row(row, rank, size)) continue;

            std::cout << "rank " << rank << " fitting row " << row
                      << " (" << EVENT_LABELS[row] << "), events="
                      << row_data[row].R_at_events.size() << std::endl;

            RowFitResult fit = fit_row_projected_gradient(
                row, row_data[row], G, data.T, beta, eta, max_iter
            );
            local_results[row] = fit;

            std::cout << "rank " << rank << " finished row " << row
                      << ": obj=" << fit.objective
                      << ", pg_norm=" << fit.projected_grad_norm
                      << ", iter=" << fit.iterations
                      << ", converged=" << fit.converged
                      << ", runtime=" << fit.runtime_seconds << "s\n";
        }

        // Reduce all row results to rank 0.
        std::vector<double> local_theta(D * ROW_PARAMS, 0.0);
        std::vector<double> global_theta(D * ROW_PARAMS, 0.0);
        std::vector<double> local_obj(D, 0.0), global_obj(D, 0.0);
        std::vector<double> local_pg(D, 0.0), global_pg(D, 0.0);
        std::vector<double> local_runtime(D, 0.0), global_runtime(D, 0.0);
        std::vector<long long> local_count(D, 0), global_count(D, 0);
        std::vector<int> local_iter(D, 0), global_iter(D, 0);
        std::vector<int> local_conv(D, 0), global_conv(D, 0);
        std::vector<int> local_done(D, 0), global_done(D, 0);

        for (int i = 0; i < D; ++i) {
            if (local_results[i].row == i) {
                local_done[i] = 1;
                for (int p = 0; p < ROW_PARAMS; ++p) {
                    local_theta[i * ROW_PARAMS + p] = local_results[i].theta[p];
                }
                local_obj[i] = local_results[i].objective;
                local_pg[i] = local_results[i].projected_grad_norm;
                local_runtime[i] = local_results[i].runtime_seconds;
                local_count[i] = local_results[i].event_count;
                local_iter[i] = local_results[i].iterations;
                local_conv[i] = local_results[i].converged;
            }
        }

        MPI_Reduce(local_theta.data(), global_theta.data(), D * ROW_PARAMS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_obj.data(), global_obj.data(), D, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_pg.data(), global_pg.data(), D, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_runtime.data(), global_runtime.data(), D, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_count.data(), global_count.data(), D, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_iter.data(), global_iter.data(), D, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_conv.data(), global_conv.data(), D, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_done.data(), global_done.data(), D, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int i = 0; i < D; ++i) {
                if (global_done[i] != 1) {
                    throw std::runtime_error("Row " + std::to_string(i) + " was not fitted exactly once. Try mpirun -np <= 6.");
                }
            }

            std::array<double, D> mu{};
            std::array<std::array<double, D>, D> N{};
            std::array<RowFitResult, D> final_rows;

            for (int i = 0; i < D; ++i) {
                final_rows[i].row = i;
                final_rows[i].event_count = global_count[i];
                final_rows[i].objective = global_obj[i];
                final_rows[i].projected_grad_norm = global_pg[i];
                final_rows[i].iterations = global_iter[i];
                final_rows[i].converged = global_conv[i];
                final_rows[i].runtime_seconds = global_runtime[i];

                mu[i] = global_theta[i * ROW_PARAMS + 0];
                for (int j = 0; j < D; ++j) {
                    N[i][j] = global_theta[i * ROW_PARAMS + 1 + j];
                }
            }

            const double rho_before = spectral_radius_power(N);
            double rho_after = rho_before;
            int rescaled = 0;
            constexpr double TARGET_RHO = 0.95;
            if (rho_before > TARGET_RHO && rho_before > 0.0) {
                const double factor = TARGET_RHO / rho_before;
                for (int i = 0; i < D; ++i) {
                    for (int j = 0; j < D; ++j) N[i][j] *= factor;
                }
                rho_after = spectral_radius_power(N);
                rescaled = 1;
            }

            write_outputs(out_dir, beta, eta, data.T, data.t0_original, data.t1_original,
                          mu, N, rho_before, rho_after, rescaled, final_rows);

            std::cout << "\nFinished.\n"
                      << "Output directory: " << out_dir << "\n"
                      << "Spectral radius before rescale: " << rho_before << "\n"
                      << "Spectral radius after  rescale: " << rho_after << "\n"
                      << "Rescaled: " << rescaled << "\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "Rank " << rank << " error: " << ex.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}

