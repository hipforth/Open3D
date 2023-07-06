// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Eigen/Dense>
#include <unordered_set>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Parallel.h"
#include "open3d/utility/ProgressBar.h"

namespace open3d {
namespace geometry {

std::vector<int> PointCloud::ClusterDBSCAN(double eps,
                                           size_t min_points,
                                           bool print_progress) const {
    KDTreeFlann kdtree(*this);

    // Precompute all neighbors.
    utility::LogDebug("Precompute neighbors.");
    utility::ProgressBar progress_bar(points_.size(), "Precompute neighbors.",
                                      print_progress);
    std::vector<std::vector<int>> nbs(points_.size());
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
    for (int idx = 0; idx < int(points_.size()); ++idx) {
        std::vector<double> dists2;
        kdtree.SearchRadius(points_[idx], eps, nbs[idx], dists2);

#pragma omp critical(ClusterDBSCAN)
        { ++progress_bar; }
    }
    utility::LogDebug("Done Precompute neighbors.");

    // Set all labels to undefined (-2).
    utility::LogDebug("Compute Clusters");
    progress_bar.Reset(points_.size(), "Clustering", print_progress);
    std::vector<int> labels(points_.size(), -2);
    int cluster_label = 0;
    for (size_t idx = 0; idx < points_.size(); ++idx) {
        // Label is not undefined.
        if (labels[idx] != -2) {
            continue;
        }

        // Check density.
        if (nbs[idx].size() < min_points) {
            labels[idx] = -1;
            continue;
        }

        std::unordered_set<int> nbs_next(nbs[idx].begin(), nbs[idx].end());
        std::unordered_set<int> nbs_visited;
        nbs_visited.insert(int(idx));

        labels[idx] = cluster_label;
        ++progress_bar;
        while (!nbs_next.empty()) {
            int nb = *nbs_next.begin();
            nbs_next.erase(nbs_next.begin());
            nbs_visited.insert(nb);

            // Noise label.
            if (labels[nb] == -1) {
                labels[nb] = cluster_label;
                ++progress_bar;
            }
            // Not undefined label.
            if (labels[nb] != -2) {
                continue;
            }
            labels[nb] = cluster_label;
            ++progress_bar;

            if (nbs[nb].size() >= min_points) {
                for (int qnb : nbs[nb]) {
                    if (nbs_visited.count(qnb) == 0) {
                        nbs_next.insert(qnb);
                    }
                }
            }
        }

        cluster_label++;
    }

    utility::LogDebug("Done Compute Clusters: {:d}", cluster_label);
    return labels;
}

std::vector<int> PointCloud::ClusterDBSCAN(double eps,
                                           size_t min_points,
                                           const std::vector<Eigen::Vector3d>& seeds, 
                                           bool print_progress) const {
    KDTreeFlann kdtree(*this);
    std::vector<Eigen::Vector3d> pointseeds;
    pointseeds.reserve(points_.size()+seeds.size());
    pointseeds.insert(pointseeds.end(), seeds.begin(), seeds.end());
    pointseeds.insert(pointseeds.end(), points_.begin(), points_.end());

    // Precompute all neighbors.
    utility::LogDebug("Precompute neighbors.");
    utility::ProgressBar progress_bar(pointseeds.size(), "Precompute neighbors.",
                                      print_progress);
    std::vector<std::vector<int>> nbs(pointseeds.size());
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
    for (int idx = 0; idx < int(pointseeds.size()); ++idx) {
        std::vector<double> dists2;
        kdtree.SearchRadius(pointseeds[idx], eps, nbs[idx], dists2);

#pragma omp critical(ClusterDBSCAN)
        { ++progress_bar; }
    }
    utility::LogDebug("Done Precompute neighbors.");
    int cluster_label = 0;
    // Set all labels to undefined (-2).
    std::vector<int> seed_labels(seeds.size(), cluster_label);
    std::vector<int> point_labels(points_.size(), -2);
    std::vector<int> labels;
    labels.reserve(pointseeds.size());
    labels.insert(labels.end(), seed_labels.begin(), seed_labels.end());
    labels.insert(labels.end(), point_labels.begin(), point_labels.end());

    for(size_t idx = 0; idx < seeds.size(); ++idx) {
        for(const int& next_idx : nbs[idx]) {
            labels[next_idx] = cluster_label;
        }
    }
    utility::LogDebug("Done compute cluster.");
    return labels;
}

}  // namespace geometry
}  // namespace open3d
