// Copyright 2019 the Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Co-developed by Tier IV, Inc. and Apex.AI, Inc.
 
#ifndef NDT__NDT_LOCALIZER_HPP_
#define NDT__NDT_LOCALIZER_HPP_
 
#include <localization_common/localizer_base.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <ndt/ndt_common.hpp>
#include <ndt/ndt_optimization_problem.hpp>
#include <optimization/optimizer_options.hpp>
#include <utility>
#include <string>
 
namespace autoware
{
namespace localization
{
namespace ndt
{
using CloudT = sensor_msgs::msg::PointCloud2;
  
 template<
   typename ScanT,
   typename MapT,
   typename NDTOptimizationProblemT,
   typename OptimizationProblemConfigT,
   typename OptimizerT>
 class NDT_PUBLIC NDTLocalizerBase : public localization_common::RelativeLocalizerBase<CloudT,
     CloudT, localization_common::OptimizedRegistrationSummary>
 {
 public:
   using Transform = geometry_msgs::msg::TransformStamped;
   using PoseWithCovarianceStamped = geometry_msgs::msg::PoseWithCovarianceStamped;
  
   explicit NDTLocalizerBase(
     const NDTLocalizerConfigBase & config,
     const OptimizationProblemConfigT & optimization_problem_config,
     const OptimizerT & optimizer,
     ScanT && scan,
     MapT && map)
   : m_config{config},
     m_optimization_problem_config{optimization_problem_config},
     m_optimizer{optimizer},
     m_scan{std::forward<ScanT>(scan)},
     m_map{std::forward<MapT>(map)} {}
  
   RegistrationSummary register_measurement_impl(
     const CloudT & msg,
     const Transform & transform_initial, PoseWithCovarianceStamped & pose_out) override
   {
     validate_msg(msg);
     validate_guess(msg, transform_initial);
     // Initial checks passed, proceed with initialization
     // Eigen representations to be used for internal computations.
     EigenPose<Real> eig_pose_initial, eig_pose_result;
     eig_pose_initial.setZero();
     eig_pose_result.setZero();
     // Convert the ros transform/pose to eigen pose vector
     transform_adapters::transform_to_pose(transform_initial.transform, eig_pose_initial);
  
     // Set the scan
     m_scan.clear();
     m_scan.insert(msg);
  
     // Define and solve the problem.
     NDTOptimizationProblemT problem(m_scan, m_map, m_optimization_problem_config);
     const auto opt_summary = m_optimizer.solve(problem, eig_pose_initial, eig_pose_result);
  
     if (opt_summary.termination_type() == common::optimization::TerminationType::FAILURE) {
       throw std::runtime_error(
               "NDT localizer has likely encountered a numerical "
               "error during optimization.");
     }
  
     // Convert eigen pose back to ros pose/transform
     transform_adapters::pose_to_transform(
       eig_pose_result,
       pose_out.pose.pose);
  
     pose_out.header.stamp = msg.header.stamp;
     pose_out.header.frame_id = map_frame_id();
  
     // Populate covariance information. It is implementation defined.
     set_covariance(problem, eig_pose_initial, eig_pose_result, pose_out);
     return localization_common::OptimizedRegistrationSummary{opt_summary};
   }
  
   void set_map_impl(const CloudT & msg) override
   {
     m_map.clear();
     m_map.insert(msg);
   }
  
   void insert_to_map_impl(const CloudT & msg) override
   {
     m_map.insert(msg);
   }
  
   const ScanT & scan() const noexcept
   {
     return m_scan;
   }
   const MapT & map() const noexcept
   {
     return m_map;
   }
   const OptimizerT & optimizer() const noexcept
   {
     return m_optimizer;
   }
   const NDTLocalizerConfigBase & config() const noexcept
   {
     return m_config;
   }
   const OptimizationProblemConfigT & optimization_problem_config() const noexcept
   {
     return m_optimization_problem_config;
   }
   const std::string & map_frame_id() const noexcept override
   {
     return m_map.frame_id();
   }
  
   std::chrono::system_clock::time_point map_stamp() const noexcept override
   {
     return m_map.stamp();
   }
  
 protected:
   virtual void set_covariance(
     const NDTOptimizationProblemT & problem,
     const EigenPose<Real> & initial_guess,
     const EigenPose<Real> & pose_result,
     PoseWithCovarianceStamped & solution) const
   {
     (void) problem;
     (void) initial_guess;
     (void) pose_result;
     (void) solution;
     // For now, do nothing.
   }
  
   virtual void validate_msg(const CloudT & msg) const
   {
     const auto message_time = ::time_utils::from_message(msg.header.stamp);
     // Map shouldn't be newer than a measurement.
     if (message_time < m_map.stamp()) {
       throw std::logic_error(
               "Lidar scan should not have a timestamp older than the timestamp of"
               "the current map.");
     }
   }
  
   virtual void validate_guess(const CloudT & msg, const Transform & transform_initial) const
   {
     const auto message_time = ::time_utils::from_message(msg.header.stamp);
  
     const auto guess_scan_diff = ::time_utils::from_message(transform_initial.header.stamp) -
       message_time;
     const auto stamp_tol = m_config.guess_time_tolerance();
     // An initial estimate should be comparable in time to the measurement's time stamp
     if (std::abs(
         std::chrono::duration_cast<std::decay_t<decltype(stamp_tol)>>(guess_scan_diff).
         count()) >
       std::abs(stamp_tol.count()))
     {
       throw std::domain_error(
               "Initial guess is not within: " + std::to_string(stamp_tol.count()) +
               "ns range of the scan's time stamp. Either increase the tolerance range or"
               "make sure the localizer takes in timely initial pose guesses.");
     }
   }
  
 private:
   NDTLocalizerConfigBase m_config;
   OptimizationProblemConfigT m_optimization_problem_config;
   OptimizerT m_optimizer;
   ScanT m_scan;
   MapT m_map;
 };
  
 template<typename OptimizerT, typename MapT = StaticNDTMap>
 class NDT_PUBLIC P2DNDTLocalizer : public NDTLocalizerBase<
     P2DNDTScan, MapT, P2DNDTOptimizationProblem<MapT>, P2DNDTOptimizationConfig, OptimizerT>
 {
 public:
   using CloudT = sensor_msgs::msg::PointCloud2;
   using ParentT = NDTLocalizerBase<
     P2DNDTScan, MapT, P2DNDTOptimizationProblem<MapT>, P2DNDTOptimizationConfig, OptimizerT>;
   using Transform = typename ParentT::Transform;
   using PoseWithCovarianceStamped = typename ParentT::PoseWithCovarianceStamped;
   using ScanT = P2DNDTScan;
  
   explicit P2DNDTLocalizer(
     const P2DNDTLocalizerConfig & config,
     const OptimizerT & optimizer,
     const Real outlier_ratio)
   : ParentT{
       config,
       P2DNDTOptimizationConfig{outlier_ratio},
       optimizer,
       ScanT{config.scan_capacity()},
       MapT{config.map_config()}} {}
  
 protected:
   void set_covariance(
     const P2DNDTOptimizationProblem<MapT> &,
     const EigenPose<Real> &,
     const EigenPose<Real> &,
     PoseWithCovarianceStamped &) const override
   {
     // For now, do nothing.
   }
 };
  
 }  // namespace ndt
 }  // namespace localization
 }  // namespace autoware
  
 #endif  // NDT__NDT_LOCALIZER_HPP_