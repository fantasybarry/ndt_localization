#ifndef NDT__NDT_OPTIMIZATION_HPP_
#define NDT__NDT_OPTIMIZATION_HPP_

#include "ndt/ndt_map.hpp"
#include "ndt/ndt_scan.hpp"
#include "common/types.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <tuple>
#include <type_traits>

namespace autoware
{
namespace localization
{
namespace ndt
{

// ============================================================================
// Expression<Derived, DomainValueT> - CRTP interface
// ============================================================================

/// CRTP base for objective expressions that provide score, Jacobian and Hessian
///   - operator()(x) ->double                (score / cost)
///   - jacobian(x)   ->Eigen::Matrix         (first derivative)
///   - hessian(x)    ->Eigen::Matrix         (second derivative)
template <typename Derived, typename DomainValueT>
class Expression
{
public:
  using Value = DomainValueT;
  
  double operator()(const DomainValueT & x) const
  {
    return static_cast<const Derived &>(*this).score_(x);
  }

  auto jacobian(const DomainValueT & x) const
  {
    return static_cast<const Derived &>(*this).jacobian_(x);
  }
  
  auto hessian(const DomainValueT & x) const
  {
    return static_cast<const Derived &>(*this).hessian_(x);
  }

protected:
  ~Expression() = default;
};

// ============================================================================
// OptimizationProblem<Derived, DomainValueT, ObjectiveT,
//                      EqualityConstrainsT, InequalityConstraintsT>
// ============================================================================

/// CRTP base that bundles an objective expression with equality and inequality
/// constraints (stored as std::tuple<>)
template <
  typename Derived,
  typename DomainValueT,
  typename ObjectiveT,
  typename EqualityConstraintsT = std::tuple<>,
  typename InequalityConstraintsT = std::tuple<>>
class OptimizationProblem
{
public:
  using DomainValue = DomainValueT;
  using Objective = ObjectiveT;

  explicit OptimizationProblem(
    ObjectiveT objective,
    EqualityConstraintsT eq = {},
    InequalityConstraintsT ineq = {})
  : m_objective(std::move(objective)),
    m_equality_constraints(std::move(eq)),
    m_inequality_constraints(std::move(ineq)) {}
  
  const ObjectiveT & objective() const noexcept { return m_objective; }
  ObjectiveT & objective() noexcept { return m_objective; }

  const EqualityConstraintsT & equality_constraints() const noexcept
  {
    return m_equality_constraints;
  }

  const InequalityConstraintsT & inequality_constraints() const noexcept
  {
    return m_inequality_constraints;
  }

protected:
  ~OptimizationProblem() = default;

private:
  ObjectiveT m_objective;
  EqualityConstraintsT m_equality_constraints;
  InequalityConstraintsT m_inequality_constraints;
};

// =======================================================================
// UnconstrainedOptimizationProblem - binds both constraint tuples to empty
// =======================================================================

/// Convenience base for problems with no equality or inequality constraints
template <typename Derived, typename DomainValueT, typename ObjectiveT>
class UnconstrainedOptimizationProblem
  : public OptimizationProblem<
      Derived, DomainValueT, ObjectiveT, std::tuple<>, std::tuple<>>
{
public:
  using Base = OptimizationProblem<
    Derived, DomainValueT, ObjectiveT, std::tuple<>, std::tuple<>>;

  explicit UnconstrainedOptimizationProblem(ObjectiveT objective)
  : Base(std::move(objective)) {}

protected:
  ~UnconstrainedOptimizationProblem() = default;
};


// ==========================================================================
// P2DNDTObjective - concrete Expression for Point-to-Distribution NDT
// ==========================================================================

/// The P2D NDT cost function, computing score, Jacobian, and Hessian
/// using the CachedExpression pattern (Magnusson 2009, Section 6.2)
///
/// All three quantities share intermediate exponential terms, so they are
/// computed together for efficiency.
// EigenPose is defined in ndt_common.hpp (via ndt_map.hpp).

class P2DNDTObjective
  : public Expression<P2DNDTObjective, EigenPose>
{
public:
  using DomainValue = EigenPose;

  /// Bind a scan and map reference before evaluating
  void set_scan(const P2DNDTScan * scan) { scan_ = scan; }

  template <typename MapT>
  void set_map(const MapT * map);
  
  void set_search_radius(double r) { search_radius_ = r; }

  // -- Expression interface implementation --
  
  double score_(const DomainValue & x) const;

  Eigen::Matrix<double, 6, 1> jacobian_(const DomainValue & x) const;
  Eigen::Matrix<double, 6, 6> hessian_(const DomainValue & x) const;

  /// Evaluate score, Jacobian, and Hessian in a single pass (preferred path).
  struct Result
  {
    double score = 0.0;
    Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
  };

  template <typename MapT>
  Result evaluate(
    const P2DNDTScan & scan,
    const MapT & map,
    const DomainValue & transform,
    double search_radius = 2.0) const;

  /// Convert a 6-DoF vector into a 4x4 homogeneous transform.
  static Eigen::Matrix4d to_matrix(const DomainValue & t);

  /// Build the 3x6 Jacobian of a transformed point w.r.t. the 6-DoF parameters.
  static Eigen::Matrix<double, 3, 6> point_jacobian(
    const Eigen::Vector3d & point,
    const DomainValue & transform);


private:
  /// Score contribution from a single (point, voxel) pair - CachedExpression.
  /// Takes the centroid and inverse covariance directly to avoid double-inversion.
  Result score_point_voxel(
    const Eigen::Vector3d & transformed_point,
    const Eigen::Vector3d & centroid,
    const Eigen::Matrix3d & cov_inv,
    const Eigen::Matrix<double, 3, 6> & jacobian_of_point) const;

  const P2DNDTScan * scan_ = nullptr;
  double search_radius_ = 2.0;
};

// ============================================================================
// P2DNDTOptimizationProblem - unconstrained problem wrapping the P2D objective
// ============================================================================

class P2DNDTOptimizationProblem
  : public UnconstrainedOptimizationProblem<
      P2DNDTOptimizationProblem, EigenPose, P2DNDTObjective>
{
public:
  P2DNDTOptimizationProblem()
  : UnconstrainedOptimizationProblem(P2DNDTObjective{}) {}

  explicit P2DNDTOptimizationProblem(P2DNDTObjective obj)
  : UnconstrainedOptimizationProblem(std::move(obj)) {}

  /// Convenience: evaluate the objective for a given scan, map, and transform.
  template <typename MapT>
  P2DNDTObjective::Result evaluate(
    const P2DNDTScan & scan,
    const MapT & map,
    const EigenPose & transform,
    double search_radius = 2.0) const
  {
    return objective().evaluate(scan, map, transform, search_radius);
  }
};

// ==========================================================================
// LineSearch<Derived> -- CRTP interface
// ==========================================================================

/// CRTP base for line search strategies that determine the step length
template <typename Derived>
class LineSearch
{
public:
  /// Compute the step length given a direction, current value, gradient, etc.
  double compute_step_length() const
  {
    return static_cast<const Derived &>(*this).compute_step_length_();
  }

protected:
  ~LineSearch() = default;
};

/// Trivial line search that always returns a fixed step size.
class FixedLineSearch : public LineSearch<FixedLineSearch>
{
public:
  explicit FixedLineSearch(double step = 1.0) : step_(step) {}

  double compute_step_length_() const { return step_; }

private:
  double step_;
};


// ===========================================================================
// Optimizer<Derived, LineSearchT> — CRTP interface
// ===========================================================================

/// CRTP base for iterative optimisers.
///   solve(problem, x0, x_out) runs the optimisation loop.
template <typename Derived, typename LineSearchT>
class Optimizer
{
public:
  explicit Optimizer(LineSearchT line_search = {})
  : line_search_(std::move(line_search)) {}

  template <typename OptimizationProblemT>
  void solve(
    OptimizationProblemT & optimization_problem,
    const EigenPose & x0,
    EigenPose & x_out) const
  {
    static_cast<const Derived &>(*this).solve_(optimization_problem, x0, x_out);
  }

  const LineSearchT & line_search() const noexcept { return line_search_; }

protected:
  ~Optimizer() = default;
  LineSearchT line_search_;
};

// ===========================================================================
// NewtonsMethod — concrete Optimizer using Newton / Gauss-Newton steps
// ===========================================================================

struct NewtonsMethodParams
{
  int max_iterations = 30;
  double step_size = 1.0;
  double epsilon = 1e-4;       // convergence threshold on parameter change
  double score_epsilon = 1e-6; // convergence threshold on score change
};

template <typename LineSearchT = FixedLineSearch>
class NewtonsMethod : public Optimizer<NewtonsMethod<LineSearchT>, LineSearchT>
{
public:
  using Base = Optimizer<NewtonsMethod<LineSearchT>, LineSearchT>;

  explicit NewtonsMethod(
    const NewtonsMethodParams & params = {},
    LineSearchT line_search = {})
  : Base(std::move(line_search)), params_(params) {}

  struct Result
  {
    EigenPose transform = EigenPose::Zero();
    double final_score = 0.0;
    int iterations = 0;
    bool converged = false;
  };

  /// Run Newton optimisation on an NDT problem.
  template <typename ScanT, typename MapT>
  Result optimize(
    const ScanT & scan,
    const MapT & map,
    const EigenPose & initial_estimate,
    const P2DNDTOptimizationProblem & problem) const
  {
    Result res;
    res.transform = initial_estimate;
    double prev_score = 0.0;

    for (int i = 0; i < params_.max_iterations; ++i) {
      auto eval = problem.evaluate(scan, map, res.transform);
      res.final_score = eval.score;
      res.iterations = i + 1;

      Eigen::Matrix<double, 6, 6> H = eval.hessian;
      H.diagonal().array() += 1e-6;

      Eigen::Matrix<double, 6, 1> delta = H.ldlt().solve(-eval.gradient);
      delta *= this->line_search().compute_step_length();

      res.transform += delta;

      if (delta.norm() < params_.epsilon) {
        res.converged = true;
        break;
      }
      if (i > 0 && std::abs(eval.score - prev_score) < params_.score_epsilon) {
        res.converged = true;
        break;
      }
      prev_score = eval.score;
    }
    return res;
  }

  const NewtonsMethodParams & params() const noexcept { return params_; }

private:
  NewtonsMethodParams params_;
};

// ---------------------------------------------------------------------------
// Backward-compatible aliases
// ---------------------------------------------------------------------------
using OptimizerParams = NewtonsMethodParams;
using NewtonOptimizer = NewtonsMethod<FixedLineSearch>;
using P2DOptimizationProblem = P2DNDTOptimizationProblem;

}  // namespace ndt
}  // namespace localization
}  // namespace autoware

#endif  // NDT__NDT_OPTIMIZATION_HPP_
