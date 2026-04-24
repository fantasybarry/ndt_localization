# NDT Localization — Core Design

A distilled reference covering the essential ideas behind the NDT localizer: representation, scoring, optimization, quality control, and real-time architecture. Implementation details, tuning history, and project-specific parameters are intentionally omitted — see `ndt_workflow.md` for those.

---

## 1. The problem

Given:
- A prior 3-D point cloud map `M` of the environment (built offline).
- A live lidar scan `S = {p₁, …, pₙ}` taken every 100 ms.
- An initial pose guess `T₀ ∈ SE(3)` (from constant-velocity prediction + IMU).

Find the rigid transform `T* ∈ SE(3)` that aligns `S` onto `M`. The estimate becomes the vehicle's 6-DoF pose in the map frame.

**Two stages:**

| Stage | When | Input | Output |
|---|---|---|---|
| A. Build NDT map | once at startup | PCD `M` | voxel grid of Gaussians |
| B. Align scan | every 100 ms | `S`, `T₀` | refined pose `T*` |

---

## 2. Representation — voxel grid of Gaussians

Stage A replaces the raw point cloud with a **voxel grid** where each occupied cell stores a local Gaussian `(μᵢ, Σᵢ⁻¹)`:

```
μᵢ  =  (1/k) · Σⱼ pⱼ                        # mean of points in voxel i
Σᵢ  =  (1/(k−1)) · Σⱼ (pⱼ − μᵢ)(pⱼ − μᵢ)ᵀ   # their covariance
Σᵢ⁻¹ = inv(Σᵢ)                              # pre-inverted for fast scoring
```

Three jobs the voxel grid does at once:

1. **Compression.** A map with millions of points collapses to tens of thousands of Gaussians (~20× smaller) while preserving the surface geometry relevant to alignment.
2. **O(1) lookup.** A point at `(x, y, z)` maps to a voxel index by integer arithmetic:
   ```
   ix = floor((x − min_x) / v)
   iy = floor((y − min_y) / v)
   iz = floor((z − min_z) / v)
   ```
   No kd-tree, no nearest-neighbor search. This is the main reason NDT is 10–50× faster than ICP on large maps.
3. **Smoothing.** Discrete points become a smooth, differentiable likelihood field — which is what enables analytic gradients and Hessians in Stage B.

**Bounding box.** The grid is defined by `(min_point, max_point, voxel_size)`. The bounding box serves simultaneously as the **coordinate origin** for indexing, the **memory cap** for the grid, the **ROI filter** (points outside are discarded), and a **numerical-precision guard** for floats.

**Occupancy threshold.** A voxel must contain ≥ 5 points to be assigned a Gaussian; fewer points yield a rank-deficient `Σᵢ`. Sparse voxels are skipped.

---

## 3. Score function

For a candidate pose `T`, the NDT score sums per-point likelihoods:

```
score(T)  =  Σₚ∈S  φ( T·p )
φ(x)      =  exp( −½ · (x − μᵢ)ᵀ Σᵢ⁻¹ (x − μᵢ) )
```

where `i` is the voxel containing the transformed point `T·p`. Points that land in empty voxels contribute 0.

**Interpretation.** `φ(x)` is (proportional to) the Gaussian probability density that the map surface passes through `x`. Summing over all scan points measures "how well does this scan sit on the map's surfaces?" Higher is better.

---

## 4. Optimization — Newton's method

Stage B maximizes `score(T)` starting from `T₀`. At each iteration `k`:

```
T_{k+1}  =  T_k ⊕ α·Δξ
```

where `Δξ ∈ ℝ⁶` is a twist in the Lie algebra `se(3)` representing small rigid motions `(δx, δy, δz, δroll, δpitch, δyaw)`.

### The Newton step

Take a local quadratic approximation of the score:

```
score(T_k ⊕ Δξ)  ≈  score(T_k)  +  gᵀΔξ  +  ½ ΔξᵀHΔξ
```

with

- `g = ∂score/∂ξ`  — gradient (6×1)
- `H = ∂²score/∂ξ∂ξᵀ` — Hessian (6×6)

Maximizing the quadratic w.r.t. `Δξ`:

```
H·Δξ  =  −g
```

Both `g` and `H` are **analytic** — closed-form derivatives of the Gaussian exponent and the `se(3)` exponential map. No numerical differentiation; no nearest-neighbor search. Each iteration is `O(|S|)` to assemble `g` and `H`, plus a tiny 6×6 linear solve.

### Line search

The quadratic approximation is only accurate near `T_k`. A **Wolfe line search** scales the step by `α ∈ [0, step_size]` so the actual update never overshoots:

```
T_{k+1}  =  T_k ⊕ α·Δξ     with α chosen by line search
```

### Convergence

Stop when either

```
‖α·Δξ‖ < ε          (update below tolerance — at a local peak)
k ≥ max_iterations  (safety cap on pathological cases)
```

Typical behavior after lock-on: **3–10 iterations, 10–30 ms per scan.**

### Why Newton over gradient descent

Gradient descent uses only the slope → thousands of tiny steps. Newton uses curvature as well → a handful of large, correctly-sized steps. NDT's Gaussian sum is locally quadratic, so Newton converges in single-digit iterations. The Hessian is cheap because it's analytic.

### Why a Lie-algebra twist `ξ`

Rotations do not add (you cannot linearly sum two yaws and get a valid rotation matrix). The Lie algebra `se(3)` is the tangent space at the current pose where small rigid motions *do* add linearly. The exponential map carries `ξ` back to a 4×4 matrix. This is what makes "take the gradient of score w.r.t. pose" a well-posed operation.

---

## 5. Quality firewall

NDT is a *local* optimizer. "Converged" does not mean "correct." A three-gate filter rejects untrustworthy outputs:

| Gate | Test | Catches |
|---|---|---|
| **Fitness** | mean-squared nearest-neighbor distance from aligned scan to map `< τ_fit` | bad initial guess; scan landed in geometrically empty region |
| **Prediction consistency** | `‖ndt_result − predict_pose‖ < τ_pred` | *aliased geometry* — scan locked onto a locally-good but globally-wrong minimum (e.g. a parallel street) |
| **Convergence** | `hasConverged() == true` | optimizer timed out without settling |

**On rejection:** keep the previous pose, publish nothing, do not update state. The layered design is deliberate — each gate catches a failure mode the others miss, and together they let downstream consumers rely on a simple invariant: *every published pose has passed all three gates.*

**First-accept bypass.** The prediction-consistency check is skipped on the first scan (no prior NDT result to compare against).

---

## 6. Real-time architecture

The localizer runs at **10 Hz** (100 ms budget per scan). The core architectural choices:

### Concurrency: main thread + one worker

- `on_scan()` runs on the ROS callback thread. It preprocesses, predicts `T₀`, publishes the live scan at `T₀` (for visualization), then either spawns a worker or drops the scan.
- The worker thread runs the expensive NDT alignment plus the quality-firewall checks.

### The busy flag

An atomic `ndt_busy_` coordinates the two threads:

```
on_scan():
    if ndt_busy_: drop this scan; return
    ndt_busy_ = true
    spawn worker(scan_snapshot, prediction_snapshot)

worker():
    try:
        T* = ndt.align(scan, T₀)
    finally:
        ndt_busy_ = false              # release BEFORE publishing
    run_quality_checks_and_publish(T*)
```

Two invariants this preserves:
- **Never two aligners at once.** The PCL NDT object is not thread-safe; a concurrent `align()` would corrupt its internal state.
- **Fast release.** The flag is cleared the moment NDT returns, not after publish. Post-processing of the current scan can overlap with NDT of the *next* scan.

### Drop-not-queue policy

If a scan arrives while NDT is still running, it is dropped — not queued. This is a deliberate real-time design:

| | Drop | Queue |
|---|---|---|
| Pose age at publish | ≤ one NDT cycle | unbounded, grows with every slow frame |
| Memory | bounded | unbounded |
| Recovery after slow period | immediate | never; backlog compounds |
| Skipped measurements? | yes | no |

A pose from 2 s ago is worse than no pose — at highway speed the vehicle has moved 60 m. Queuing would produce compounding lag: each slow NDT run processes an older scan, and the system never catches up. Dropping keeps the output bounded to at most one NDT cycle of latency and self-heals the moment load eases.

This is the standard "newest message wins" pattern for real-time sensor fusion.

### Why the prediction + live-scan publish happen *before* the busy check

So that even when NDT drops this scan, RViz still sees the vehicle move (at the predicted pose `T₀`). The published TF stays fresh from the last accepted NDT result; the live scan follows the prediction. The visualization stays smooth under load.

---

## 7. Data flow

```
 ┌───────────────┐    [startup]     ┌──────────────────┐
 │  Prior map M  │ ───────────────▶ │ Build voxel grid │
 │   (PCD)       │                  │  {(μᵢ, Σᵢ⁻¹)}    │
 └───────────────┘                  └─────────┬────────┘
                                              │
                                              ▼
 ┌───────────────┐   preprocess    ┌────────────────────┐
 │ Live scan S   │ ──────────────▶ │  T₀ prediction     │
 │  (10 Hz)      │  (NaN/crop/vox) │  (const vel + IMU) │
 └───────────────┘                 └─────────┬──────────┘
                                              │
                                              ▼
                              ┌───────────────────────────────┐
                              │  Newton's method on score(T)  │
                              │  H·Δξ = −g, line search,      │
                              │  analytic derivatives         │
                              └───────────────┬───────────────┘
                                              │
                                              ▼
                              ┌───────────────────────────────┐
                              │  Quality firewall             │
                              │   fitness / prediction-dist / │──(reject)──► keep previous pose
                              │   convergence                 │
                              └───────────────┬───────────────┘
                                              │ accept
                                              ▼
                              ┌───────────────────────────────┐
                              │  Publish pose, TF, diagnostics│
                              │  Update velocity for next T₀  │
                              └───────────────────────────────┘
```

---

## 8. Why NDT (vs ICP) in one table

| Property | ICP | NDT |
|---|---|---|
| Map representation | raw points (kd-tree) | voxel grid of Gaussians |
| Correspondence search | per-iteration nearest neighbor | none — voxel index lookup |
| Cost per iteration | `O(|S| log |M|)` | `O(|S|)` |
| Derivatives | numerical / point-to-plane heuristics | analytic gradient & Hessian |
| Robustness to noise | sensitive | smoothed by Gaussians |
| Convergence steps | tens to hundreds | 3–10 |
| Scales with | map point count | voxel count |

For large prior maps with a reasonable initial guess, NDT is typically 10–50× faster than ICP at comparable accuracy. The price is the need for a voxel grid at build time and a usable `T₀` at scan time — both cheap.

---

## 9. Invariants the design guarantees

| Invariant | Enforced by |
|---|---|
| At most one NDT alignment in flight | `ndt_busy_` atomic flag |
| Every published pose passed all quality checks | Section 5 firewall with no-publish-on-reject |
| Pose latency ≤ one NDT cycle under overload | drop-not-queue |
| `ndt_busy_` always releases, even on exception | `try/catch` around `align()` |
| Downstream consumers always see the last *good* pose | latched TF + publish-on-accept-only |

These five invariants together are what distinguishes a research implementation from a localizer that survives real-world edge cases.
