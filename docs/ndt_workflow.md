# NDT (Normal Distributions Transform) Localization — Step-by-Step Workflow

This document walks through every stage of the NDT localization pipeline in this project — from loading the prior map at startup, to registering each live lidar scan, to publishing the estimated pose. Each section includes the math, the pseudocode, the relevant parameters, and the reason each step exists.

---

## Contents

1. [High-level overview](#1-high-level-overview)
2. [Notation & coordinate frames](#2-notation--coordinate-frames)
3. [Stage A — Offline: build the NDT map (once at startup)](#3-stage-a--offline-build-the-ndt-map-once-at-startup)
4. [Stage B — Online: align each scan (every 100 ms)](#4-stage-b--online-align-each-scan-every-100-ms)
   - 4.1 Preprocess the scan
   - 4.2 Predict the initial pose
   - 4.3 Evaluate the NDT score
   - 4.4 Optimize with Newton's method
   - 4.5 Quality checks — accept or reject
   - 4.6 Update state and publish
5. [Data flow / block diagram](#5-data-flow--block-diagram)
6. [Per-scan timeline](#6-per-scan-timeline)
7. [Parameters and how they affect behavior](#7-parameters-and-how-they-affect-behavior)
8. [Why NDT instead of ICP](#8-why-ndt-instead-of-icp)
9. [Failure modes and how the pipeline handles them](#9-failure-modes-and-how-the-pipeline-handles-them)

---

## 1. High-level overview

**Goal:** continuously estimate the vehicle's 6-DoF pose in a prior 3D map, using only a live lidar scan and an IMU.

**Core idea:** represent the prior map as a field of local 3-D Gaussians (one per voxel), then find the rigid transform that makes the live scan points land on high-probability regions of that field.

**Two stages:**

| Stage | When | Input | Output |
|---|---|---|---|
| **A. Build NDT map** | once at startup | prior PCD file (`pointcloud_map.pcd`) | set of voxels, each `{μᵢ, Σᵢ⁻¹}` |
| **B. Align scan** | every 100 ms | live scan + prediction | 6-DoF pose in map frame |

---

## 2. Notation & coordinate frames

**Variables**

| Symbol | Meaning |
|---|---|
| `M` | target point cloud (the prior map) |
| `S = {p₁, p₂, …, pₙ}` | source point cloud (a single live lidar scan) |
| `Vᵢ` | the *i*-th voxel of the NDT grid |
| `μᵢ ∈ ℝ³` | mean of the map points in voxel `Vᵢ` |
| `Σᵢ ∈ ℝ³ˣ³` | covariance of the map points in voxel `Vᵢ` |
| `T ∈ SE(3)` | candidate rigid transform (4×4 matrix), 6 parameters `(x, y, z, roll, pitch, yaw)` |
| `T·p` | point `p` transformed by `T` |
| `T*` | optimal transform found by NDT |

**Frames in this project**

```
earth  ──(static, from map_info.yaml ECEF)──▶  map
map    ──(published by localizer every scan)──▶  base_link
base_link ──(static from launch file)──▶  velodyne
```

---

## 3. Stage A — Offline: build the NDT map (once at startup)

Executed once by the `ndt_map_publisher` node when it reads the PCD file.

### Step A1 — Load the prior point cloud

```
M  ←  read PCD(/home/mila/Desktop/pointcloud_map.pcd)
```

In this project, `M` contains **1,977,212 points**.

### Step A2 — Define the voxel grid

This step slices 3D space into a regular grid of cubic cells, because NDT will compute **one Gaussian per occupied cell** in the next step. Think of it as laying a 3D "egg-crate" over the point cloud — each compartment will later be summarized by a single Gaussian `(μᵢ, Σᵢ)` computed from the map points that fell inside it.

Parameters from `map_publisher.param.yaml`:

```yaml
min_point:  (-500, -500, -10)
max_point:  ( 500,  500,  50)
voxel_size: (  1.0,  1.0, 1.0)   # cell edge length
```

The three questions this step answers:

1. **Where does the grid cover?** — The bounding box from `min_point` to `max_point`. Any point in the prior PCD outside this box is discarded. Here that is a 1000 m × 1000 m × 60 m volume centred at the map origin.
2. **How big is one cell?** — `voxel_size = 1.0 m`, so each cube is 1 m on a side. Smaller cells capture finer geometry but produce more voxels; larger cells are coarser but faster.
3. **How many cells at most?** — Grid volume ÷ cell volume:

```
Δx × Δy × Δz / v³  =  1000 × 1000 × 60 / 1  =  60 M possible cells
```

Only **occupied** cells are actually stored, so for this project's map you end up with ~98 k voxels out of the 60 M potential slots.

**Two independent grids in this project:**

| Grid | Used by | Parameter | Value | Purpose |
|---|---|---|---|---|
| Map grid | `ndt_map_publisher` | `voxel_size` | 1.0 m | compression + visualization + storage |
| Alignment grid | `p2d_ndt_localizer_exe` | `ndt_resolution` | 2.0 m | the Gaussian field that Newton's method optimizes against |

They are built independently — the map side is what gets serialized and published on `/localization/ndt_map`; the alignment side is rebuilt internally by the localizer when it loads the map. Using a coarser 2 m resolution on the alignment side gives a larger basin of convergence (helpful when the initial pose guess is a few meters off), at the cost of slightly coarser fit.

### Step A3 — For each voxel that has ≥ 5 points

Let `{p₁, …, pₖ}` be the points of `M` that fall in voxel `Vᵢ`.

**Compute the mean:**

```
μᵢ  =  (1/k) · Σⱼ pⱼ
```

**Compute the covariance:**

```
Σᵢ  =  (1/(k−1)) · Σⱼ (pⱼ − μᵢ) (pⱼ − μᵢ)ᵀ
```

**Pre-invert it:**

```
Σᵢ⁻¹  =  inv(Σᵢ)
```

Why pre-invert: during online scoring, every source point looks up `Σᵢ⁻¹` and does a single 3×3 matrix-vector product. Doing the inversion at build time turns scan-time inversions into fast lookups.

Why ≥ 5 points threshold: covariance of fewer than ~5 points is rank-deficient and produces a singular `Σᵢ`. Skipping them avoids numerical explosions.

### Step A4 — Store and publish

Each voxel is serialized with fields `{x, y, z, icov_xx, icov_xy, icov_xz, icov_yy, icov_yz, icov_zz, cell_id}`.

The final NDT map in this project: **98,610 voxels** representing 1.97 M points (≈ 20× compression).

```
/localization/ndt_map            ← voxel cloud for localizers  (transient_local)
/localization/pointcloud_map     ← raw XYZ map for reference   (transient_local)
/localization/viz_ndt_map        ← RViz-friendly viz map        (transient_local)
```

**Transient Local** QoS means late subscribers (like the localizer starting seconds later, or RViz reloading) still receive the map.

---

## 4. Stage B — Online: align each scan (every 100 ms)

Everything below happens inside the `on_scan()` callback of `p2d_ndt_localizer_exe` and its background worker thread.

### 4.1 Preprocess the scan

Raw Velodyne scans are noisy, huge, and contain self-returns from the vehicle itself.

**Step B1a — Remove NaN points**

Lidar can produce NaN returns when a ray hits nothing. Strip them:

```
S  ←  removeNaNFromPointCloud(raw_scan)
```

**Step B1b — Range filter (box crop)**

Keep only returns within ±80 m laterally:

```
keep p  iff  |p.x| < 80  and  |p.y| < 80
```

Why: distant returns (> 80 m) have poor range accuracy and add noise without improving the match.

**Step B1c — Voxel downsample**

Using `scan_voxel_leaf_size: 1.0 m`:

```
for each 1 m³ voxel in the scan:
    replace all points inside with their centroid
```

Typical result: a raw ~100 k-point Velodyne scan → **~29 k points**.

Why downsample: NDT cost is `O(|S| × constant)`, not `O(|S| × |M|)`, but still linear in scan size. Going from 100 k → 29 k gives ~3× speed-up with negligible accuracy loss.

**Edge case:** if the filtered scan is empty, the callback returns immediately (no NDT run).

### 4.2 Predict the initial pose `T₀`

NDT is a local optimizer — it needs `T₀` within a few voxels of truth for the gradient to be non-zero.

**Step B2a — Constant-velocity prediction**

```
Δt  =  current_scan_time − previous_scan_time
predict_pose.x  =  previous_pose.x + velocity_x × Δt
predict_pose.y  =  previous_pose.y + velocity_y × Δt
predict_pose.z  =  previous_pose.z + velocity_z × Δt
predict_pose.yaw = previous_pose.yaw + angular_velocity × Δt
```

Velocities here are derived from the delta of the last two NDT results.

**Step B2b — Fuse IMU (if `use_imu: true`)**

```
Δyaw_imu  =  imu.angular_velocity.z × Δt
predict_pose_imu.yaw += Δyaw_imu
```

(Linear acceleration is intentionally **not** double-integrated here because raw accel contains gravity and small biases, which would produce several meters per second of drift per minute.)

**Step B2c — Fuse wheel odometry (if `use_odom: true`)**

Similar structure but using wheel-odometry linear velocity instead of a derived one. Disabled in this project because the bag has no odometry topic.

The chosen prediction source becomes `T₀` (as a 4×4 matrix `initial_guess`).

**Step B2d — Bootstrap / initialization**

On startup, the localizer auto-initializes at `(0, 0, 0)`. This is usually wrong. The user supplies a better initial pose by clicking **2D Pose Estimate** in RViz, which publishes to `/initialpose`:

```
on_initial_pose(msg):
    previous_pose   ←  msg.pose
    current_pose    ←  msg.pose
    init_pose_received ← true
```

### 4.3 Evaluate the NDT score

For a candidate transform `T`, define the score over the whole scan:

```
score(T)  =  Σₚ∈S  φ( T·p )
```

where

```
φ(x)  =  exp( −½ · (x − μᵢ)ᵀ Σᵢ⁻¹ (x − μᵢ) )
```

and `i` is the index of the voxel that contains the transformed point `T·p`.

**Interpretation:** `φ(x)` is (proportional to) the Gaussian probability density that the map surface passes through `x`. Summing over all scan points measures "how well does this scan sit on the map's surfaces?"

**Practical details:**

- If `T·p` falls into a voxel that has no Gaussian (empty cell), that point contributes 0.
- In practice PCL's implementation uses a slightly modified score with robust constants to improve numerical behavior, but the structure is the same.

### 4.4 Optimize with Newton's method

NDT iteratively refines `T_k` to maximize `score(T)`:

```
T_{k+1}  =  T_k  +  Δξ
```

where the update `Δξ ∈ ℝ⁶` (a small twist in se(3)) comes from solving the Newton system:

```
H · Δξ  =  −g
```

- `g = ∂score/∂ξ` — gradient (6×1)
- `H = ∂²score/∂ξ∂ξᵀ` — Hessian (6×6), analytic (not numerical)

Because `φ(·)` is a composition of a quadratic form and an exponential, both `g` and `H` have closed-form expressions in terms of `μᵢ`, `Σᵢ⁻¹`, and the Jacobians of `T·p` w.r.t. `ξ`. This is the reason NDT is fast: **no nearest-neighbor search, analytic derivatives**.

**Step B3a — Line search**

To prevent an over-long Newton step (which could jump to a wrong basin), PCL uses a Wolfe line search bounded by `step_size: 0.5`. In practice:

```
α ← line_search(T_k, Δξ, max_step = 0.5)
T_{k+1}  =  T_k  +  α · Δξ
```

**Step B3b — Convergence test**

Stop when

```
‖T_{k+1} − T_k‖  <  epsilon  =  0.01
```

or when

```
k  ≥  max_iterations  =  30
```

**Step B3c — Typical behavior in this project**

After lock-on: **3–10 iterations, 10–30 ms per scan.**

**Pseudocode**

```
T ← T₀
for k = 0 … max_iterations − 1:
    g, H = compute_gradient_and_hessian(T, S, NDT_map)
    Δξ   = solve(H, −g)
    α    = line_search(Δξ, bound = step_size)
    T    = T + α · Δξ
    if ‖α · Δξ‖ < epsilon: break
return T, k+1, fitness(T)
```

### 4.5 Quality checks — accept or reject

After the optimizer returns, we do not trust the result blindly.

**Check 1 — Fitness score**

`getFitnessScore()` returns the **mean squared distance** from each source point to the nearest map point under the aligned transform.

```
if fitness_score > score_threshold (500):
    log "NDT fitness too high, rejecting"
    return  # keep previous pose
```

A large fitness (e.g., 17 000) means the scan is hundreds of meters away from where we think the vehicle is — almost always wrong, often caused by a bad initial guess.

**Check 2 — Prediction consistency**

```
d = ‖ndt_result.xyz − predict_pose.xyz‖
if already_accepted and d > predict_pose_threshold (15 m):
    log "NDT result too far from prediction, rejecting"
    return  # keep previous pose
```

Guards against the optimizer finding a *locally* good but *globally* wrong minimum (e.g., a parallel street with similar building geometry).

Note: the first NDT result is always accepted (no prior to compare against).

**Check 3 — Convergence**

```
if not converged:
    return  # PCL thinks the optimizer did not reach a stable solution
```

### 4.6 Update state and publish

**Step B5a — Update shared state (under mutex)**

```
diff_x  = ndt_result.x − previous_pose.x
diff_y  = ndt_result.y − previous_pose.y
diff_3d = √(diff_x² + diff_y² + diff_z²)

current_velocity_x = diff_x / Δt
current_velocity_y = diff_y / Δt
current_velocity   = diff_3d / Δt

previous_pose      = ndt_result
last_fitness_score = fitness_score
```

The new velocity feeds back into the next scan's prediction (Step B2a).

**Step B5b — Publish**

| Topic | QoS | Purpose |
|---|---|---|
| `/localization/ndt_pose` | Reliable | pose only |
| `/localization/ndt_pose_with_covariance` | Reliable | pose + cov (cov ∝ fitness) |
| `/tf` (map → base_link) | — | TF broadcast for downstream |
| `/localization/points_aligned` | **Best Effort, depth 1** | scan transformed into map, for RViz |
| `/localization/estimate_twist` | Reliable | linear + angular velocity |
| `/localization/exe_time_ms`, `/iteration_num`, `/transform_probability`, `/nvtl`, `/estimated_vel_mps`, `/estimated_vel_kmph`, … | Reliable | diagnostics |

Best-Effort QoS on `points_aligned` prevents the publisher from blocking when a slow subscriber (RViz rendering a huge cloud) can't keep up.

---

## 5. Data flow / block diagram

```
 ┌────────────────┐      ┌────────────────────┐
 │  Prior PCD     │      │  Live /velodyne    │
 │  map (1.97M)   │      │  points (10 Hz)    │
 └───────┬────────┘      └──────────┬─────────┘
         │  [startup]               │
         ▼                          ▼
 ┌────────────────┐      ┌────────────────────┐
 │ Build NDT grid │      │  Preprocess scan   │
 │ (μᵢ, Σᵢ⁻¹)    │      │  NaN-filter, crop, │
 │  98k voxels    │      │  voxel-downsample  │
 └───────┬────────┘      └──────────┬─────────┘
         │                          │
         │                          ▼
         │        ┌──────────────────────────┐
         │        │  Predict T₀              │
         │        │  prev_pose + vel·Δt      │
         │        │  ⊕ IMU yaw integration   │
         │        └──────────┬───────────────┘
         │                   │
         ▼                   ▼
 ┌──────────────────────────────────────────┐
 │   NDT score  +  Newton optimizer         │
 │   maximize  Σ φ(T·p)  analytically       │
 │   (3–10 iters, 10–30 ms typical)         │
 └──────────────────┬───────────────────────┘
                    │
                    ▼
 ┌──────────────────────────────────────┐
 │ Fitness / prediction-distance /      │
 │ convergence checks                   │──(reject)─► keep previous pose
 └──────────────────┬───────────────────┘
                    │ accept
                    ▼
 ┌──────────────────────────────────────┐
 │ Update previous_pose + velocities    │
 │ (feeds into next prediction)         │
 └──────────────────┬───────────────────┘
                    │
                    ▼
 ┌──────────────────────────────────────┐
 │ Publish                              │
 │   /ndt_pose_with_covariance          │
 │   /tf (map → base_link)              │
 │   /points_aligned                    │
 │   diagnostics                        │
 └──────────────────────────────────────┘
```

---

## 6. Per-scan timeline

What happens in real time for a single lidar frame:

```
t = 0    ms   new scan arrives in on_scan()
t = 0–2  ms   preprocess (NaN, crop, voxel) + prediction + publish live scan at T₀
t = 2    ms   if ndt_busy_ → drop this scan; else launch worker thread
              (busy-flag is released as soon as PCL finishes, so post-processing
              never blocks the next scan)
t = 5–35 ms   PCL NDT aligns inside the worker thread
t ≈ 35   ms   extract pose, release busy flag
t = 35–40 ms  fitness & prediction checks, update state, publish pose/TF
t = 40+  ms   thread exits; next scan can already be running by this point
```

Budget at 10 Hz = 100 ms per scan. Actual use in this project: ~30 ms + publish → plenty of headroom, typically ~3× real time.

---

## 7. Parameters and how they affect behavior

### Map-side (`map_publisher.param.yaml`)

| Parameter | Effect if larger | Effect if smaller |
|---|---|---|
| `voxel_size` | fewer voxels, faster, coarser map | more voxels, slower build, finer map |
| `capacity` | holds more voxels before eviction | old voxels evicted sooner |

### Localizer-side (`ndt_localizer.param.yaml`)

| Parameter | Effect if larger | Effect if smaller |
|---|---|---|
| `ndt_resolution` | fewer target voxels, faster, coarser fit | finer fit, slower, more sensitive to initial guess |
| `scan_voxel_leaf_size` | fewer scan points, faster, less accurate | more points, slower, more accurate |
| `max_iterations` | more chances to converge, slower worst case | may stop before converging |
| `step_size` | bigger jumps, can overshoot | smaller jumps, may not reach optimum in time |
| `epsilon` | stops earlier, faster, less accurate | runs longer, more accurate |
| `score_threshold` | tolerates worse matches | rejects more — risk of never accepting |
| `predict_pose_threshold` | tolerates bigger jumps | rejects outlier corrections |
| `use_imu` | helps during rotation / brief occlusion | purely scan-to-scan, no attitude aid |
| `use_odom` | great for motion prediction if available | falls back to last-NDT velocity |

**Tuning lesson learned during testing:**

- `step_size: 0.1` + `epsilon: 0.001` → optimizer reports `iter=1 converged=1` but actually didn't move (gradient × step below epsilon). Vehicle appears stuck. **Use `step_size: 0.5` + `epsilon: 0.01`** for vehicle-scale motion.
- If fitness grows monotonically (e.g. 17 000 → 22 000), the initial pose is too far from truth. Click **2D Pose Estimate** in RViz to re-seed.

---

## 8. Why NDT instead of ICP

| Property | ICP | NDT |
|---|---|---|
| Correspondence | nearest-neighbor search every iter | none — voxel lookup |
| Data structure | kd-tree on map points | pre-built Gaussian grid |
| Cost per iter | `O(|S| log |M|)` | `O(|S|)` |
| Derivatives | numerical / point-to-plane tricks | analytic gradient & Hessian |
| Robustness to noise | sensitive | smoothed by Gaussian |
| Multi-resolution | needs scaffolding | natural (change voxel size) |
| Map size tolerance | scales with `|M|` | scales with number of voxels |
| Trade-off | very accurate if well seeded | needs voxel grid + initial guess |

For large maps (millions of points) with a reasonable initial guess, NDT is typically **10–50× faster** than ICP at comparable accuracy.

---

## 9. Failure modes and how the pipeline handles them

| Failure | Symptom | Mitigation in code |
|---|---|---|
| Initial pose too far off | fitness huge (`>10 000`), `iter=0` every scan | Check 1 rejects; user must click `/initialpose` in RViz |
| NDT converges to wrong basin | pose jumps >15 m from prediction | Check 2 rejects; keep previous pose |
| PCL throws an exception inside `align()` | thread dies | `try/catch` + reset `ndt_busy_` so next scan can still run |
| Slow subscriber (RViz) | publisher blocks → scan drops | `points_aligned` publisher is Best-Effort, depth 1 |
| Map not yet received | nothing to align against | `on_scan()` early-returns until `map_received_ = true` |
| Scan empty after filtering | PCL would crash on `setInputSource` | explicit empty-check before launching the thread |
| Previous NDT still running | would cause concurrent access to `ndt_` object | `ndt_busy_` atomic flag; new scan is simply skipped |
| Clock mismatch (use_sim_time without --clock) | TF timestamps unreadable by RViz | always play bag with `--clock` |

---

## Further reading

- **Original NDT paper:** Biber & Straßer, *The Normal Distributions Transform: A New Approach to Laser Scan Matching*, IROS 2003.
- **3D extension:** Magnusson, *The Three-Dimensional Normal-Distributions Transform — an Efficient Representation for Registration, Surface Analysis, and Loop Detection*, PhD thesis, 2009.
- **PCL implementation:** `pcl::NormalDistributionsTransform` (used by this project).
