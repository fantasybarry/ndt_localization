# NDT Node Design

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NDT Localizer                               │
│                     (P2DNDTLocalizer)                                │
│                                                                     │
│  Inputs: scan, map, initial_estimate, optimizer                     │
│  Output: pose_with_covariance                                       │
│                                                                     │
│  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────┐  │
│  │  Scan         │   │  Map             │   │  Optimization      │  │
│  │ (P2DNDTScan)  │   │                  │   │  Problem           │  │
│  │               │   │  ┌────────────┐  │   │  (P2D NDT)         │  │
│  │ Wraps         │   │  │DynamicNDT  │  │   │                    │  │
│  │ Eigen::Vector │   │  │  Map       │  │   │ Uses               │  │
│  │ 3d vector     │   │  │(Welford's  │  │   │ CachedExpression   │  │
│  │               │   │  │ algorithm) │  │   │ to compute:        │  │
│  │ Provides      │   │  ├────────────┤  │   │  - Score           │  │
│  │ iterator      │   │  │StaticNDT   │  │   │  - Jacobian        │  │
│  │ access for    │   │  │  Map       │  │   │  - Hessian         │  │
│  │ optimization  │   │  │(precomputed│  │   │                    │  │
│  │               │   │  │ voxels)    │  │   │ Based on Magnusson │  │
│  └──────┬───────┘   │  └─────┬──────┘  │   │ 2009 (P2D)        │  │
│         │           └────────┼─────────┘   └─────────┬──────────┘  │
│         │                    │                        │             │
│         └────────────┬───────┘────────────────────────┘             │
│                      ▼                                              │
│         ┌────────────────────────┐                                  │
│         │   NDTLocalizerBase     │                                  │
│         │  (template interface)  │                                  │
│         │                        │                                  │
│         │  register_scan()       │                                  │
│         │  register_map()        │                                  │
│         │  localize()            │                                  │
│         └───────────┬────────────┘                                  │
│                     ▼                                               │
│         ┌────────────────────────┐                                  │
│         │  Pose with Covariance  │                                  │
│         └────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Point Cloud (PCD)              LiDAR Scan
       │                            │
       ▼                            ▼
┌──────────────┐            ┌──────────────┐
│  NDT Map     │            │  P2DNDTScan  │
│              │            │              │
│ DynamicNDTMap│            │ Eigen::Vec3d │
│  or          │            │   vector     │
│ StaticNDTMap │            │              │
│              │            │              │
│ (voxel grid  │            │              │
│  centroids + │            │              │
│  covariance) │            │              │
└──────┬───────┘            └──────┬───────┘
       │                           │
       │    ┌──────────────────┐   │
       └───►│ P2D Optimization │◄──┘
            │    Problem       │
            │                  │
            │ + Initial Pose   │◄── Initial Estimate
            │   Estimate       │
            │                  │
            │ Computes:        │
            │  • Score         │
            │  • Jacobian      │
            │  • Hessian       │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │  Optimizer       │
            │ (Newton's method │
            │  or similar)     │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Output:          │
            │ Pose with        │
            │ Covariance       │
            │ (map → base_link)│
            └──────────────────┘
```

## Key Components

| Component | Description |
|---|---|
| **DynamicNDTMap** | Converts point clouds to NDT representation on-the-fly using Welford's algorithm. Uses `std::unordered_map` for fast lookups. |
| **StaticNDTMap** | Loads pre-computed voxel data (centroids + covariance) without recalculation. |
| **P2DNDTScan** | Wraps `Eigen::Vector3d` vector, provides iterator access for optimization. |
| **P2D Optimization** | Computes score, Jacobian, and Hessian simultaneously via `CachedExpression` for efficiency. Based on Magnusson 2009. |
| **NDTLocalizerBase** | Template-based interface supporting various NDT method variants. |
| **P2DNDTLocalizer** | Concrete Point-to-Distribution NDT localizer implementation. |

## Initialization

The NDT localizer requires an initial guess of the vehicle pose close to the truth before it can optimize. Two methods are available:

1. **RViz**: Click "2D pose estimation" and drag an arrow to match vehicle orientation. The pose is published to `/localization/initialpose`.
2. **Terminal**: Manually publish a `PoseWithCovarianceStamped` message via `ros2 topic pub`.

## NDT Map Provider Node

### Required Files

- `.pcd` file: 3D point cloud data of the map
- `.yaml` file: Map origin in geocentric (WGS84) coordinates

### YAML Format

```yaml
map_config:
  latitude: 37.380811523812845
  longitude: -121.90840595108715
  elevation: 16.0
  roll: 0.0
  pitch: 0.0
  yaw: 0.0
```

### Published Topics

| Topic | Description |
|---|---|
| `ndt_map` | NDT map data |
| `viz_ndt_map` | Point cloud visualization |
| `viz_map_subsampled` | Subsampled point cloud (when voxel grid used) |

### Parameters

| Parameter | Purpose |
|---|---|
| `map_pcd_file` | Point cloud data filename |
| `map_yaml_file` | Map information filename |
| `map_frame` | Coordinate frame identifier |
| `map_config.capacity` | Max voxel capacity |
| `map_config.min_point.{x,y,z}` | Minimum bounds |
| `map_config.max_point.{x,y,z}` | Maximum bounds |
| `map_config.voxel_size.{x,y,z}` | Voxel dimensions |
| `viz_map` | Visualization publishing flag |

## References

- [NDT Design (GitLab mirror)](https://autowarefoundation.gitlab.io/autoware.auto/AutowareAuto/ndt.html)
- [NDT Map Provider Node](https://autowarefoundation.gitlab.io/autoware.auto/AutowareAuto/ndt-map-provider-node.html)
- [NDT Initialization](https://autowarefoundation.gitlab.io/autoware.auto/AutowareAuto/ndt-initialization.html)
- [Original paper - Magnusson 2009](http://www.diva-portal.org/smash/get/diva2:276162/FULLTEXT02.pdf)
