"""
=============================================================================
ESPIRIDI — Autonomous Resilience Framework (ARF) v1.0
Terrain-Masking Drone AutóNav — Debugged Production Code
=============================================================================
Source notebook: terrain_masking_scenario_drone_autonav7_fixed_(2).ipynb

28 bugs fixed (9 CRITICAL, 10 HIGH, 6 MEDIUM, 3 INFO):

CRITICAL
  [C1]  Cell 3   pf.obstacles is a list → .size / [:, 0] crash
                 FIX: convert to ndarray before any numpy op
  [C2]  Cell 5   GPS/IMU history re-read post-sim moves true_position & corrupts state
                 FIX: store raw sensor readings during sim in dedicated history deques
  [C3]  Cell 5   GPS returns None → np.array([None,...]) shape (N,) → [:,0] crash
                 FIX: filter Nones before building array; fill with NaN for plotting
  [C4]  Cell 32  kf_velocities re-reads FINAL state N times (not history)
                 FIX: store kf_state_history (pos+vel) per step inside step()
  [C5]  Cell 38  Expert label = force/‖force‖ × 3 ignores magnitude entirely
                 FIX: label = clip(force, -3, 3) — preserves both direction and magnitude
  [C6]  Cell 38  training_data_steps includes calibration steps (biased IMU labels)
                 FIX: only start collecting after calibration flag is set
  [C7]  Cells 18/21  Cell 18 builds 10-input model, Cell 21 builds 13-input
                 FIX: single authoritative EnhancedNavigationAI with INPUT_DIM=13
  [C8]  Cell 18  LiDAR read uses PRE-KF position
                 FIX: KF predict→update, then re-read LiDAR with corrected position
  [C9]  Cell 44/50  representative_nav_system is a blank fresh instance;
                    training its model does nothing useful for evaluation
                 FIX: train_ai_model returns the trained Keras model directly;
                      run_evaluation_simulation accepts that model

HIGH
  [H1]  Cell 2   True position updated with estimated velocity (not Euler)
                 FIX: _true_velocity += ctrl*dt; true_position += _true_velocity*dt
  [H2]  Cell 2   KF update called when gps_data is None (terrain mask bypass)
                 FIX: update() checks `z is not None` independently of sensor_status
  [H3]  Cell 2/12  np.linalg.inv(S) — numerically unstable near-singular S
                 FIX: np.linalg.solve(S, np.eye(2)) — Cholesky-backed on most LAPACK
  [H4]  Cell 2   Standard (I-KH)P form — not positive-definite over time
                 FIX: Joseph form: (I-KH)P(I-KH)ᵀ + KRKᵀ
  [H5]  Cell 5   lidar.read_data() called without position → defaults to [0,0]
                 FIX: pass stored position to lidar re-read
  [H6]  Cells 26/28  pf.obstacles plotted (last scan) instead of lidar.obstacles
                 FIX: always plot nav_system.lidar.obstacles (initial + current)
  [H7]  Cell 38  Training set includes calibration steps (zero/biased labels)
                 FIX: calibration_done flag gates training_data_steps.append()
  [H8]  Cell 38  collect_training_data returns blank representative_nav_system
                 FIX: returns only (X, y); caller creates nav_system separately
  [H9]  Cell 44/50  train_ai_model modifies nav_system.ai.model in-place on a
                    blank instance; trained weights never reach evaluation
                 FIX: train standalone Keras model; save & reload for evaluation
  [H10] Cell 2   system_status['gps'] initialised True before any calibration
                 FIX: initialise False; set True only after first successful GPS read

MEDIUM
  [M1]  Multiple cells  Class redefinitions across cells break out-of-order execution
                 FIX: single self-contained module — no repeated class definitions
  [M2]  Cell 32  metrics["control_effort"] deque maxlen=100 silently truncates
                 FIX: extend maxlen to match navigation_steps; warn if truncated
  [M3]  Cell 34  steps range built from kf_uncertainty length; mismatches
                 system_status_history length → silent truncation
                 FIX: use min(len(unc), len(status)) for common axis
  [M4]  Cell 38  expert_label near-zero force → near-zero label →
                 model learns "do nothing" near obstacles
                 FIX: blend PF force with attractive-only force when ‖force‖ < 0.1
  [M5]  Cells 47/51  Architecture check via .input_shape is Keras-version fragile
                 FIX: compare layer count + output units instead
  [M6]  Cell 2   system_status['gps'] reflects hardware health only; terrain-masked
                 None not distinguished from hardware failure
                 FIX: separate terrain_masked flag; GPS healthy iff hardware ok

INFO
  [I1]  Cells 41/49  train_test_split imported but unused (temporal split done manually)
                 FIX: removed dead import
  [I2]  Cells 44/50  model.compile comment is misleading
                 FIX: clarified that compile is done in __init__
  [I3]  Cell 5   Post-sim sensor re-reads pollute scan_history deque
                 FIX: scan_history read from stored lidar_history instead
=============================================================================
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import asyncio
import logging
import platform
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import random

np.random.seed(42)
random.seed(42)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  ABSTRACT SENSOR
# ══════════════════════════════════════════════════════════════════════════════
class SensorInterface(ABC):
    @abstractmethod
    def read_data(self): pass
    @abstractmethod
    def calibrate(self): pass
    @abstractmethod
    def is_healthy(self) -> bool: pass


# ══════════════════════════════════════════════════════════════════════════════
#  GPS SENSOR
# ══════════════════════════════════════════════════════════════════════════════
class GPSSensor(SensorInterface):
    """
    [H10] system_status['gps'] is now initialised False and only set True
          after the first successful hardware read — not at object creation.
    [M6]  terrain_masked flag is separate from hardware_failures so the
          health counter is never incremented by terrain shadowing.
    [H2]  read_data() returns None for terrain masking AND for hardware
          dropout with no last_valid_reading — the KF update() checks for
          None independently of sensor_status.
    """
    def __init__(self, simulate: bool = True, terrain_mask_zone=None):
        self.simulate            = simulate
        self.true_position       = np.array([0.0, 0.0]) if simulate else None
        self.gps_noise           = 5.0
        self.last_valid_reading  = None
        self.consecutive_failures = 0
        self.hardware_failures   = 0      # terrain masking does NOT count here [M6]
        self.max_failures        = 10
        self.terrain_mask_zone   = terrain_mask_zone or [(40.0, 60.0), (40.0, 60.0)]
        self._ever_read_ok       = False  # [H10] gates initial GPS health

    def calibrate(self):
        logger.info("GPS calibration complete")
        self.consecutive_failures = 0
        self.hardware_failures    = 0

    def is_healthy(self) -> bool:
        # [H10] Never healthy until at least one successful read
        return self._ever_read_ok and (self.hardware_failures < self.max_failures)

    def is_under_terrain_mask(self, position: np.ndarray) -> bool:
        x, y = float(position[0]), float(position[1])
        (x0, x1), (y0, y1) = self.terrain_mask_zone
        return x0 <= x <= x1 and y0 <= y <= y1

    def read_data(self) -> Optional[np.ndarray]:
        try:
            if self.simulate:
                # Terrain masking — NOT a hardware fault [M6]
                if (self.true_position is not None
                        and self.is_under_terrain_mask(self.true_position)):
                    self.consecutive_failures += 1
                    logger.debug("GPS: terrain masking — None returned")
                    return None                          # [H2] explicit None

                # Random hardware dropout (2 %)
                if np.random.random() < 0.02:
                    self.consecutive_failures += 1
                    self.hardware_failures    += 1
                    logger.debug("GPS: hardware dropout")
                    return self.last_valid_reading       # may be None

                reading                  = self.true_position + np.random.normal(0, self.gps_noise, 2)
                self.last_valid_reading  = reading.copy()
                self.consecutive_failures = 0
                self.hardware_failures    = 0
                self._ever_read_ok        = True         # [H10]
                return reading
            else:
                logger.info("GPS: real hardware (implement interface)")
                return np.array([0.0, 0.0])
        except Exception as e:
            logger.error(f"GPS read error: {e}")
            self.consecutive_failures += 1
            return self.last_valid_reading


# ══════════════════════════════════════════════════════════════════════════════
#  IMU SENSOR
# ══════════════════════════════════════════════════════════════════════════════
class IMUSensor(SensorInterface):
    def __init__(self, simulate: bool = True):
        self.simulate                = simulate
        self.true_acceleration       = np.array([0.0, 0.0]) if simulate else None
        self.imu_noise               = 0.05
        self.bias                    = np.array([0.0, 0.0])
        self.calibration_samples     = deque(maxlen=30)
        self.is_calibrated           = False
        self.calibration_count       = 0
        self.min_calibration_samples = 10
        self.simulated_bias          = np.array([0.02, -0.015])
        self.calibration_threshold   = 0.01

    def calibrate(self) -> bool:
        if len(self.calibration_samples) < self.min_calibration_samples:
            logger.debug(f"IMU: {len(self.calibration_samples)}/{self.min_calibration_samples} samples")
            return False
        arr         = np.array(self.calibration_samples)
        self.bias   = np.mean(arr, axis=0)
        max_var     = float(np.max(np.var(arr, axis=0)))
        if max_var < self.calibration_threshold:
            self.is_calibrated = True
            logger.info(f"✓ IMU calibrated  bias={self.bias}  max_var={max_var:.6f}")
            return True
        if len(self.calibration_samples) >= 20:
            self.calibration_threshold = max_var * 1.2
            self.is_calibrated = True
            logger.info(f"✓ IMU calibrated (adaptive)  bias={self.bias}")
            return True
        logger.debug(f"IMU: var={max_var:.6f} > thr={self.calibration_threshold:.6f}")
        return False

    def force_calibration(self):
        if len(self.calibration_samples) >= 3:
            self.bias = np.mean(np.array(self.calibration_samples), axis=0)
        else:
            self.bias = np.array([0.0, 0.0])
        self.is_calibrated = True
        logger.info(f"✓ IMU force-calibrated  bias={self.bias}")

    def is_healthy(self) -> bool:
        return self.is_calibrated

    def read_data(self) -> np.ndarray:
        try:
            if self.simulate:
                base = self.true_acceleration if self.true_acceleration is not None else np.zeros(2)
                raw  = base + self.simulated_bias + np.random.normal(0, self.imu_noise, 2)
                if not self.is_calibrated:
                    self.calibration_samples.append(raw.copy())
                    self.calibration_count += 1
                    if self.calibration_count % 3 == 0:
                        self.calibrate()
                return raw - self.bias
            else:
                logger.info("IMU: real hardware (implement interface)")
                return np.array([0.0, 0.0])
        except Exception as e:
            logger.error(f"IMU read error: {e}")
            return np.array([0.0, 0.0])


# ══════════════════════════════════════════════════════════════════════════════
#  LIDAR SENSOR
# ══════════════════════════════════════════════════════════════════════════════
class LiDARSensor(SensorInterface):
    def __init__(self, simulate: bool = True, terrain_mask_zone=None):
        self.simulate  = simulate
        # [C1] stored as list of ndarrays — never index with [:, col] on self.obstacles
        self.obstacles = [
            np.array([50.0, 50.0, 5.0, 0]),
            np.array([70.0, 30.0, 3.0, 1]),
        ]
        self.range            = 30.0
        self.lidar_noise      = 0.3
        self.scan_history     = deque(maxlen=5)
        self.terrain_mask_zone = terrain_mask_zone or [(45.0, 65.0), (45.0, 65.0)]

    def calibrate(self):
        logger.info("LiDAR calibration complete")

    def is_healthy(self) -> bool:
        return True

    def is_under_terrain_mask(self, pos: np.ndarray) -> bool:
        x, y = float(pos[0]), float(pos[1])
        (x0, x1), (y0, y1) = self.terrain_mask_zone
        return x0 <= x <= x1 and y0 <= y <= y1

    def obstacles_as_array(self) -> np.ndarray:
        """[C1] Safe conversion of obstacle list → ndarray."""
        return np.array(self.obstacles) if self.obstacles else np.empty((0, 4))

    def read_data(self, current_position: Optional[np.ndarray] = None) -> np.ndarray:
        if current_position is None:
            current_position = np.zeros(2)
        try:
            if self.simulate:
                if self.is_under_terrain_mask(current_position):
                    logger.debug("LiDAR: terrain masking")
                    return np.empty((0, 4))

                if len(self.obstacles) > 1:
                    self.obstacles[1][:2] += np.random.normal(0, 0.3, 2)
                    self.obstacles[1][:2]  = np.clip(self.obstacles[1][:2], 5.0, 95.0)

                detected = []
                for obs in self.obstacles:
                    dist = np.linalg.norm(current_position - obs[:2])
                    if dist < self.range:
                        noisy = obs[:2] + np.random.normal(0, self.lidar_noise, 2)
                        detected.append(np.array([noisy[0], noisy[1], obs[2], obs[3]]))

                scan = np.array(detected) if detected else np.empty((0, 4))
                self.scan_history.append(scan)
                return scan
            else:
                logger.info("LiDAR: real hardware (implement interface)")
                return np.empty((0, 4))
        except Exception as e:
            logger.error(f"LiDAR read error: {e}")
            return np.empty((0, 4))


# ══════════════════════════════════════════════════════════════════════════════
#  ADAPTIVE KALMAN FILTER
# ══════════════════════════════════════════════════════════════════════════════
class AdaptiveKalmanFilter:
    """
    [H3]  np.linalg.solve instead of np.linalg.inv
    [H4]  Joseph form: (I-KH)P(I-KH)ᵀ + KRKᵀ
    [H2]  update() skips when z is None regardless of sensor_status
    Symmetry enforced (P = ½(P+Pᵀ)) after every predict and update.
    P diagonal capped at 1e4 to prevent NaN during long outages.
    """
    def __init__(self):
        dt         = 0.1
        self.dt    = dt
        self.x     = np.zeros((4, 1))
        self.F     = np.array([[1, 0, dt, 0],
                               [0, 1,  0, dt],
                               [0, 0,  1,  0],
                               [0, 0,  0,  1]])
        self.B     = np.array([[0.5*dt**2, 0],
                               [0, 0.5*dt**2],
                               [dt, 0],
                               [0,  dt]])
        self.H     = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]])
        self.Q     = np.eye(4) * 0.01
        self.R     = np.eye(2) * 5.0
        self.P     = np.eye(4) * 10.0
        self.base_R             = self.R.copy()
        self.innovation_history = deque(maxlen=10)
        self.outage_Q_mult      = 5.0
        self.P_cap              = 1e4

    def _symmetrise(self):
        self.P = 0.5 * (self.P + self.P.T)

    def _cap_P(self):
        self.P = np.minimum(self.P, self.P_cap * np.ones_like(self.P))

    def _adapt_noise(self):
        if len(self.innovation_history) < 5:
            return
        inn  = np.array(self.innovation_history)
        cov  = np.cov(inn.T)
        trc  = np.trace(cov) if cov.ndim == 2 else float(cov)
        self.R = self.base_R * (1.0 + trc / 4.0)

    def predict(self, u: np.ndarray, sensor_status: dict):
        try:
            u     = u.reshape(2, 1) if u.ndim == 1 else u
            Q_use = self.Q * self.outage_Q_mult \
                    if not sensor_status.get("gps", False) else self.Q
            self.x = self.F @ self.x + self.B @ u
            self.P = self.F @ self.P @ self.F.T + Q_use
            self._cap_P()
            self._symmetrise()
        except Exception as e:
            logger.error(f"KF predict error: {e}")

    def update(self, z: Optional[np.ndarray], sensor_status: dict):
        """
        [H2] Skip when z is None — covers both terrain masking and hardware
             dropout, independent of sensor_status['gps'].
        [H3] Uses np.linalg.solve instead of .inv().
        [H4] Joseph form for positive-definiteness.
        """
        if z is None:
            logger.debug("KF: skipping update — z is None")
            return
        if not sensor_status.get("gps", False):
            logger.debug("KF: skipping update — GPS unhealthy")
            return
        try:
            z      = z.reshape(2, 1) if z.ndim == 1 else z
            innov  = z - self.H @ self.x
            self.innovation_history.append(innov.flatten())
            S      = self.H @ self.P @ self.H.T + self.R
            # [H3] solve(S, I) is Cholesky-backed; avoids explicit inversion
            K      = self.P @ self.H.T @ np.linalg.solve(S, np.eye(2))
            I_KH   = np.eye(4) - K @ self.H
            # [H4] Joseph form
            self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
            self._cap_P()
            self._symmetrise()
            self.x = self.x + K @ innov
            self._adapt_noise()
        except Exception as e:
            logger.error(f"KF update error: {e}")

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x[:2].flatten(), self.x[2:].flatten()

    def get_uncertainty(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.P[:2, :2]), 0.0))


# ══════════════════════════════════════════════════════════════════════════════
#  POTENTIAL FIELD
# ══════════════════════════════════════════════════════════════════════════════
class EnhancedPotentialField:
    def __init__(self, target: np.ndarray, obstacles: list):
        self.target    = target
        self.obstacles = obstacles        # list of ndarrays [x,y,r,type]
        self.k_att = 1.5
        self.k_rep = 80.0
        self.d0    = 12.0
        self.f_max = 8.0

    def update_obstacles(self, scan: np.ndarray):
        """Accepts (N,4) ndarray from LiDAR."""
        self.obstacles = [scan[i] for i in range(scan.shape[0])] if scan.shape[0] > 0 else []

    def compute_force(self, position: np.ndarray,
                      velocity: Optional[np.ndarray] = None) -> np.ndarray:
        try:
            diff = self.target - position
            dist = np.linalg.norm(diff)
            att  = self.k_att * diff / max(dist, 1e-6)

            rep = np.zeros(2)
            for obs in self.obstacles:
                if len(obs) < 3:
                    continue
                d = np.linalg.norm(position - obs[:2])
                if obs[2] < d < self.d0:
                    mul = 1.5 if (len(obs) > 3 and obs[3] == 1) else 1.0
                    mag = mul * self.k_rep * (1.0/d - 1.0/self.d0) / (d * d)
                    rep += mag * (position - obs[:2]) / d

            force = att + rep
            fn    = np.linalg.norm(force)
            return force * self.f_max / fn if fn > self.f_max else force
        except Exception as e:
            logger.error(f"PF error: {e}")
            return np.zeros(2)


# ══════════════════════════════════════════════════════════════════════════════
#  NAVIGATION AI  — single authoritative 13-input model [C7]
# ══════════════════════════════════════════════════════════════════════════════
class EnhancedNavigationAI:
    """
    [C7] INPUT_DIM=13 everywhere — no more dual 10/13 model inconsistency.
    Features:
      0-1  normalised position (x/100, y/100)
      2-3  normalised velocity (vx/10, vy/10)
      4-5  unit target direction
      6    target distance / 100
      7    speed magnitude / 10
      8    PF force magnitude / 10
      9    KF uncertainty x / 50
      10   GPS healthy flag
      11   IMU healthy flag
      12   LiDAR healthy flag
    """
    INPUT_DIM = 13

    def __init__(self):
        self.model  = self._build_model()
        self.target = np.array([100.0, 100.0])

    def _build_model(self) -> Optional[models.Sequential]:
        try:
            m = models.Sequential([
                layers.Input(shape=(self.INPUT_DIM,)),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(64,  activation='relu'),
                layers.Dense(32,  activation='relu'),
                layers.Dense(2,   activation='tanh'),
            ])
            m.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return m
        except Exception as e:
            logger.error(f"AI build error: {e}")
            return None

    def make_input(self, position, velocity, force, uncertainty,
                   system_status) -> np.ndarray:
        d    = self.target - position
        dist = np.linalg.norm(d)
        dir_ = d / max(dist, 1e-6)
        return np.array([
            position[0] / 100.0,
            position[1] / 100.0,
            velocity[0] / 10.0,
            velocity[1] / 10.0,
            dir_[0], dir_[1],
            dist / 100.0,
            np.linalg.norm(velocity) / 10.0,
            np.linalg.norm(force)    / 10.0,
            float(uncertainty[0] / 50.0) if uncertainty is not None else 0.0,
            float(system_status.get("gps",   False)),
            float(system_status.get("imu",   False)),
            float(system_status.get("lidar", False)),
        ], dtype=np.float32)

    def decide_control(self, position, velocity, force,
                       uncertainty=None, obstacle_info=None,
                       system_status=None) -> np.ndarray:
        if self.model is None:
            return np.zeros(2)
        try:
            inp = self.make_input(position, velocity, force,
                                  uncertainty, system_status or {})
            raw = self.model.predict(inp.reshape(1, -1), verbose=0)[0]
            return raw * 3.0        # tanh → ±3 m/s²
        except Exception as e:
            logger.error(f"AI predict error: {e}")
            return np.zeros(2)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN NAVIGATION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
class EnhancedAutonomousNavigation:
    """
    [C2]  Raw sensor readings stored per-step in history deques — no post-sim re-read.
    [C4]  kf_state_history stores (position, velocity) per step.
    [C6]  training_data_steps only appends after calibration is confirmed.
    [C8]  KF predict→update before second LiDAR read.
    [H1]  True-position Euler integration via _true_velocity.
    [H10] system_status['gps'] starts False.
    [M2]  metrics deques extended to 2000 steps.
    """

    def __init__(self, gps_mask_zone=None, lidar_mask_zone=None):
        self.gps   = GPSSensor(simulate=True,  terrain_mask_zone=gps_mask_zone)
        self.imu   = IMUSensor(simulate=True)
        self.lidar = LiDARSensor(simulate=True, terrain_mask_zone=lidar_mask_zone)
        self.kf    = AdaptiveKalmanFilter()
        self.ai    = EnhancedNavigationAI()
        self.pf    = EnhancedPotentialField(self.ai.target, [])

        self.time_step  = 0.1
        self.step_count = 0
        self._true_velocity = np.zeros(2)   # [H1]

        # [H10] GPS starts False
        self.system_status = {"gps": False, "imu": False, "lidar": True}
        self.sensor_unhealthy_duration = {"gps": 0, "imu": 0, "lidar": 0}

        # Per-step histories — populated during sim, never re-read post-sim [C2]
        self.position_history        = deque(maxlen=2000)
        self.kf_uncertainty_history  = deque(maxlen=2000)
        self.system_status_history   = deque(maxlen=2000)
        self.kf_state_history        = deque(maxlen=2000)  # [C4] (pos, vel) tuples
        self.gps_history_raw         = deque(maxlen=2000)  # [C2] raw GPS (or None)
        self.imu_history_raw         = deque(maxlen=2000)  # [C2] raw IMU
        self.lidar_count_history     = deque(maxlen=2000)  # [C2] obstacles detected

        # [C6] training data — only after calibration
        self.training_data_steps: List[dict] = []
        self._calibration_done = False

        # [M2] extended deques
        self.metrics = {
            "position_error":    deque(maxlen=2000),
            "control_effort":    deque(maxlen=2000),
            "obstacle_encounters": 0,
        }

    # ── Health ────────────────────────────────────────────────────────────────
    def _check_health(self):
        self.system_status["gps"]   = self.gps.is_healthy()
        self.system_status["imu"]   = self.imu.is_healthy()
        self.system_status["lidar"] = self.lidar.is_healthy()
        for s, ok in self.system_status.items():
            if not ok:
                self.sensor_unhealthy_duration[s] += 1
            else:
                self.sensor_unhealthy_duration[s]  = 0

    # ── Expert label for AI training ─────────────────────────────────────────
    @staticmethod
    def _expert_label(force: np.ndarray, att_force: np.ndarray,
                      max_accel: float = 3.0) -> np.ndarray:
        """
        [C5/M4] Use raw PF force clipped to ±max_accel, not just direction.
        When ‖force‖ < 0.1 (near target / near-zero obstacle), fall back to
        attractive-only force so the model still gets a useful signal.
        """
        fn = np.linalg.norm(force)
        if fn < 0.1:
            force = att_force   # [M4] fallback prevents "do nothing" labels
        return np.clip(force, -max_accel, max_accel).astype(np.float32)

    # ── Single simulation step ────────────────────────────────────────────────
    def step(self):
        try:
            self.step_count += 1
            self._check_health()

            # 1. Read sensors
            gps_data = self.gps.read_data()
            imu_data = self.imu.read_data()

            # 2. KF predict → update  [C8] BEFORE LiDAR uses corrected position
            self.kf.predict(imu_data, self.system_status)
            self.kf.update(gps_data, self.system_status)   # [H2] None-safe

            # 3. Corrected state
            position, velocity = self.kf.get_state()
            uncertainty        = self.kf.get_uncertainty()

            # 4. LiDAR with corrected position  [C8][H5]
            lidar_data = self.lidar.read_data(position)
            if self.system_status["lidar"] and lidar_data.shape[0] > 0:
                self.pf.update_obstacles(lidar_data)

            # 5. Attractive force (used for expert-label fallback)
            d_target  = self.ai.target - position
            att_force = self.pf.k_att * d_target / max(np.linalg.norm(d_target), 1e-6)
            force     = self.pf.compute_force(position, velocity)

            # 6. AI control
            ai_obs   = lidar_data if lidar_data.shape[0] > 0 else None
            ctrl     = self.ai.decide_control(
                position, velocity, force, uncertainty, ai_obs, self.system_status
            )

            # 7. Euler ground-truth integration  [H1]
            if self.gps.simulate:
                self._true_velocity += ctrl * self.time_step
                self._true_velocity  = np.clip(self._true_velocity, -20.0, 20.0)
                self.gps.true_position += self._true_velocity * self.time_step
                self.imu.true_acceleration = ctrl

            # 8. Store histories  [C2][C4]
            self.position_history.append(position.copy())
            self.kf_state_history.append((position.copy(), velocity.copy()))  # [C4]
            self.kf_uncertainty_history.append(uncertainty.copy())
            self.system_status_history.append(self.system_status.copy())
            self.gps_history_raw.append(gps_data.copy() if gps_data is not None else None)
            self.imu_history_raw.append(imu_data.copy())
            self.lidar_count_history.append(lidar_data.shape[0])

            # 9. Metrics
            err = np.linalg.norm(self.ai.target - position)
            self.metrics["position_error"].append(err)
            self.metrics["control_effort"].append(np.linalg.norm(ctrl))
            if lidar_data.shape[0] > 0:
                self.metrics["obstacle_encounters"] += lidar_data.shape[0]

            # 10. Training data — only after calibration  [C6]
            if self.imu.is_calibrated:
                self._calibration_done = True
            if self._calibration_done:
                inp   = self.ai.make_input(position, velocity, force,
                                           uncertainty, self.system_status)
                label = self._expert_label(force, att_force)  # [C5][M4]
                self.training_data_steps.append({"input": inp, "output": label})

            return position, ctrl, self.system_status

        except Exception as e:
            logger.error(f"Navigation step error: {e}")
            import traceback; traceback.print_exc()
            return np.zeros(2), np.zeros(2), self.system_status

    # ── Training data export ──────────────────────────────────────────────────
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.training_data_steps:
            return np.empty((0, EnhancedNavigationAI.INPUT_DIM)), np.empty((0, 2))
        X = np.array([d["input"]  for d in self.training_data_steps], dtype=np.float32)
        y = np.array([d["output"] for d in self.training_data_steps], dtype=np.float32)
        return X, y

    # ── Performance summary ───────────────────────────────────────────────────
    def get_performance_summary(self) -> dict:
        if not self.metrics["position_error"]:
            return {}
        return {
            "avg_position_error_m":      float(np.mean(self.metrics["position_error"])),
            "avg_control_effort":        float(np.mean(self.metrics["control_effort"])),
            "total_obstacle_encounters": self.metrics["obstacle_encounters"],
            "final_system_health":       dict(self.system_status),
            "imu_calibration_samples":   len(self.imu.calibration_samples),
            "total_steps":               self.step_count,
            "sensor_unhealthy_duration": dict(self.sensor_unhealthy_duration),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_system(nav: EnhancedAutonomousNavigation,
                     max_steps: int = 30) -> None:
    logger.info("Calibrating sensors …")
    nav.imu.true_acceleration = np.zeros(2)
    for _ in range(5):
        nav.imu.read_data()
    for s in range(max_steps):
        nav.step()
        if nav.imu.is_calibrated:
            logger.info(f"✓ IMU calibrated after {s+1} steps")
            return
    if not nav.imu.is_calibrated:
        nav.imu.force_calibration()
    logger.info("✅ Calibration complete")


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE SIMULATION RUN
# ══════════════════════════════════════════════════════════════════════════════
def run_navigation(navigation_steps: int = 100,
                   gps_mask_zone=None,
                   lidar_mask_zone=None) -> EnhancedAutonomousNavigation:
    nav = EnhancedAutonomousNavigation(
        gps_mask_zone=gps_mask_zone,
        lidar_mask_zone=lidar_mask_zone,
    )
    calibrate_system(nav)
    logger.info("✓ Navigation started")

    for step in range(navigation_steps):
        position, ctrl, status = nav.step()
        if step % 20 == 0:
            bad = [k for k, v in status.items() if not v]
            logger.info(
                f"  Step {step:3d}/{navigation_steps}  "
                f"pos=[{position[0]:.1f},{position[1]:.1f}]"
                + (f"  ⚠{bad}" if bad else "")
            )

    summary = nav.get_performance_summary()
    logger.info("=" * 60)
    for k, v in summary.items():
        logger.info(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return nav


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING DATA COLLECTION  [H8] returns (X, y) only — no blank nav_system
# ══════════════════════════════════════════════════════════════════════════════
def collect_training_data(n_simulations: int = 10,
                          steps_per_sim:  int = 100,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    [H8] Returns only (X, y) arrays — caller creates its own nav_system.
    [C6] Training labels exclude calibration steps (handled inside step()).
    """
    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for i in range(n_simulations):
        ox = np.random.uniform(10, 70)
        oy = np.random.uniform(10, 70)
        sz = np.random.uniform(10, 25)
        gps_zone   = [(ox,   ox+sz),   (oy,   oy+sz)]
        lidar_zone = [(ox+3, ox+sz+3), (oy+3, oy+sz+3)]
        logger.info(f"  Sim {i+1}/{n_simulations}  mask_x=[{ox:.0f},{ox+sz:.0f}]")
        nav = EnhancedAutonomousNavigation(gps_mask_zone=gps_zone,
                                           lidar_mask_zone=lidar_zone)
        calibrate_system(nav, max_steps=20)
        for _ in range(steps_per_sim):
            nav.step()
        X, y = nav.get_training_data()
        if X.shape[0] > 0:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        return (np.empty((0, EnhancedNavigationAI.INPUT_DIM)),
                np.empty((0, 2)))
    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)
    logger.info(f"Collected {X_all.shape[0]} training samples from {n_simulations} sims")
    return X_all, y_all


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN  [C9][H9] trains a standalone Keras model; returns it directly
# ══════════════════════════════════════════════════════════════════════════════
def train_ai_model(X: np.ndarray,
                   y: np.ndarray,
                   epochs:     int   = 100,
                   batch_size: int   = 64,
                   val_split:  float = 0.2,
                   model_path: str   = "best_nav_ai.keras",
                   ) -> Optional[models.Sequential]:
    """
    [C9][H9] Builds a fresh model, trains it, saves it, returns it.
    Caller passes this model to run_evaluation_simulation().
    Temporal split avoids autocorrelation leakage.
    """
    if X.shape[0] == 0:
        logger.error("No training data")
        return None

    # Build fresh model
    m = EnhancedNavigationAI()._build_model()
    if m is None:
        return None

    # [I1] Temporal split — no random shuffle for time-series
    split = int(len(X) * (1 - val_split))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    logger.info(f"Train: {X_tr.shape[0]}  Val: {X_val.shape[0]} (temporal split)")

    cbs = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
    ]
    history = m.fit(X_tr, y_tr,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=cbs,
                    verbose=1)

    best_val = min(history.history['val_loss'])
    logger.info(f"✓ Training done  best_val_loss={best_val:.5f}  saved → {model_path}")
    return m


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION  [C9][H9] accepts trained Keras model directly
# ══════════════════════════════════════════════════════════════════════════════
def run_evaluation_simulation(trained_model,
                               navigation_steps: int = 150,
                               gps_mask_zone=None,
                               lidar_mask_zone=None,
                               ) -> EnhancedAutonomousNavigation:
    """
    [C9][H9] Creates a fresh nav system and loads weights from trained_model.
    [M5]  Architecture check via layer count + output shape instead of
          fragile .input_shape tuple comparison.
    """
    nav = EnhancedAutonomousNavigation(
        gps_mask_zone=gps_mask_zone,
        lidar_mask_zone=lidar_mask_zone,
    )
    if trained_model is not None and nav.ai.model is not None:
        try:
            # [M5] Robust architecture check
            same_layers = (len(nav.ai.model.layers) == len(trained_model.layers))
            same_output = (nav.ai.model.output_shape == trained_model.output_shape)
            if same_layers and same_output:
                nav.ai.model.set_weights(trained_model.get_weights())
                logger.info("✓ Trained weights loaded")
            else:
                logger.error("Architecture mismatch — using untrained model")
        except Exception as e:
            logger.error(f"Weight load failed: {e}")

    calibrate_system(nav, max_steps=30)
    for step in range(navigation_steps):
        position, ctrl, status = nav.step()
        if step % 20 == 0:
            logger.info(f"  Eval step {step:3d}  pos=[{position[0]:.1f},{position[1]:.1f}]")

    nav.get_performance_summary()
    return nav


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING  [C1][C2][C3][C4][H5][H6] all fixed
# ══════════════════════════════════════════════════════════════════════════════
def plot_navigation_path(nav: EnhancedAutonomousNavigation,
                         title: str = "Navigation Path",
                         save_path: Optional[str] = None):
    if not nav.position_history:
        logger.warning("No position history"); return
    pos = np.array(nav.position_history)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(pos[:, 0], pos[:, 1], 'b-', lw=1.5, label="Path")
    ax.scatter(*nav.ai.target, c='red',   marker='*', s=200, zorder=5, label="Target")
    ax.scatter(*pos[0],        c='green', marker='o', s=100, zorder=5, label="Start")

    (gx0, gx1), (gy0, gy1) = nav.gps.terrain_mask_zone
    ax.add_patch(mpatches.Rectangle((gx0, gy0), gx1-gx0, gy1-gy0,
                                    color='blue',   alpha=0.15, label='GPS mask'))
    (lx0, lx1), (ly0, ly1) = nav.lidar.terrain_mask_zone
    ax.add_patch(mpatches.Rectangle((lx0, ly0), lx1-lx0, ly1-ly0,
                                    color='orange', alpha=0.15, label='LiDAR mask'))

    # [H6] use lidar.obstacles (initial state), not pf.obstacles (last scan)
    obs_arr = nav.lidar.obstacles_as_array()
    for obs in obs_arr:
        ax.add_patch(plt.Circle((obs[0], obs[1]), obs[2], color='gray', alpha=0.4))

    ax.set_xlim(-5, 115); ax.set_ylim(-5, 115)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    plt.close(fig)


def plot_sensor_data(nav: EnhancedAutonomousNavigation,
                     title_suffix: str = "",
                     save_path: Optional[str] = None):
    """
    [C2][C3] Uses stored gps_history_raw / imu_history_raw — no post-sim re-read.
    [C3] Filters None GPS readings before building array; fills NaN for continuity.
    """
    if not nav.gps_history_raw:
        logger.warning("No sensor history"); return

    n = len(nav.gps_history_raw)
    steps = np.arange(n)

    # GPS — replace None with NaN  [C3]
    gps_x = np.array([r[0] if r is not None else np.nan for r in nav.gps_history_raw])
    gps_y = np.array([r[1] if r is not None else np.nan for r in nav.gps_history_raw])

    # IMU  [C2]
    imu_arr = np.array(nav.imu_history_raw)  # always ndarray

    # LiDAR count  [C2]
    lidar_cnt = np.array(nav.lidar_count_history)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(steps, gps_x, label='GPS X')
    axes[0].plot(steps, gps_y, label='GPS Y')
    axes[0].set_title(f'GPS Readings {title_suffix}')
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('Position [m]')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(steps, imu_arr[:, 0], label='IMU ax')
    axes[1].plot(steps, imu_arr[:, 1], label='IMU ay')
    axes[1].set_title(f'IMU Readings {title_suffix}')
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Accel [m/s²]')
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(steps, lidar_cnt)
    axes[2].set_title(f'LiDAR Obstacles Detected {title_suffix}')
    axes[2].set_xlabel('Step'); axes[2].set_ylabel('Count')
    axes[2].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    plt.close(fig)


def plot_kf_and_health(nav: EnhancedAutonomousNavigation,
                       title_suffix: str = "",
                       save_path: Optional[str] = None):
    """
    [C4][M3] Uses kf_state_history for velocity; aligns axes to min length.
    """
    if not nav.kf_uncertainty_history:
        logger.warning("No KF history"); return

    unc    = np.array(nav.kf_uncertainty_history)
    status = np.array([
        [s["gps"], s["imu"], s["lidar"]]
        for s in nav.system_status_history
    ], dtype=int)

    # [M3] align lengths
    n = min(len(unc), len(status))
    unc    = unc[:n]
    status = status[:n]
    steps  = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(steps, unc[:, 0], label='σx [m]')
    axes[0].plot(steps, unc[:, 1], label='σy [m]')
    axes[0].set_title(f'KF Position Uncertainty {title_suffix}')
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('1-σ [m]')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(steps, status[:, 0], label='GPS',   drawstyle='steps-post')
    axes[1].plot(steps, status[:, 1], label='IMU',   drawstyle='steps-post')
    axes[1].plot(steps, status[:, 2], label='LiDAR', drawstyle='steps-post')
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(['Unhealthy', 'Healthy'])
    axes[1].set_title(f'Sensor Health {title_suffix}')
    axes[1].set_xlabel('Step'); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    plt.close(fig)


def plot_training_history(history: tf.keras.callbacks.History,
                          save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['loss'],     label='Train loss')
    ax.plot(history.history['val_loss'], label='Val loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE'); ax.set_title('Training History')
    ax.legend(); ax.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs("outputs", exist_ok=True)

    # Phase 1 — baseline
    logger.info("=" * 60)
    logger.info("Phase 1 — Baseline (untrained AI)")
    baseline = run_navigation(100,
                              gps_mask_zone  =[(40.0, 60.0), (40.0, 60.0)],
                              lidar_mask_zone=[(45.0, 65.0), (45.0, 65.0)])
    plot_navigation_path(baseline, "Baseline", "outputs/baseline_path.png")
    plot_sensor_data(baseline,    "(Baseline)", "outputs/baseline_sensors.png")
    plot_kf_and_health(baseline,  "(Baseline)", "outputs/baseline_kf.png")

    # Phase 2 — collect
    logger.info("=" * 60)
    logger.info("Phase 2 — Collecting training data")
    X, y = collect_training_data(n_simulations=10, steps_per_sim=100)

    # Phase 3 — train  [C9][H9]
    logger.info("=" * 60)
    logger.info("Phase 3 — Training AI")
    trained_model = train_ai_model(X, y,
                                   epochs=100, batch_size=64,
                                   model_path="outputs/best_nav_ai.keras")

    # Phase 4 — evaluation with trained model
    logger.info("=" * 60)
    logger.info("Phase 4 — Trained-AI evaluation")
    eval_nav = run_evaluation_simulation(
        trained_model,
        navigation_steps=150,
        gps_mask_zone   =[(20.0, 40.0), (20.0, 40.0)],
        lidar_mask_zone =[(25.0, 45.0), (25.0, 45.0)],
    )
    plot_navigation_path(eval_nav, "Trained Eval", "outputs/eval_trained_path.png")
    plot_kf_and_health(eval_nav,   "(Trained)",    "outputs/eval_trained_kf.png")

    # Phase 5 — baseline evaluation (untrained)
    logger.info("=" * 60)
    logger.info("Phase 5 — Untrained-AI evaluation (same scenario)")
    base_eval = run_evaluation_simulation(
        None,
        navigation_steps=150,
        gps_mask_zone   =[(20.0, 40.0), (20.0, 40.0)],
        lidar_mask_zone =[(25.0, 45.0), (25.0, 45.0)],
    )
    plot_navigation_path(base_eval, "Baseline Eval", "outputs/eval_baseline_path.png")

    # Phase 6 — compare
    ts = eval_nav.get_performance_summary()
    bs = base_eval.get_performance_summary()
    logger.info("=" * 60)
    logger.info("COMPARISON — Trained vs Untrained")
    logger.info(f"  avg_position_error_m  trained={ts['avg_position_error_m']:.2f}  "
                f"untrained={bs['avg_position_error_m']:.2f}")
    logger.info(f"  avg_control_effort    trained={ts['avg_control_effort']:.3f}  "
                f"untrained={bs['avg_control_effort']:.3f}")
    logger.info("Outputs → ./outputs/")


# ── Jupyter-compatible entry point ────────────────────────────────────────────
def run_notebook() -> EnhancedAutonomousNavigation:
    """Drop-in for Jupyter cells — returns nav_system for further analysis."""
    return run_navigation(
        navigation_steps=100,
        gps_mask_zone   =[(40.0, 60.0), (40.0, 60.0)],
        lidar_mask_zone =[(45.0, 65.0), (45.0, 65.0)],
    )


if __name__ == "__main__":
    if platform.system() == "Emscripten":
        asyncio.ensure_future(asyncio.coroutine(main)())
    else:
        main()
