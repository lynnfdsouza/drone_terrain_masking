import numpy as np
import matplotlib.pyplot as plt

class ESPIRIDI_ARF_TRL7_Integrated:
    def __init__(self):
        # 1. HARDWARE DYNAMICS (The "Plant")
        self.dt = 0.05       # 20Hz Flight Controller Loop
        self.mass = 2.5      # kg
        self.drag = 0.22     # N/(m/s)
        self.max_thrust = 60.0 # Newtons (Hardware Limit)
        
        # 2. STATE VECTORS
        self.true_pos = np.array([0.0, 0.0, 5.0])
        self.true_vel = np.array([0.0, 0.0, 0.0])
        self.slam_pos = np.array([0.0, 0.0, 5.0])
        
        # 3. ENVIRONMENT & GUIDANCE
        self.target = np.array([95.0, 95.0, 35.0])
        self.gps_mask_zone = {'x': (40, 65), 'y': (40, 65)}
        self.obstacles = [{'pos': np.array([50, 55, 25]), 'radius': 12.0}]
        
        # 4. PID CONTROL STACK
        self.kp = 9.0        # Velocity Tracking Gain
        self.ki = 0.15       # Steady-state Error Correction
        self.kd = 4.5        # Damping / Stability
        self.v_integral = np.zeros(3)
        
        self.history = []

    def get_velocity_setpoint(self):
        """Guidance Layer: Vortex Bypass Logic."""
        to_target = self.target - self.slam_pos
        dist_target = np.linalg.norm(to_target)
        
        # Commanded Cruise Speed: 12.5 m/s
        v_base = (to_target / max(dist_target, 1.0)) * 12.5
        
        v_bypass = np.zeros(3)
        for obs in self.obstacles:
            diff = self.slam_pos - obs['pos']
            dist = np.linalg.norm(diff)
            safety_margin = obs['radius'] + 15.0 # TRL 7 Buffer
            
            if dist < safety_margin + 20:
                # Calculate Vortex Tangent[cite: 1]
                tangent = np.array([-diff[1], diff[0], 0])
                if np.dot(tangent, to_target) < 0: 
                    tangent = -tangent
                
                # Dynamic Blending based on proximity
                blend = np.clip((safety_margin + 20 - dist) / 20, 0, 1)
                v_bypass = (tangent / np.linalg.norm(tangent)) * 15.0 * blend
                v_base = v_base * (1 - blend)
                
        return v_base + v_bypass

    def motor_controller(self, v_target):
        """Control Layer: PID tracking of velocity setpoint[cite: 1]."""
        v_err = v_target - self.true_vel
        
        # Update Integral with basic Anti-Windup[cite: 1]
        self.v_integral += v_err * self.dt
        self.v_integral = np.clip(self.v_integral, -5, 5) 
        
        # PID Output (Thrust Request)
        thrust = (self.kp * v_err) + (self.ki * self.v_integral)
        
        # Hardware Saturation Guard
        mag = np.linalg.norm(thrust)
        if mag > self.max_thrust:
            thrust = (thrust / mag) * self.max_thrust
        return thrust

    def step(self):
        # A. SENSORS (Simulating GPS Masking & Drift)[cite: 1]
        mx, my = self.gps_mask_zone['x'], self.gps_mask_zone['y']
        if mx[0] <= self.true_pos[0] <= mx[1] and my[0] <= self.true_pos[1] <= my[1]:
            # SLAM Mode: Drift is coupled to velocity[cite: 1]
            drift = (self.true_vel * 0.012) + np.random.normal(0, 0.04, 3)
            self.slam_pos += self.true_vel * self.dt + drift
        else:
            # GPS Mode: High Precision
            self.slam_pos = self.true_pos + np.random.normal(0, 0.05, 3)

        # B. GUIDANCE & PID CONTROL[cite: 1]
        v_target = self.get_velocity_setpoint()
        thrust_cmd = self.motor_controller(v_target)

        # C. PHYSICS (Plant Dynamics)[cite: 1]
        # accel = (Thrust - Drag) / Mass
        drag_force = -self.drag * self.true_vel
        accel = (thrust_cmd + drag_force) / self.mass
        
        # Integrate
        self.true_vel += accel * self.dt
        self.true_pos += self.true_vel * self.dt
        
        self.history.append({'true': self.true_pos.copy(), 'slam': self.slam_pos.copy()})

# --- EXECUTION ---
sim = ESPIRIDI_ARF_TRL7_Integrated()
for _ in range(1500):
    sim.step()
    if np.linalg.norm(sim.true_pos - sim.target) < 0.6: 
        break

# Visualization
tp = np.array([h['true'] for h in sim.history])
sp = np.array([h['slam'] for h in sim.history])

plt.figure(figsize=(10, 6))
plt.plot(tp[:,0], tp[:,1], 'cyan', lw=2, label='True Physical Trajectory')
plt.plot(sp[:,0], sp[:,1], 'orange', ls='--', alpha=0.5, label='SLAM/GPS Estimate')
plt.scatter(sim.target[0], sim.target[1], c='gold', marker='*', s=300, label='Target')
plt.title("ESPIRIDI ARF TRL 7: PID-Guidance SITL Demonstration")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()