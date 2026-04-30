import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ARF_HighFidelity_v1_2:
    def __init__(self):
        # 1. PHYSICAL CONSTANTS & CONFIGURATION[cite: 2]
        self.dt = 0.1  # Time step (seconds)
        self.target = np.array([90.0, 90.0, 30.0])
        self.gps_mask_zone = {'x': (40, 60), 'y': (40, 60)}  #[cite: 2, 3]
        
        # 2. STATE VECTORS
        # Ground Truth (Physical Reality)
        self.true_pos = np.array([0.0, 0.0, 5.0])
        self.true_vel = np.array([0.0, 0.0, 0.0])
        
        # Kalman Filter State: [x, y, z, vx, vy, vz]
        self.kf_x = np.zeros((6, 1))
        self.kf_P = np.eye(6) * 10.0  # Initial uncertainty
        
        # 3. OBSTACLE DEFINITIONS
        self.obstacles = [
            {'pos': np.array([50, 50, 25]), 'radius': 10.0},
            {'pos': np.array([75, 30, 20]), 'radius': 7.0}
        ]
        
        self.history = []

    def is_gps_masked(self):
        """Check if drone is in terrain-masking zone[cite: 2, 3]."""
        x, y = self.true_pos[0], self.true_pos[1]
        mx, my = self.gps_mask_zone['x'], self.gps_mask_zone['y']
        return mx[0] <= x <= mx[1] and my[0] <= y <= my[1]

    def get_nav_control(self, est_pos):
        """Exponential Potential Field Navigation[cite: 1, 2]."""
        # Attractive force to target
        dist_to_target = np.linalg.norm(self.target - est_pos)
        f_att = 2.0 * (self.target - est_pos) / max(dist_to_target, 1.0)
        
        # Repulsive force (The Obstacle Avoidance)
        f_rep = np.zeros(3)
        for obs in self.obstacles:
            diff = est_pos - obs['pos']
            dist = np.linalg.norm(diff)
            safety_buffer = obs['radius'] + 3.0  # 3m clearance[cite: 2]
            
            if dist < safety_buffer:
                # EXPONENTIAL WALL: Repulsion grows toward infinity[cite: 1, 3]
                mag = 2000.0 * np.exp(safety_buffer - dist)
                f_rep += (diff / dist) * mag
            elif dist < 15.0:
                # Gradual steering influence
                mag = 100.0 * (1.0/dist - 1.0/15.0) / (dist**2)
                f_rep += (diff / dist) * mag
                
        return f_att + f_rep

    def update_kf_joseph(self, z):
        """Joseph Form Kalman Filter Update for stability."""
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        R = np.eye(3) * 0.5  # Measurement noise
        
        # Innovation
        S = H @ self.kf_P @ H.T + R
        K = self.kf_P @ H.T @ np.linalg.inv(S)
        
        # Joseph Form: P = (I-KH)P(I-KH)' + KRK'[cite: 3]
        I_KH = np.eye(6) - K @ H
        self.kf_P = I_KH @ self.kf_P @ I_KH.T + K @ R @ K.T
        
        y = z.reshape(3, 1) - H @ self.kf_x
        self.kf_x = self.kf_x + K @ y

    def step(self):
        # A. KF PREDICT[cite: 3]
        F = np.eye(6)
        F[0, 3], F[1, 4], F[2, 5] = self.dt, self.dt, self.dt
        Q = np.eye(6) * 0.1
        self.kf_x = F @ self.kf_x
        self.kf_P = F @ self.kf_P @ F.T + Q
        
        # B. KF UPDATE (If GPS available)[cite: 2, 3]
        if not self.is_gps_masked():
            gps_noise = np.random.normal(0, 0.5, 3)
            z = self.true_pos + gps_noise
            self.update_kf_joseph(z)
            
        # C. CONTROL & PHYSICS[cite: 1, 2]
        est_pos = self.kf_x[:3].flatten()
        control_accel = self.get_nav_control(est_pos)
        
        # EULER INTEGRATION [H1] FIX: Velocity leads position[cite: 3]
        self.true_vel += control_accel * self.dt
        # Physical constraints (max speed 15 m/s)[cite: 2]
        speed = np.linalg.norm(self.true_vel)
        if speed > 15.0:
            self.true_vel = (self.true_vel / speed) * 15.0
            
        self.true_pos += self.true_vel * self.dt
        self.history.append(self.true_pos.copy())

# --- RENDERING ENGINE ---
def run_simulation():
    sim = ARF_HighFidelity_v1_2()
    for _ in range(400):
        sim.step()
        if np.linalg.norm(sim.true_pos - sim.target) < 2.0:
            break
            
    path = np.array(sim.history)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Trajectory[cite: 2]
    ax.plot(path[:,0], path[:,1], path[:,2], 'g-', lw=2, label='True Flight Path')
    
    # Plot Mask Zone[cite: 2, 3]
    mx, my = sim.gps_mask_zone['x'], sim.gps_mask_zone['y']
    ax.bar3d(mx[0], my[0], 0, mx[1]-mx[0], my[1]-my[0], 60, color='orange', alpha=0.1)
    
    # Plot Obstacles[cite: 1]
    for obs in sim.obstacles:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = obs['pos'][0] + obs['radius'] * np.cos(u) * np.sin(v)
        y = obs['pos'][1] + obs['radius'] * np.sin(u) * np.sin(v)
        z = obs['pos'][2] + obs['radius'] * np.cos(v)
        ax.plot_wireframe(x, y, z, color="red", alpha=0.3)
        
    ax.scatter(*sim.target, color='gold', marker='*', s=200, label='Target')
    ax.set_title("ESPIRIDI ARF v1.2 High-Fidelity Simulation")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    run_simulation()