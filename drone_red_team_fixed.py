#!/usr/bin/env python3
"""
Drone Cybersecurity Wargaming Simulation Framework
For authorized security testing and training scenarios only

BUGS FIXED:
1. MAVLinkMessage.to_bytes(): msg_id packed as '<B' (ubyte, max 255), but
   COMPONENT_ARM_DISARM used msg_id=400 and countermeasures used msg_id=500/501
   — all overflow a single byte. Fixed by:
     a) Using '<H' (uint16) for msg_id in to_bytes() to support extended IDs
     b) _process_command reads msg_id as uint16 to match

2. _process_command used a hardcoded byte offset [5:6] to read msg_id assuming
   the old single-byte format. Updated to read 2 bytes with '<H'.

3. OSError "Address already in use": Added SO_REUSEPORT where available, and
   each scenario now uses a unique port range to avoid collisions on re-runs.

4. RedTeamFramework attack methods (gps_spoofing_attack, command_injection_attack)
   referenced an undefined `scenario_context` local variable. Fixed by passing
   scenario_context as an explicit parameter consistently across all methods.

5. BlueTeamFramework.actions_dict was set as a class attribute after instantiation
   via `BlueTeamFramework.actions_dict = ...` which creates a shared mutable
   class-level dict. Moved to __init__ as an instance attribute.

6. generate_report() accepted an optional scenario_context kwarg in the final
   cell but the method signature didn't support it — caused a TypeError. Fixed.

7. WargamingScenario._display_results() referenced report['blue_team_actions']
   which was never populated in generate_report(). Added blue_team_actions
   collection to generate_report().
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import random
import time
import logging
from datetime import datetime
import json
import struct
import threading
import socket

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DroneStatus:
    """Represents the current status of a simulated drone"""
    drone_id: str
    position: Tuple[float, float, float]   # lat, lon, alt
    battery: float
    armed: bool
    mode: str
    velocity: Tuple[float, float, float]   # x, y, z
    timestamp: float
    countermeasures_active: bool = False
    system_hardened: bool = True           # Hardened by default
    flight_profile: str = "STANDARD"      # STANDARD | EVASIVE | LOITER


@dataclass
class AttackVector:
    """Represents a potential attack vector"""
    name: str
    severity: str
    description: str
    success_rate: float   # Probability of success
    detection_rate: float # Probability of detection if successful
    impact: str = "Medium"


@dataclass
class BlueTeamAction:
    """Represents a potential Blue Team defensive action"""
    name: str
    description: str
    effectiveness: Dict[str, float]        # {attack_type: mitigation_prob}
    detection_capability: Dict[str, float] # {attack_type: detection_prob}


# ---------------------------------------------------------------------------
# MAVLink message (simplified)
# ---------------------------------------------------------------------------

# FIX 1: The original code packed msg_id as a single ubyte ('<B'), which
# overflows for IDs > 255 (e.g., 400 = ARM/DISARM, 500/501 = custom).
# We use a 2-byte uint16 ('<H') for msg_id so all custom IDs fit.
# The header layout becomes: magic(B) len(B) seq(B) sys_id(B) comp_id(B) msg_id(H)

MAV_HEADER_FMT = '<BBBBBH'   # magic, length, seq, sys_id, comp_id, msg_id(uint16)
MAV_HEADER_SIZE = struct.calcsize(MAV_HEADER_FMT)   # = 7 bytes


class MAVLinkMessage:
    """Simplified MAVLink-like message — msg_id stored as uint16 to support IDs > 255."""

    def __init__(self, msg_id: int, payload: bytes):
        if not (0 <= msg_id <= 65535):
            raise ValueError(f"msg_id {msg_id} out of uint16 range [0, 65535]")
        self.magic      = 0xFE
        self.length     = len(payload)
        self.sequence   = random.randint(0, 255)
        self.system_id  = 1
        self.component_id = 1
        self.msg_id     = msg_id
        self.payload    = payload

    def to_bytes(self) -> bytes:
        """Serialise to bytes: header + payload + uint16 checksum."""
        header = struct.pack(
            MAV_HEADER_FMT,
            self.magic, self.length, self.sequence,
            self.system_id, self.component_id, self.msg_id  # now uint16
        )
        checksum_data = header + self.payload
        checksum = 0
        for byte in checksum_data:
            checksum = (checksum + byte) & 0xFFFF
        return header + self.payload + struct.pack('<H', checksum)


# ---------------------------------------------------------------------------
# Blue Team Framework
# ---------------------------------------------------------------------------

class BlueTeamFramework:
    """Blue team framework for drone cybersecurity defence."""

    def __init__(self):
        self.actions: List[BlueTeamAction] = [
            BlueTeamAction(
                name="Deploy Firewall",
                description="Implement network filtering rules",
                effectiveness={'Network Scan': 0.5, 'Brute Force Auth': 0.6},
                detection_capability={'Network Scan': 0.7}
            ),
            BlueTeamAction(
                name="Activate IDS",
                description="Monitor network traffic for malicious patterns",
                effectiveness={},
                detection_capability={
                    'GPS Spoofing': 0.4, 'Command Injection': 0.5,
                    'Telemetry Sniffing': 0.8, 'Swarm Attack': 0.6,
                    'Electronic Warfare Attack': 0.5
                }
            ),
            BlueTeamAction(
                name="Patch Firmware",
                description="Update drone firmware to address known vulnerabilities",
                effectiveness={'Firmware Exploit': 0.9, 'Supply Chain Attack': 0.7},
                detection_capability={}
            ),
            BlueTeamAction(
                name="Activate Countermeasures",
                description="Engage onboard defensive systems (anti-jamming, anti-spoofing)",
                effectiveness={
                    'Jamming Attack': 0.8, 'GPS Spoofing': 0.7,
                    'Electronic Warfare Attack': 0.7
                },
                detection_capability={
                    'Jamming Attack': 0.6, 'GPS Spoofing': 0.6,
                    'Electronic Warfare Attack': 0.6
                }
            ),
            BlueTeamAction(
                name="Implement Strong Authentication",
                description="Enforce strong password policies and MFA",
                effectiveness={'Weak Auth': 0.9, 'Brute Force Auth': 0.8},
                detection_capability={'Brute Force Auth': 0.7}
            ),
            BlueTeamAction(
                name="Encrypt Communications",
                description="Secure communication channels with encryption",
                effectiveness={'Telemetry Sniffing': 0.9, 'Unencrypted Comms': 1.0},
                detection_capability={}
            ),
            BlueTeamAction(
                name="Supply Chain Monitoring",
                description="Monitor hardware/software supply chains for compromises",
                effectiveness={},
                detection_capability={'Supply Chain Attack': 0.9}
            ),
        ]
        # FIX 5: actions_dict as an instance attribute, not a post-hoc class attribute
        self.actions_dict: Dict[str, BlueTeamAction] = {a.name: a for a in self.actions}

        logger.info(f"Blue Team Framework initialised with {len(self.actions)} actions.")


# ---------------------------------------------------------------------------
# Drone Simulator
# ---------------------------------------------------------------------------

class DroneSimulator:
    """Simulates a drone for wargaming scenarios."""

    def __init__(self, drone_id: str, port: int = 14550):
        self.drone_id = drone_id
        self.port = port
        self.status = DroneStatus(
            drone_id=drone_id,
            position=(37.7749, -122.4194, 100.0),  # San Francisco
            battery=95.0,
            armed=False,
            mode="STABILIZE",
            velocity=(0.0, 0.0, 0.0),
            timestamp=time.time(),
            countermeasures_active=False,
            system_hardened=True,
            flight_profile="STANDARD"
        )
        self.socket: Optional[socket.socket] = None
        self.running = False
        self.vulnerabilities = {
            'weak_auth':        not self.status.system_hardened,
            'unencrypted_comms': not self.status.system_hardened,
            'outdated_firmware': not self.status.system_hardened,
            'open_ports': [14550] if self.status.system_hardened else [22, 23, 80, 14550],
        }
        self.active_defenses: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the drone simulator (binds UDP socket, starts telemetry thread)."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # SO_REUSEPORT (Linux/macOS) allows quick reuse after restart
        if hasattr(socket, 'SO_REUSEPORT'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.socket.bind(('localhost', self.port))
        self.running = True

        telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        telemetry_thread.start()

        logger.info(f"Drone {self.drone_id} started on port {self.port} "
                    f"(profile={self.status.flight_profile})")
        self._listen_for_commands()

    def stop(self):
        """Stop the drone simulator."""
        self.running = False
        if self.socket:
            self.socket.close()
            self.socket = None

    # ------------------------------------------------------------------
    # Defence API
    # ------------------------------------------------------------------

    def apply_defense(self, defense_name: str, active: bool):
        self.active_defenses[defense_name] = active
        logger.info(f"Drone {self.drone_id}: defence '{defense_name}' → {active}")

    def is_vulnerable(self, vulnerability_type: str) -> bool:
        base = self.vulnerabilities.get(vulnerability_type, False)
        if vulnerability_type == 'outdated_firmware' and self.active_defenses.get('Patch Firmware'):
            return False
        if vulnerability_type == 'weak_auth' and self.active_defenses.get('Implement Strong Authentication'):
            return False
        if vulnerability_type == 'unencrypted_comms' and self.active_defenses.get('Encrypt Communications'):
            return False
        return base

    def get_detection_chance(self, attack_type: str) -> float:
        chance = 0.1  # base
        if self.active_defenses.get('Activate IDS'):
            if attack_type in ('GPS Spoofing', 'Command Injection', 'Telemetry Sniffing',
                               'Swarm Attack', 'Electronic Warfare Attack'):
                chance += 0.3
        if self.active_defenses.get('Supply Chain Monitoring') and attack_type == 'Supply Chain Attack':
            chance += 0.5
        if self.active_defenses.get('Activate Countermeasures'):
            if attack_type in ('Jamming Attack', 'GPS Spoofing', 'Electronic Warfare Attack'):
                chance += 0.4
        return min(chance, 1.0)

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    def _telemetry_loop(self):
        while self.running:
            self._update_status()
            try:
                telemetry = self._create_telemetry_message()
                self._broadcast_telemetry(telemetry)
            except Exception as e:
                logger.debug(f"Telemetry error for {self.drone_id}: {e}")
            time.sleep(1)

    def _update_status(self):
        self.status.timestamp = time.time()
        self.status.battery -= random.uniform(0.01, 0.05)

        if self.status.armed:
            fp = self.status.flight_profile
            if fp == "STANDARD":
                self.status.velocity = (
                    random.uniform(-5, 5),
                    random.uniform(-5, 5),
                    random.uniform(-2, 2),
                )
            elif fp == "EVASIVE":
                self.status.velocity = (
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                    random.uniform(-5, 5),
                )
            elif fp == "LOITER":
                self.status.velocity = (
                    random.uniform(-1, 1),
                    random.uniform(-1, 1),
                    random.uniform(-0.5, 0.5),
                )

            lat, lon, alt = self.status.position
            vx, vy, vz = self.status.velocity
            self.status.position = (
                lat + vx * 0.00001,
                lon + vy * 0.00001,
                max(0, alt + vz),
            )

    def _create_telemetry_message(self) -> bytes:
        payload = json.dumps(asdict(self.status)).encode()
        return MAVLinkMessage(33, payload).to_bytes()  # 33 = GLOBAL_POSITION_INT

    def _broadcast_telemetry(self, data: bytes):
        try:
            bcast = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            bcast.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            if not (self.status.countermeasures_active and random.random() < 0.5):
                bcast.sendto(data, ('255.255.255.255', self.port + 1))
            else:
                logger.debug(f"Telemetry suppressed for {self.drone_id} (countermeasures)")
            bcast.close()
        except Exception as e:
            logger.debug(f"Telemetry broadcast failed: {e}")

    def _listen_for_commands(self):
        while self.running:
            try:
                if self.status.countermeasures_active and random.random() < 0.3:
                    logger.debug(f"Command dropped for {self.drone_id} (countermeasures)")
                    time.sleep(0.1)
                    continue
                self.socket.settimeout(1.0)
                try:
                    data, addr = self.socket.recvfrom(4096)
                    self._process_command(data, addr)
                except socket.timeout:
                    pass
            except Exception as e:
                if self.running:
                    logger.error(f"Command loop error on {self.drone_id}: {e}")

    def _process_command(self, data: bytes, addr: tuple):
        """
        FIX 2: msg_id is now uint16 (2 bytes) starting at offset 5.
        Header layout: magic(1) len(1) seq(1) sys(1) comp(1) msg_id(2) = 7 bytes.
        Payload starts at offset 7 (not 8 as before).
        """
        try:
            if len(data) < MAV_HEADER_SIZE:
                return
            _magic, _length, _seq, _sys, _comp, msg_id = struct.unpack_from(
                MAV_HEADER_FMT, data, 0
            )
            payload = data[MAV_HEADER_SIZE:]

            if msg_id == 11:      # SET_MODE
                mode_data = json.loads(payload.decode())
                self.status.mode = mode_data.get('mode', self.status.mode)
                logger.info(f"[{self.drone_id}] Mode → {self.status.mode}")

            elif msg_id == 400:   # COMPONENT_ARM_DISARM
                arm_data = json.loads(payload.decode())
                self.status.armed = arm_data.get('arm', False)
                logger.info(f"[{self.drone_id}] {'Armed' if self.status.armed else 'Disarmed'}")

            elif msg_id == 500:   # Custom: toggle countermeasures
                cm_data = json.loads(payload.decode())
                self.status.countermeasures_active = cm_data.get('active', False)
                logger.info(f"[{self.drone_id}] Countermeasures → {self.status.countermeasures_active}")

            elif msg_id == 501:   # Custom: change flight profile
                profile_data = json.loads(payload.decode())
                new_profile = profile_data.get('profile', "STANDARD")
                if new_profile in ("STANDARD", "EVASIVE", "LOITER"):
                    self.status.flight_profile = new_profile
                    logger.info(f"[{self.drone_id}] Flight profile → {new_profile}")
                else:
                    logger.warning(f"[{self.drone_id}] Invalid profile: {new_profile}")

            # ACK
            ack = MAVLinkMessage(77, b'{"result": 0}')
            self.socket.sendto(ack.to_bytes(), addr)

        except Exception as e:
            logger.error(f"_process_command error on {self.drone_id}: {e}")


# ---------------------------------------------------------------------------
# Red Team Framework
# ---------------------------------------------------------------------------

class RedTeamFramework:
    """Red team framework for drone cybersecurity testing."""

    def __init__(self):
        self.attack_vectors: List[AttackVector] = [
            AttackVector("GPS Spoofing",            "HIGH",     "Inject false GPS coordinates",                         0.8, 0.3),
            AttackVector("Command Injection",        "CRITICAL", "Inject malicious flight commands",                     0.6, 0.5),
            AttackVector("Telemetry Sniffing",       "MEDIUM",   "Intercept telemetry data",                             0.9, 0.1),
            AttackVector("Jamming Attack",           "HIGH",     "Disrupt communication channels",                       0.7, 0.4),
            AttackVector("Firmware Exploit",         "CRITICAL", "Exploit firmware vulnerabilities",                     0.4, 0.7),
            AttackVector("Brute Force Auth",         "MEDIUM",   "Attempt to crack authentication",                      0.5, 0.6),
            AttackVector("Swarm Attack",             "CRITICAL", "Coordinate multiple drones for synchronised attack",   0.5, 0.6),
            AttackVector("Electronic Warfare Attack","HIGH",     "Utilise electronic means to disrupt or deceive",       0.6, 0.5),
            AttackVector("Supply Chain Attack",      "CRITICAL", "Compromise hardware/software during manufacturing",    0.3, 0.8),
        ]
        self.discovered_targets: List[Dict] = []
        self.successful_attacks: List[Dict] = []

    # ------------------------------------------------------------------
    # Helper: generic attack attempt with blue-team awareness
    # ------------------------------------------------------------------

    def _attempt_attack(
        self,
        attack_vector: AttackVector,
        target: DroneSimulator,
        blue_team: Optional[BlueTeamFramework] = None,
    ) -> Dict:
        success = random.random() < attack_vector.success_rate
        detected = random.random() < target.get_detection_chance(attack_vector.name)

        # Apply defence mitigation
        if success and blue_team:
            for defense_name, is_active in target.active_defenses.items():
                if is_active and defense_name in blue_team.actions_dict:
                    defense = blue_team.actions_dict[defense_name]
                    eff = defense.effectiveness.get(attack_vector.name, 0.0)
                    if eff > 0 and random.random() < eff:
                        success = False
                        logger.info(f"Attack '{attack_vector.name}' mitigated by '{defense_name}' on {target.drone_id}")
                        break

        impact = "None"
        if success:
            impact = attack_vector.impact if not detected else f"{attack_vector.impact} (DETECTED)"

        result = {
            'attack_type': attack_vector.name,
            'target': target.drone_id,
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'detected': detected,
            'impact': impact,
        }
        if success:
            self.successful_attacks.append(result)
            logger.warning(f"Attack '{attack_vector.name}' {'DETECTED' if detected else 'SUCCESSFUL'} — impact: {impact}")
        else:
            logger.info(f"Attack '{attack_vector.name}' FAILED{' (detected)' if detected else ''}")
        return result

    # ------------------------------------------------------------------
    # Reconnaissance
    # ------------------------------------------------------------------

    def network_scan(self, target_range: str = "127.0.0.1") -> List[Dict]:
        logger.info(f"Scanning network range: {target_range}")
        port_map = {
            22: "SSH",   23: "Telnet",  80: "HTTP",
            14550: "MAVLink", 14551: "MAVLink-GCS",
            14600: "MAVLink-Friendly", 14700: "MAVLink-Enemy",
        }
        vuln_map = {
            22:    ["weak-passwords", "outdated-version"],
            23:    ["unencrypted-auth", "default-credentials"],
            80:    ["directory-traversal", "xss"],
            14550: ["unencrypted-comms", "no-auth"],
            14551: ["replay-attacks", "packet-injection"],
            14600: ["encrypted-comms-weak-keys"],
            14700: ["unencrypted-comms", "default-credentials", "outdated-firmware"],
        }
        discovered = []
        for port, service in port_map.items():
            if random.random() > 0.3:
                vulns = [v for v in vuln_map.get(port, []) if random.random() > 0.4]
                discovered.append({
                    'ip': target_range, 'port': port, 'service': service,
                    'version': f"v{random.randint(1,5)}.{random.randint(0,9)}",
                    'vulnerabilities': vulns,
                })
        self.discovered_targets = discovered
        logger.info(f"Discovered {len(discovered)} services")
        return discovered

    # ------------------------------------------------------------------
    # Attack methods  (FIX 4: scenario_context replaced by explicit params)
    # ------------------------------------------------------------------

    def gps_spoofing_attack(
        self, target: DroneSimulator, fake_coords: Tuple[float, float, float],
        blue_team: Optional[BlueTeamFramework] = None
    ) -> Dict:
        logger.info(f"GPS spoofing → {target.drone_id}")
        av = next(a for a in self.attack_vectors if a.name == "GPS Spoofing")
        result = self._attempt_attack(av, target, blue_team)
        result['fake_coordinates'] = fake_coords
        return result

    def command_injection_attack(
        self, target: DroneSimulator, malicious_command: str,
        blue_team: Optional[BlueTeamFramework] = None
    ) -> Dict:
        logger.info(f"Command injection → {target.drone_id}")
        av = next(a for a in self.attack_vectors if a.name == "Command Injection")
        result = self._attempt_attack(av, target, blue_team)
        result['command'] = malicious_command
        return result

    def jamming_attack(
        self, frequency: float, duration: int, drones: List[DroneSimulator],
        blue_team: Optional[BlueTeamFramework] = None
    ) -> Dict:
        logger.info(f"Jamming {frequency} MHz for {duration}s")
        av = next(a for a in self.attack_vectors if a.name == "Jamming Attack")
        success = random.random() < av.success_rate
        detected = any(
            random.random() < d.get_detection_chance(av.name) for d in drones
        )
        result = {
            'attack_type': 'Jamming Attack', 'frequency': frequency,
            'duration': duration, 'timestamp': datetime.now().isoformat(),
            'success': success, 'detected': detected,
            'impact': 'Communication disruption' if success else 'None',
        }
        if success:
            self.successful_attacks.append(result)
            logger.warning(f"Jamming {'DETECTED' if detected else 'SUCCESSFUL'}")
        else:
            logger.info("Jamming FAILED")
        return result

    def swarm_attack(
        self, targets: List[DroneSimulator],
        blue_team: Optional[BlueTeamFramework] = None
    ) -> Dict:
        logger.info(f"Swarm attack → {[d.drone_id for d in targets]}")
        av = next(a for a in self.attack_vectors if a.name == "Swarm Attack")
        affected, detected_count = [], 0
        for target in targets:
            success = random.random() < av.success_rate
            det = random.random() < target.get_detection_chance(av.name)
            if success and blue_team:
                for dn, is_active in target.active_defenses.items():
                    if is_active and dn in blue_team.actions_dict:
                        eff = blue_team.actions_dict[dn].effectiveness.get(av.name, 0)
                        if eff > 0 and random.random() < eff:
                            success = False
                            break
            if success:
                affected.append(target.drone_id)
            if det:
                detected_count += 1

        impact = f"Affected {len(affected)}/{len(targets)} drones" if affected else "None"
        result = {
            'attack_type': 'Swarm Attack',
            'targets': [d.drone_id for d in targets],
            'timestamp': datetime.now().isoformat(),
            'success': bool(affected),
            'detected': detected_count > 0,
            'impact': impact,
            'affected_drones': affected,
        }
        if affected:
            self.successful_attacks.append(result)
            logger.warning(f"Swarm {'DETECTED' if result['detected'] else 'SUCCESSFUL'}: {impact}")
        else:
            logger.info("Swarm attack FAILED")
        return result

    def electronic_warfare_attack(
        self, frequency_range: Tuple[float, float], duration: int,
        drones: List[DroneSimulator], blue_team: Optional[BlueTeamFramework] = None
    ) -> Dict:
        logger.info(f"EW attack {frequency_range} MHz for {duration}s")
        av = next(a for a in self.attack_vectors if a.name == "Electronic Warfare Attack")
        success = random.random() < av.success_rate
        detected = any(random.random() < d.get_detection_chance(av.name) for d in drones)
        if success and blue_team:
            for drone in drones:
                for dn, is_active in drone.active_defenses.items():
                    if is_active and dn in blue_team.actions_dict:
                        eff = blue_team.actions_dict[dn].effectiveness.get(av.name, 0)
                        if eff > 0 and random.random() < eff:
                            success = False
                            break
                if not success:
                    break

        impact = (f"Comms/Nav disrupted in range {frequency_range} MHz"
                  if success else "None")
        result = {
            'attack_type': 'Electronic Warfare Attack',
            'frequency_range': frequency_range, 'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'success': success, 'detected': detected, 'impact': impact,
        }
        if success:
            self.successful_attacks.append(result)
            logger.warning(f"EW {'DETECTED' if detected else 'SUCCESSFUL'}: {impact}")
        else:
            logger.info("EW attack FAILED")
        return result

    def supply_chain_attack(
        self, target: DroneSimulator, compromise_type: str,
        blue_team: Optional[BlueTeamFramework] = None
    ) -> Dict:
        logger.info(f"Supply-chain attack → {target.drone_id} ({compromise_type})")
        av = next(a for a in self.attack_vectors if a.name == "Supply Chain Attack")
        success = random.random() < av.success_rate
        detected = random.random() < target.get_detection_chance(av.name)
        if success and blue_team:
            for dn, is_active in target.active_defenses.items():
                if is_active and dn in blue_team.actions_dict:
                    eff = blue_team.actions_dict[dn].effectiveness.get(av.name, 0)
                    if eff > 0 and random.random() < eff:
                        success = False
                        break

        impact = (f"Compromise exploited: {compromise_type} on {target.drone_id}"
                  if success else "None")
        result = {
            'attack_type': 'Supply Chain Attack',
            'target': target.drone_id, 'compromise_type': compromise_type,
            'timestamp': datetime.now().isoformat(),
            'success': success, 'detected': detected, 'impact': impact,
        }
        if success:
            self.successful_attacks.append(result)
            logger.warning(f"Supply-chain {'DETECTED' if detected else 'SUCCESSFUL'}: {impact}")
        else:
            logger.info("Supply-chain attack FAILED")
        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    # FIX 6: generate_report() now accepts optional scenario_context kwarg
    #         and FIX 7: populates blue_team_actions from drone state
    def generate_report(self, scenario_context: Optional[Dict] = None) -> Dict:
        # Collect blue team actions from drones if available
        blue_team_actions: Dict[str, Dict] = {}
        drones: Dict[str, DroneSimulator] = {}
        if scenario_context:
            drones = scenario_context.get('drones', {})
            for drone_id, drone in drones.items():
                if drone.active_defenses:
                    blue_team_actions[drone_id] = dict(drone.active_defenses)

        report = {
            'assessment_date': datetime.now().isoformat(),
            'targets_discovered': len(self.discovered_targets),
            'attacks_attempted': len(self.successful_attacks),
            'success_rate': len(self.successful_attacks) / max(1, len(self.attack_vectors)) * 100,
            'critical_vulnerabilities': [
                t for t in self.discovered_targets
                if any(
                    'unencrypted' in v or 'no-auth' in v or 'weak-keys' in v
                    for v in t.get('vulnerabilities', [])
                )
            ],
            'successful_attacks': self.successful_attacks,
            'blue_team_actions': blue_team_actions,   # FIX 7
            'recommendations': [
                "Implement encrypted communication protocols with strong key management",
                "Enable strong authentication mechanisms",
                "Deploy intrusion detection systems and monitoring for anomalies",
                "Regular firmware updates and security patches",
                "Implement GPS anti-spoofing and validation measures",
                "Use frequency hopping and spread spectrum techniques for communication",
                "Strengthen supply chain security protocols and hardware integrity checks",
                "Implement advanced anti-jamming and cognitive electronic warfare defences",
                "Develop countermeasures against coordinated swarm attacks and autonomous response",
            ],
        }

        # War-scenario metrics
        def _matches(attack, tag):
            return (tag in str(attack.get('target', '')) or
                    any(tag in t for t in attack.get('targets', [])))

        friendly_hits = [a for a in self.successful_attacks if _matches(a, 'FRIENDLY_DRONE')]
        enemy_hits    = [a for a in self.successful_attacks if _matches(a, 'ENEMY_DRONE')]
        report['war_scenario_metrics'] = {
            'friendly_drones_attacked':          len(friendly_hits),
            'enemy_drones_attacked':             len(enemy_hits),
            'successful_attacks_on_friendly':    sum(1 for a in friendly_hits if a.get('success')),
            'successful_attacks_on_enemy':       sum(1 for a in enemy_hits   if a.get('success')),
            'detected_attacks_on_friendly':      sum(1 for a in friendly_hits if a.get('detected')),
            'detected_attacks_on_enemy':         sum(1 for a in enemy_hits   if a.get('detected')),
            'impactful_attacks_on_friendly':     sum(1 for a in friendly_hits if a.get('success') and a.get('impact', 'None') != 'None'),
            'impactful_attacks_on_enemy':        sum(1 for a in enemy_hits   if a.get('success') and a.get('impact', 'None') != 'None'),
        }
        return report


# ---------------------------------------------------------------------------
# Wargaming Scenario orchestrator
# ---------------------------------------------------------------------------

# FIX 3: Port ranges are spread out and do not collide with prior runs.
_FRIENDLY_BASE_PORT = 15100
_ENEMY_BASE_PORT    = 15200
_GENERIC_BASE_PORT  = 15000


class WargamingScenario:
    """Controls and orchestrates wargaming scenarios."""

    def __init__(self):
        self.drones: Dict[str, DroneSimulator] = {}
        self.red_team  = RedTeamFramework()
        self.blue_team = BlueTeamFramework()
        self.blue_team_score = 0
        self.red_team_score  = 0

    def _start_drone(self, drone: DroneSimulator):
        t = threading.Thread(target=drone.start, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Generic scenario
    # ------------------------------------------------------------------

    def setup_scenario(self, num_drones: int = 3):
        logger.info(f"Setting up scenario with {num_drones} drones")
        for i in range(num_drones):
            drone_id = f"DRONE_{i+1}"
            port = _GENERIC_BASE_PORT + i
            drone = DroneSimulator(drone_id, port)
            self.drones[drone_id] = drone
            self._start_drone(drone)
        time.sleep(2)
        logger.info("Scenario setup complete")

    def run_red_team_exercise(self):
        logger.info("Starting red team exercise")
        self.red_team.network_scan()
        all_drones = list(self.drones.values())

        for drone in all_drones:
            self.red_team.gps_spoofing_attack(drone, (40.7128, -74.0060, 50.0))
            self.red_team.command_injection_attack(drone, "EMERGENCY_LAND")
            time.sleep(1)

        self.red_team.jamming_attack(2400.0, 5, all_drones)
        if len(all_drones) > 1:
            self.red_team.swarm_attack(all_drones[:2])
        self.red_team.electronic_warfare_attack((2000.0, 3000.0), 10, all_drones)
        if all_drones:
            self.red_team.supply_chain_attack(all_drones[0], "Malicious Firmware")

        report = self.red_team.generate_report(scenario_context={'drones': self.drones})
        self._display_results(report)

    # ------------------------------------------------------------------
    # War scenario
    # ------------------------------------------------------------------

    def run_war_scenario(self, num_friendly_drones: int = 2, num_enemy_drones: int = 2):
        logger.info(f"War scenario: {num_friendly_drones} friendly, {num_enemy_drones} enemy")

        friendly, enemy = [], []

        for i in range(num_friendly_drones):
            drone_id = f"FRIENDLY_DRONE_{i+1}"
            drone = DroneSimulator(drone_id, _FRIENDLY_BASE_PORT + i)
            self.drones[drone_id] = drone
            friendly.append(drone)
            self._start_drone(drone)

        for i in range(num_enemy_drones):
            drone_id = f"ENEMY_DRONE_{i+1}"
            drone = DroneSimulator(drone_id, _ENEMY_BASE_PORT + i)
            drone.status.system_hardened = False
            drone.vulnerabilities = {
                'weak_auth': True, 'unencrypted_comms': True,
                'outdated_firmware': True,
                'open_ports': [22, 23, 80, 14550, _ENEMY_BASE_PORT + i],
            }
            self.drones[drone_id] = drone
            enemy.append(drone)
            self._start_drone(drone)

        time.sleep(3)
        logger.info("War scenario setup complete")

        # Apply Blue Team defences to friendly drones
        for drone in friendly:
            for action_name in self.blue_team.actions_dict:
                drone.apply_defense(action_name, True)

        logger.info("Executing red team attacks")

        for drone in friendly:
            self.red_team.supply_chain_attack(drone, "Malicious Hardware Implant", self.blue_team)
            time.sleep(0.5)

        if enemy:
            self.red_team.swarm_attack(enemy, self.blue_team)
            time.sleep(1)

        self.red_team.electronic_warfare_attack(
            (2200.0, 2500.0), 15, friendly + enemy, self.blue_team
        )
        time.sleep(1)

        for drone in enemy:
            self.red_team.command_injection_attack(drone, "RETURN_TO_BASE", self.blue_team)
            time.sleep(0.5)

        self.red_team.network_scan("127.0.0.1")

        logger.info("Red team attacks complete")
        report = self.red_team.generate_report(scenario_context={'drones': self.drones})
        self._display_results(report)
        return report

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _display_results(self, report: Dict):
        print("\n" + "="*60)
        print("DRONE CYBERSECURITY WARGAMING RESULTS")
        print("="*60)
        print(f"Assessment Date:        {report['assessment_date']}")
        print(f"Targets Discovered:     {report['targets_discovered']}")
        print(f"Attacks Attempted:      {report['attacks_attempted']}")
        print(f"Success Rate:           {report['success_rate']:.1f}%")
        print(f"\nCritical Vulnerabilities Found: {len(report['critical_vulnerabilities'])}")

        print("\nSuccessful Attacks:")
        for attack in report['successful_attacks']:
            det = 'DETECTED' if attack['detected'] else 'UNDETECTED'
            print(f"  - {attack['attack_type']}: {det}")
            if attack.get('impact', 'None') != 'None':
                print(f"    Impact: {attack['impact']}")

        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")

        # Blue team actions
        if report.get('blue_team_actions'):
            print("\nBlue Team Actions Taken:")
            for drone_id, actions in report['blue_team_actions'].items():
                print(f"  {drone_id}:")
                for action, active in actions.items():
                    print(f"    - {action}: {'Active' if active else 'Inactive'}")

        # War scenario metrics
        if report.get('war_scenario_metrics'):
            m = report['war_scenario_metrics']
            print("\nWar Scenario Metrics:")
            print(f"  Friendly Drones Attacked:       {m['friendly_drones_attacked']}")
            print(f"  Enemy Drones Attacked:          {m['enemy_drones_attacked']}")
            print(f"  Successful on Friendly:         {m['successful_attacks_on_friendly']}")
            print(f"  Successful on Enemy:            {m['successful_attacks_on_enemy']}")
            print(f"  Detected on Friendly:           {m['detected_attacks_on_friendly']}")
            print(f"  Detected on Enemy:              {m['detected_attacks_on_enemy']}")
            print(f"  Impactful on Friendly:          {m['impactful_attacks_on_friendly']}")
            print(f"  Impactful on Enemy:             {m['impactful_attacks_on_enemy']}")

        print("="*60)

    def cleanup(self):
        for drone in self.drones.values():
            drone.stop()


# ---------------------------------------------------------------------------
# Analysis helper
# ---------------------------------------------------------------------------

def analyze_wargaming_results(report: Dict) -> Dict:
    analysis = {
        "summary":          "Wargaming Scenario Analysis Summary\n" + "="*40 + "\n",
        "attack_summary":   "Red Team Attack Summary:\n",
        "blue_team_summary":"Blue Team Defence Summary:\n",
        "scenario_outcome": "Scenario Outcome:\n",
        "lessons_learned":  "Lessons Learned:\n",
        "recommendations":  "Recommendations:\n",
    }

    attacks           = report.get('successful_attacks', [])
    successful_count  = len(attacks)
    detected_count    = sum(1 for a in attacks if a.get('detected'))
    undetected_count  = successful_count - detected_count
    impactful_count   = sum(1 for a in attacks if a.get('impact', 'None') != 'None')

    analysis["attack_summary"] += (
        f"Total Attacks Attempted:   {report.get('attacks_attempted', 0)}\n"
        f"Successful:                {successful_count}\n"
        f"Detected:                  {detected_count}\n"
        f"Undetected:                {undetected_count}\n"
        f"Impactful:                 {impactful_count}\n\n"
    )
    if attacks:
        analysis["attack_summary"] += "Details:\n"
        for a in attacks:
            analysis["attack_summary"] += (
                f"  - {a.get('attack_type')}: target={a.get('target','N/A')} "
                f"detected={a.get('detected')} impact={a.get('impact','None')}\n"
            )
            if 'affected_drones' in a:
                analysis["attack_summary"] += f"    Affected: {', '.join(a['affected_drones'])}\n"

    bt = report.get('blue_team_actions', {})
    analysis["blue_team_summary"] += "Blue Team Actions on Friendly Drones:\n"
    if bt:
        for drone_id, actions in bt.items():
            analysis["blue_team_summary"] += f"  {drone_id}:\n"
            for action, active in actions.items():
                analysis["blue_team_summary"] += f"    - {action}: {'Active' if active else 'Inactive'}\n"
    else:
        analysis["blue_team_summary"] += "  None recorded.\n"

    wm = report.get('war_scenario_metrics', {})
    fi = wm.get('impactful_attacks_on_friendly', 0)
    ei = wm.get('impactful_attacks_on_enemy', 0)
    fd = wm.get('detected_attacks_on_friendly', 0)

    if fi > ei and undetected_count > detected_count:
        outcome = "Red Team achieved significant impact and evaded detection."
    elif fi < ei and detected_count > undetected_count:
        outcome = "Blue Team defences were largely effective."
    elif fi == ei and undetected_count == detected_count:
        outcome = "Stalemate — comparable impact and detection rates."
    else:
        outcome = "Mixed results — further analysis required."

    analysis["scenario_outcome"] += (
        f"Outcome: {outcome}\n"
        f"Friendly Drones Impacted: {fi}\n"
        f"Enemy Drones Impacted:    {ei}\n"
        f"Detected on Friendly:     {fd}\n\n"
    )

    analysis["lessons_learned"] += "Based on this simulation:\n"
    if undetected_count:
        analysis["lessons_learned"] += f"- {undetected_count} attacks went undetected.\n"
    if impactful_count:
        analysis["lessons_learned"] += f"- {impactful_count} attacks had significant impact.\n"
    if not bt:
        analysis["lessons_learned"] += "- No Blue Team actions were active.\n"
    analysis["lessons_learned"] += "- Simulation highlights defence gaps for remediation.\n\n"

    analysis["recommendations"] += "Recommendations:\n"
    for rec in report.get('recommendations', []):
        analysis["recommendations"] += f"  • {rec}\n"
    if undetected_count:
        analysis["recommendations"] += "  • Enhance IDS visibility for subtle attack techniques.\n"
    if impactful_count:
        analysis["recommendations"] += "  • Prioritise defences against high-impact vectors.\n"

    final = "".join(analysis.values())
    return {"summary_report": final}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Drone Cybersecurity Wargaming Framework")
    print("For authorised security testing only")
    print("-" * 40)

    scenario = WargamingScenario()
    try:
        report = scenario.run_war_scenario(num_friendly_drones=2, num_enemy_drones=2)

        analysis = analyze_wargaming_results(report)
        print("\n" + "="*60)
        print("WARGAMING SCENARIO ANALYSIS SUMMARY")
        print("="*60)
        print(analysis["summary_report"])
        print("="*60)

        print("\nPress Ctrl+C to stop the simulation")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
        scenario.cleanup()
        print("Done.")
    except Exception as e:
        logger.error(f"Scenario error: {e}", exc_info=True)
        scenario.cleanup()
