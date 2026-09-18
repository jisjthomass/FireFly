"""
attack_sim.py — Attack Simulation Lab
======================================
Generates synthetic network attack packets using Scapy for live demos.
Each method creates packets mimicking a specific NSL-KDD attack category.
"""

from scapy.all import IP, TCP, UDP, ICMP, Raw, Ether, RandShort
import random
import time


class AttackSimulator:
    """Generates realistic synthetic attack packets for all 5 threat classes."""
    
    def __init__(self, target_ip: str = "192.168.1.100", attacker_ip: str = "10.0.0.1"):
        self.target_ip = target_ip
        self.attacker_ip = attacker_ip
    
    def generate_normal(self, count: int = 5) -> list:
        """Generate normal HTTP/HTTPS browsing traffic."""
        packets = []
        for _ in range(count):
            # Normal web traffic
            sport = random.randint(1024, 65535)
            dport = random.choice([80, 443])
            
            # We generate typical packets for a connection, but just taking 'count' packets.
            # So if count=5 we might just generate 5 separate PA packets, or just simple data.
            # For realism let's mix some handshake & data packets
            flags = random.choice(["S", "SA", "A", "PA"])
            
            # Payload
            payload_size = random.randint(200, 1500) if flags == "PA" else 0
            
            pkt = IP(src=self.attacker_ip, dst=self.target_ip)/TCP(sport=sport, dport=dport, flags=flags)
            if payload_size > 0:
                pkt = pkt/Raw(load=b"X"*payload_size)
            
            packets.append(pkt)
            
        return packets
    
    def generate_dos(self, count: int = 10) -> list:
        """Generate DoS flood packets (SYN flood)."""
        packets = []
        for _ in range(count):
            # Neptune/SYN flood: random spoofed source IPs, same target port
            spoofed_ip = f"{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
            sport = random.randint(1024, 65535)
            dport = 80
            
            pkt = IP(src=spoofed_ip, dst=self.target_ip)/TCP(sport=sport, dport=dport, flags="S")
            packets.append(pkt)
        return packets
    
    def generate_probe(self, count: int = 8) -> list:
        """Generate port scan / reconnaissance packets."""
        packets = []
        # Satan/Port sweep
        for i in range(count):
            sport = random.randint(1024, 65535)
            dport = 1 + i # scan sequential ports or random ports
            
            pkt = IP(src=self.attacker_ip, dst=self.target_ip)/TCP(sport=sport, dport=dport, flags="S")
            packets.append(pkt)
        return packets
    
    def generate_r2l(self, count: int = 5) -> list:
        """Generate Remote-to-Local attack packets."""
        packets = []
        for _ in range(count):
            # Guess password / FTP Brute force
            sport = random.randint(1024, 65535)
            dport = random.choice([21, 22]) # FTP or SSH
            
            # Large payloads to simulate trying lots of passwords or exploiting
            payload = b"USER root\r\nPASS password123\r\n" * 10 
            pkt = IP(src=self.attacker_ip, dst=self.target_ip)/TCP(sport=sport, dport=dport, flags="PA")/Raw(load=payload)
            packets.append(pkt)
        return packets
    
    def generate_u2r(self, count: int = 3) -> list:
        """Generate User-to-Root privilege escalation packets."""
        packets = []
        for _ in range(count):
            # Buffer overflow attempts (e.g. against telnet or specific services)
            sport = random.randint(1024, 65535)
            dport = 23 # Telnet
            
            # NOP sled + shellcode simulation
            payload = b"\x90" * 200 + b"/bin/sh" 
            pkt = IP(src=self.attacker_ip, dst=self.target_ip)/TCP(sport=sport, dport=dport, flags="PA")/Raw(load=payload)
            packets.append(pkt)
        return packets
    
    def generate_mixed_scenario(self, total: int = 20) -> list:
        """Generate a realistic mixed traffic scenario with all attack types."""
        # Mix of normal (50%), DoS (20%), Probe (15%), R2L (10%), U2R (5%)
        normal_count = int(total * 0.5)
        dos_count = int(total * 0.2)
        probe_count = int(total * 0.15)
        r2l_count = int(total * 0.1)
        u2r_count = total - (normal_count + dos_count + probe_count + r2l_count)
        
        scenario = []
        for p in self.generate_normal(normal_count):
            scenario.append((p, 'normal'))
        for p in self.generate_dos(dos_count):
            scenario.append((p, 'dos'))
        for p in self.generate_probe(probe_count):
            scenario.append((p, 'probe'))
        for p in self.generate_r2l(r2l_count):
            scenario.append((p, 'r2l'))
        for p in self.generate_u2r(u2r_count):
            scenario.append((p, 'u2r'))
            
        random.shuffle(scenario)
        return scenario
    
    def get_attack_description(self, attack_type: str) -> str:
        """Return a human-readable description of the attack being simulated."""
        descriptions = {
            'normal': 'Normal Traffic — Typical benign HTTP/HTTPS web browsing and network services.',
            'dos': 'SYN Flood Attack (DoS) — Overwhelming target with half-open TCP connections to exhaust server resources.',
            'probe': 'Port Scan (Probe) — Systematically scanning target ports to discover active services and vulnerabilities.',
            'r2l': 'Brute Force (R2L) — Repeated login attempts to gain unauthorized remote access (e.g., FTP/SSH password guessing).',
            'u2r': 'Buffer Overflow (U2R) — Sending malicious payloads to exploit vulnerabilities and escalate to root privileges.'
        }
        return descriptions.get(attack_type.lower(), "Unknown Attack Type")
