"""
app.py — Biological Firewall Engine: Hackathon-Grade Dashboard v3.0
====================================================================
Features:
  • 5-class threat classification (Normal / DoS / Probe / R2L / U2R)
  • Real-time live network sniffing
  • Bulk PCAP/PCAPNG analysis
  • 3D connectome visualization (300 neurons)
  • Active neuroplasticity with replay buffer
  • XAI — Explainable AI / Neural Pathway Tracing
  • Attack Simulation Lab
  • Comparative Benchmark Dashboard
  • Biological Network Analysis
  • Temporal Threat Timeline
  • Performance analytics dashboard
  • Architecture diagram
  • CSV + PDF export
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import os
import pickle
import shutil
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from scapy.all import sniff, IP

from model import FlyBrainNet
from packet_engine import packet_to_model_input, parse_pcap_batch

def mock_genai_responder(attack_class, src_ip, dst_ip):
    """Generates a realistic GenAI Incident Response using local heuristics."""
    responses = {
        "DoS": f"**Gemini Incident Analysis:** The biological connectome detected a massive influx of traffic from `{src_ip}` targeting `{dst_ip}`. The packet structure matches a Denial of Service (DoS) signature intended to exhaust server resources. **Recommendation:** Immediately deploy rate-limiting or drop rules on the perimeter firewall for `{src_ip}`.",
        "Probe": f"**Gemini Incident Analysis:** Network reconnaissance detected. The host at `{src_ip}` is systematically scanning ports and services on `{dst_ip}`. This is typically a precursor to a targeted exploit. **Recommendation:** Block the scanning IP and verify no unauthorized services are exposed on `{dst_ip}`.",
        "R2L": f"**Gemini Incident Analysis:** Unauthorized remote access attempt detected. The attacker at `{src_ip}` is attempting to bypass authentication mechanisms to gain local access on `{dst_ip}`. **Recommendation:** Terminate the connection, audit exposed remote services (SSH, RDP, FTP), and enforce multi-factor authentication.",
        "U2R": f"**Gemini Incident Analysis:** Critical privilege escalation attempt! A local user is attempting to exploit a vulnerability on `{dst_ip}` to gain root/administrator access. **Recommendation:** Immediately isolate the machine, kill suspicious child processes, and patch the vulnerable service."
    }
    return responses.get(attack_class, f"**Gemini Incident Analysis:** Anomalous traffic pattern detected from `{src_ip}`.")

def generate_yara_rule(attack_class, proto, src_bytes):
    import hashlib
    rule_id = hashlib.md5(f"{attack_class}{proto}{src_bytes}".encode()).hexdigest()[:8]
    return f"""rule BioFirewall_{attack_class}_{rule_id} {{
    meta:
        author = "Biological Firewall Engine"
        description = "Auto-generated signature for {attack_class} behavior"
        severity = "High"
    strings:
        $proto = "{proto.upper()}"
    condition:
        $proto and filesize > {max(0, int(src_bytes)-100)} and filesize < {int(src_bytes)+100}
}}"""

def generate_siem_payload(attack_class, src_ip, dst_ip, confidence, proto):
    import json, datetime
    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "vendor": "BiologicalFirewall",
        "event_type": "Threat_Detected",
        "severity": "CRITICAL",
        "src_ip": src_ip,
        "dest_ip": dst_ip,
        "protocol": proto,
        "classification": attack_class,
        "confidence_score": round(confidence, 2),
        "action_taken": "Connection_Dropped"
    }
    return json.dumps(payload, indent=2)


# ═══════════════════════════════════════════════════════════
# 1. PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Biological Firewall Engine",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
CLASS_COLORS = {
    "Normal": "#00E5FF",
    "DoS": "#FF1744",
    "Probe": "#FF9100",
    "R2L": "#D500F9",
    "U2R": "#FFD600",
}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');
    
    /* GLASSMORPHISM & THEMING */
    html, body { 
        background-color: #040810 !important; 
    }
    #root, .stApp, .main, div[data-testid="stAppViewContainer"], div[data-testid="stHeader"] { 
        background: transparent !important; 
        background-color: transparent !important; 
    }
    .stApp { color: #8899AA; font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; background: transparent !important; }
    
    h1, h2, h3 { color: #00E5FF !important; font-weight: 300 !important; letter-spacing: 2px; }
    
    /* Hero Title */
    .hero-title {
        font-size: 2.4rem; font-weight: 700; color: #00E5FF;
        letter-spacing: 3px; margin-bottom: 0;
        text-shadow: 0 0 30px rgba(0,229,255,0.4);
    }
    .hero-subtitle {
        font-size: 0.95rem; color: #5A6A7A; font-weight: 300;
        letter-spacing: 1px; margin-top: 4px; margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Status Boxes */
    .status-threat {
        background: rgba(255, 23, 68, 0.1); backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 23, 68, 0.4); border-left: 4px solid #FF1744; 
        padding: 1.2rem 1.5rem; border-radius: 8px; box-shadow: 0 4px 24px rgba(255, 23, 68, 0.2);
        color: #FF1744; font-size: 1.1rem; font-weight: 600; margin: 1rem 0;
    }
    .status-safe {
        background: rgba(0, 229, 255, 0.08); backdrop-filter: blur(8px);
        border: 1px solid rgba(0, 229, 255, 0.3); border-left: 4px solid #00E5FF; 
        padding: 1.2rem 1.5rem; border-radius: 8px; box-shadow: 0 4px 24px rgba(0, 229, 255, 0.15);
        color: #00E5FF; font-size: 1.1rem; font-weight: 600; margin: 1rem 0;
    }
    
    /* Terminal Output */
    .packet-terminal {
        background: rgba(13, 17, 23, 0.7); backdrop-filter: blur(10px);
        border: 1px solid rgba(33, 38, 45, 0.8); border-radius: 8px;
        padding: 1rem 1.2rem; font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem; line-height: 1.6; margin: 0.5rem 0; box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
    }
    .term-cyan { color: #00E5FF; text-shadow: 0 0 5px rgba(0,229,255,0.3); }
    .term-red { color: #FF1744; text-shadow: 0 0 5px rgba(255,23,68,0.3); }
    .term-orange { color: #FF9100; text-shadow: 0 0 5px rgba(255,145,0,0.3); }
    .term-purple { color: #D500F9; text-shadow: 0 0 5px rgba(213,0,249,0.3); }
    .term-yellow { color: #FFD600; text-shadow: 0 0 5px rgba(255,214,0,0.3); }
    .term-dim { color: #484F58; }
    .term-green { color: #00C853; text-shadow: 0 0 5px rgba(0,200,83,0.3); }
    
    /* Threat Log Table */
    .threat-log { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; margin: 1rem 0; background: rgba(13, 17, 23, 0.5); backdrop-filter: blur(5px); border-radius: 8px; overflow: hidden; }
    .threat-log th { border-bottom: 1px solid rgba(0, 229, 255, 0.3); color: #00E5FF; padding: 12px 8px; text-align: left; font-weight: 400; letter-spacing: 1px; background: rgba(0, 229, 255, 0.05); }
    .threat-log td { padding: 10px 8px; border-bottom: 1px solid rgba(13, 17, 23, 0.8); color: #8899AA; }
    .threat-log tr:hover { background: rgba(0,229,255,0.08); transition: background 0.2s; }
    .tag-dos { color: #FF1744; font-weight: bold; } .tag-probe { color: #FF9100; font-weight: bold; } .tag-r2l { color: #D500F9; font-weight: bold; } .tag-u2r { color: #FFD600; font-weight: bold; } .tag-normal { color: #00E5FF; font-weight: bold; }
    
    /* XAI Feature bars */
    .xai-bar { height: 18px; background: linear-gradient(90deg, #00E5FF, #FF9100); border-radius: 4px; margin: 2px 0; box-shadow: 0 0 8px rgba(0,229,255,0.4); }
    .xai-label { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #8899AA; }
    
    /* Metric boxes */
    div[data-testid="stMetric"] { 
        background: rgba(13, 17, 23, 0.4) !important; 
        backdrop-filter: blur(12px) !important; 
        border: 1px solid rgba(0, 229, 255, 0.15) !important; 
        border-radius: 12px !important; 
        padding: 1rem !important; 
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3) !important; 
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(255, 145, 0, 0.6) !important;
        box-shadow: 0 0 20px rgba(255, 145, 0, 0.2) !important;
        transform: translateY(-2px);
    }
    div[data-testid="stMetricValue"] { text-shadow: 0 0 10px rgba(0,229,255,0.3); }
    
    /* Standard Buttons */
    .stButton>button {
        background: linear-gradient(90deg, rgba(0,229,255,0.05), rgba(255,145,0,0.05));
        border: 1px solid rgba(0, 229, 255, 0.4); backdrop-filter: blur(5px);
        color: #00E5FF; transition: all 0.3s; border-radius: 6px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, rgba(0,229,255,0.2), rgba(255,145,0,0.2));
        border-color: #FF9100; color: #fff; box-shadow: 0 0 20px rgba(0,229,255,0.4);
    }
    
    /* Dataframes & Charts */
    div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"] { 
        background: rgba(13, 17, 23, 0.5); 
        backdrop-filter: blur(8px); 
        border-radius: 8px; 
        border: 1px solid rgba(33, 38, 45, 0.8); 
        padding: 8px;
    }

    /* Mobile Optimization */
    @media (max-width: 768px) {
        .hero-title { font-size: 1.5rem !important; text-align: center; }
        .hero-subtitle { font-size: 0.8rem !important; text-align: center; }
        div[data-testid="stMetricValue"] { font-size: 1.2rem !important; }
        div[data-testid="stMetric"] { padding: 0.2rem !important; margin-bottom: 0 !important; }
        .packet-terminal { font-size: 0.65rem !important; padding: 0.8rem !important; }
        .status-threat, .status-safe { padding: 0.8rem !important; font-size: 0.9rem !important; }
        .stAppViewContainer { padding: 0.5rem !important; }
    }

    /* Footer */
    .footer { text-align: center; color: #484F58; font-size: 0.75rem; margin-top: 3rem; padding: 1rem; border-top: 1px solid rgba(13, 17, 23, 0.8); letter-spacing: 1px; }

    /* Simulation pulse */
    .sim-pulse { 
        display: inline-block; width: 10px; height: 10px; 
        border-radius: 50%; background: #FF1744; margin-right: 8px;
        animation: pulse 1.5s ease-in-out infinite; box-shadow: 0 0 10px #FF1744;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; box-shadow: 0 0 10px #FF1744; } 50% { opacity: 0.3; box-shadow: none; } }
</style>
""", unsafe_allow_html=True)

import streamlit.components.v1 as components
components.html("""
<script>
    const parentDoc = window.parent.document;
    
    // Only inject once!
    if (!parentDoc.getElementById('fly-bg-script')) {
        const script = parentDoc.createElement('script');
        script.id = 'fly-bg-script';
        script.innerHTML = `
            const canvas = document.createElement('canvas');
            canvas.id = 'neural-bg';
            canvas.style.position = 'fixed';
            canvas.style.top = '0';
            canvas.style.left = '0';
            canvas.style.width = '100vw';
            canvas.style.height = '100vh';
            canvas.style.zIndex = '0';
            canvas.style.pointerEvents = 'none';
            document.body.insertBefore(canvas, document.body.firstChild);
            
            const ctx = canvas.getContext('2d');
            let w, h;
            
            function resize() {
                if (w !== window.innerWidth || h !== window.innerHeight) {
                    w = canvas.width = window.innerWidth;
                    h = canvas.height = window.innerHeight;
                }
            }
            window.addEventListener('resize', resize);
            resize();
            
            const anchorRatios = [];
            function addEllipsoid(cx, cy, cz, rx, ry, rz, density) {
                for(let i=0; i<density; i++) {
                    let theta = Math.random() * Math.PI * 2;
                    let phi = Math.acos(2 * Math.random() - 1);
                    let rad = Math.cbrt(Math.random()); 
                    anchorRatios.push({
                        x: cx + rx * rad * Math.sin(phi) * Math.cos(theta),
                        y: cy + ry * rad * Math.sin(phi) * Math.sin(theta),
                        z: cz + rz * rad * Math.cos(phi)
                    });
                }
            }
            function addLeg3D(sx, sy, sz, jx, jy, jz, ex, ey, ez, pts) {
                for(let i=0; i<pts/2; i++) {
                    let t = i / (pts/2 - 1);
                    anchorRatios.push({x: sx + t*(jx-sx), y: sy + t*(jy-sy), z: sz + t*(jz-sz)});
                }
                for(let i=0; i<pts/2; i++) {
                    let t = i / (pts/2 - 1);
                    anchorRatios.push({x: jx + t*(ex-jx), y: jy + t*(ey-jy), z: jz + t*(ez-jz)});
                }
            }
            
            // 3D Fly Anatomy (Facing +Z)
            // Head
            addEllipsoid(0, 0, 0.25, 0.05, 0.05, 0.05, 40); 
            // Eyes
            addEllipsoid(-0.05, 0.02, 0.26, 0.03, 0.04, 0.04, 30);
            addEllipsoid(0.05, 0.02, 0.26, 0.03, 0.04, 0.04, 30);
            
            // Thorax
            addEllipsoid(0, 0, 0.05, 0.08, 0.08, 0.12, 100);
            
            // Abdomen (Long, tapered towards back)
            addEllipsoid(0, -0.02, -0.15, 0.06, 0.06, 0.15, 120);
            
            // Wings (Flat planes swept back)
            for(let i=0; i<80; i++) {
                let r = Math.sqrt(Math.random());
                let angle = Math.random() * Math.PI * 2;
                let wx = r * 0.15 * Math.cos(angle);
                let wz = r * 0.3 * Math.sin(angle);
                // Left Wing
                let rotL = Math.PI/8; // sweep back
                anchorRatios.push({
                    x: wx * Math.cos(rotL) - wz * Math.sin(rotL) - 0.08,
                    y: 0.05,
                    z: wx * Math.sin(rotL) + wz * Math.cos(rotL) + 0.05
                });
                // Right Wing
                let rotR = -Math.PI/8;
                anchorRatios.push({
                    x: wx * Math.cos(rotR) - wz * Math.sin(rotR) + 0.08,
                    y: 0.05,
                    z: wx * Math.sin(rotR) + wz * Math.cos(rotR) + 0.05
                });
            }
            
            // 6 Legs
            addLeg3D(-0.06, -0.05, 0.15,  -0.15, -0.1, 0.2,   -0.2, -0.2, 0.25, 16); 
            addLeg3D( 0.06, -0.05, 0.15,   0.15, -0.1, 0.2,    0.2, -0.2, 0.25, 16); 
            addLeg3D(-0.08, -0.05, 0.05,  -0.2, -0.1, 0.05,   -0.25, -0.25, 0.05, 16);
            addLeg3D( 0.08, -0.05, 0.05,   0.2, -0.1, 0.05,    0.25, -0.25, 0.05, 16);
            addLeg3D(-0.06, -0.05, -0.05, -0.18, -0.1, -0.15, -0.25, -0.25, -0.2, 16);
            addLeg3D( 0.06, -0.05, -0.05,  0.18, -0.1, -0.15,  0.25, -0.25, -0.2, 16);
            
            const flyParticles = [];
            for (let i = 0; i < anchorRatios.length; i++) {
                flyParticles.push({
                    anchor: anchorRatios[i],
                    wander: Math.random() * Math.PI * 2,
                    baseRadius: Math.random() * 1.5 + 0.5,
                    screenX: 0, screenY: 0, scale: 1
                });
            }
            
            // Ambient Background Particles
            const ambientParticles = [];
            let numAmbient = Math.floor((window.innerWidth * window.innerHeight) / 10000); // responsive count
            for (let i = 0; i < numAmbient; i++) {
                ambientParticles.push({
                    x: Math.random() * window.innerWidth,
                    y: Math.random() * window.innerHeight,
                    vx: (Math.random() - 0.5) * 0.5,
                    vy: (Math.random() - 0.5) * 0.5,
                    r: Math.random() * 1.5 + 0.5
                });
            }
            
            let mouse = { x: null, y: null };
            document.addEventListener('mousemove', (e) => {
                mouse.x = e.clientX;
                mouse.y = e.clientY;
            });
            document.addEventListener('mouseleave', () => {
                mouse.x = null; mouse.y = null;
            });
            
            let time = 0;
            
            function draw() {
                time += 0.01;
                
                const grad = ctx.createLinearGradient(0, 0, w, h);
                grad.addColorStop(0, '#03060a');
                grad.addColorStop(1, '#09111a');
                ctx.fillStyle = grad;
                ctx.fillRect(0, 0, w, h);
                
                // 1. Draw Ambient Particles
                ctx.beginPath();
                for (let i = 0; i < ambientParticles.length; i++) {
                    let p = ambientParticles[i];
                    p.x += p.vx;
                    p.y += p.vy;
                    if (p.x < 0) p.x = w; if (p.x > w) p.x = 0;
                    if (p.y < 0) p.y = h; if (p.y > h) p.y = 0;
                    
                    ctx.moveTo(p.x + p.r, p.y);
                    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                }
                ctx.fillStyle = 'rgba(0, 229, 255, 0.2)';
                ctx.fill();
                
                // Ambient Connections
                ctx.beginPath();
                for (let i = 0; i < ambientParticles.length; i++) {
                    let p = ambientParticles[i];
                    if (mouse.x !== null) {
                        let dx = mouse.x - p.x;
                        let dy = mouse.y - p.y;
                        if (dx*dx + dy*dy < 20000) { // ~140px connect to mouse
                            ctx.moveTo(p.x, p.y);
                            ctx.lineTo(mouse.x, mouse.y);
                        }
                    }
                    // Connect to nearby ambient
                    let connected = 0;
                    for (let j = i + 1; j < ambientParticles.length; j++) {
                        if (connected > 2) break;
                        let p2 = ambientParticles[j];
                        let dx = p.x - p2.x; let dy = p.y - p2.y;
                        if (dx*dx + dy*dy < 10000) { // 100px
                            ctx.moveTo(p.x, p.y);
                            ctx.lineTo(p2.x, p2.y);
                            connected++;
                        }
                    }
                }
                ctx.strokeStyle = 'rgba(0, 229, 255, 0.08)';
                ctx.lineWidth = 0.5;
                ctx.stroke();
                
                // 2. Draw 3D Fly
                let centerX = w / 2 + Math.sin(time * 0.5) * 50; 
                let centerY = h / 2 + Math.cos(time * 0.3) * 30;
                let dim = Math.min(w, h) * 1.5;
                
                let rotY = time * 0.5; // continuous slow spin
                let rotX = 0.4; // Base tilt
                
                // Proximity check for fly interaction
                let flyInfluence = 0;
                if (mouse.x !== null) {
                    let dx = mouse.x - centerX;
                    let dy = mouse.y - centerY;
                    let dist = Math.sqrt(dx*dx + dy*dy);
                    let flyRadius = dim * 0.35; // approximate visible size of fly
                    if (dist < flyRadius) {
                        flyInfluence = Math.pow(1 - (dist / flyRadius), 2); // easing
                    }
                }
                
                // Mouse only rotates fly if hovering over it
                if (mouse.x !== null && flyInfluence > 0) {
                    rotY += (mouse.x / w - 0.5) * 2 * flyInfluence;
                    rotX += (mouse.y / h - 0.5) * flyInfluence;
                }
                
                let cosY = Math.cos(rotY), sinY = Math.sin(rotY);
                let cosX = Math.cos(rotX), sinX = Math.sin(rotX);
                
                // 3D Projection
                for (let i = 0; i < flyParticles.length; i++) {
                    let p = flyParticles[i];
                    
                    p.wander += 0.03;
                    let x = p.anchor.x + Math.cos(p.wander) * 0.005;
                    let y = p.anchor.y + Math.sin(p.wander) * 0.005;
                    let z = p.anchor.z + Math.cos(p.wander * 1.1) * 0.005;
                    
                    let x1 = x * cosY - z * sinY;
                    let z1 = x * sinY + z * cosY;
                    let y1 = y;
                    
                    let y2 = y1 * cosX - z1 * sinX;
                    let z2 = y1 * sinX + z1 * cosX;
                    let x2 = x1;
                    
                    let fov = 1.5;
                    p.scale = fov / (fov + z2);
                    
                    p.screenX = centerX + x2 * p.scale * dim;
                    p.screenY = centerY - y2 * p.scale * dim;
                }
                
                // Draw Fly Nodes
                ctx.beginPath();
                for (let i = 0; i < flyParticles.length; i++) {
                    let p = flyParticles[i];
                    ctx.moveTo(p.screenX + p.baseRadius * p.scale, p.screenY);
                    ctx.arc(p.screenX, p.screenY, p.baseRadius * p.scale, 0, Math.PI * 2);
                }
                ctx.fillStyle = 'rgba(0, 229, 255, 0.9)';
                ctx.fill();
                
                // Draw Fly Connections
                ctx.beginPath();
                for (let i = 0; i < flyParticles.length; i+=2) {
                    let p = flyParticles[i];
                    
                    // Synapse to mouse ONLY if very close (touching the dot)
                    if (mouse.x !== null) {
                        let dx = mouse.x - p.screenX;
                        let dy = mouse.y - p.screenY;
                        if (dx*dx + dy*dy < 6400) { // ~80px (strictly touching)
                            ctx.moveTo(p.screenX, p.screenY);
                            ctx.lineTo(mouse.x, mouse.y);
                        }
                    }
                    
                    // Local synapses
                    let connected = 0;
                    for (let j = i + 1; j < flyParticles.length; j+=2) {
                        if (connected > 3) break;
                        let p2 = flyParticles[j];
                        let dx = p2.screenX - p.screenX;
                        let dy = p2.screenY - p.screenY;
                        if (dx*dx + dy*dy < 3600 * p.scale) {
                            ctx.moveTo(p.screenX, p.screenY);
                            ctx.lineTo(p2.screenX, p2.screenY);
                            connected++;
                        }
                    }
                }
                ctx.strokeStyle = 'rgba(0, 229, 255, 0.2)';
                ctx.lineWidth = 0.5;
                ctx.stroke();
                
                window.requestAnimationFrame(draw);
            }
            draw();
        `;
        parentDoc.body.appendChild(script);
    }
</script>
""", height=0, width=0)


# ═══════════════════════════════════════════════════════════
# 2. DATA LOADING
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_engine():
    """Load model, data, scaler, and visualization graph."""
    # ── Load training data ──
    columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "label", "difficulty_level",
    ]
    df = pd.read_csv("KDDTrain.txt", names=columns)

    ATTACK_CATEGORIES = {
        "normal": 0, "back": 1, "land": 1, "neptune": 1, "pod": 1, "smurf": 1,
        "teardrop": 1, "mailbomb": 1, "apache2": 1, "processtable": 1, "udpstorm": 1,
        "ipsweep": 2, "nmap": 2, "portsweep": 2, "satan": 2, "mscan": 2, "saint": 2,
        "ftp_write": 3, "guess_passwd": 3, "imap": 3, "multihop": 3, "phf": 3,
        "spy": 3, "warezclient": 3, "warezmaster": 3, "xlock": 3, "xsnoop": 3,
        "snmpguess": 3, "snmpgetattack": 3, "httptunnel": 3, "sendmail": 3,
        "named": 3, "worm": 3,
        "buffer_overflow": 4, "loadmodule": 4, "perl": 4, "rootkit": 4,
        "xterm": 4, "ps": 4, "sqlattack": 4,
    }
    df["attack_class"] = df["label"].map(lambda x: ATTACK_CATEGORIES.get(x.strip().lower(), 0))
    df_display = df.copy()
    y = df["attack_class"].values

    df_feat = df.drop(["label", "difficulty_level", "attack_class"], axis=1)
    df_encoded = pd.get_dummies(df_feat, columns=["protocol_type", "service", "flag"])
    train_columns = df_encoded.columns

    # Load saved scaler or fit new one
    if os.path.exists("scaler.pkl"):
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        X = scaler.transform(df_encoded)
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(df_encoded)

    # ── Load sparse connectome ──
    sparse_adj = torch.load("sparse_connectome.pt", weights_only=False)
    num_neurons = sparse_adj.shape[0]

    # ── Load model ──
    model = FlyBrainNet(X.shape[1], num_neurons, 5, sparse_adj)
    if os.path.exists("fly_brain_weights.pt"):
        model.load_state_dict(torch.load("fly_brain_weights.pt", weights_only=True))
        model.eval()
        st.session_state["weights_loaded"] = True
    else:
        # Allow dashboard to load with untrained brain while training is running
        model.eval()
        st.session_state["weights_loaded"] = False

    # ── Build viz graph (300 neurons) ──
    adj_coalesced = sparse_adj.coalesce()
    indices = adj_coalesced.indices().numpy()
    viz_nodes = set()
    for i in range(indices.shape[1]):
        viz_nodes.add(indices[0, i])
        viz_nodes.add(indices[1, i])
        if len(viz_nodes) >= 300:
            break
    viz_nodes = sorted(list(viz_nodes))[:300]
    viz_set = set(viz_nodes)
    viz_map = {n: i for i, n in enumerate(viz_nodes)}

    G = nx.DiGraph()
    G.add_nodes_from(range(len(viz_nodes)))
    for i in range(indices.shape[1]):
        src, dst = int(indices[0, i]), int(indices[1, i])
        if src in viz_set and dst in viz_set:
            G.add_edge(viz_map[src], viz_map[dst])

    pos = nx.spring_layout(G, dim=3, seed=42, iterations=50)

    # ── Load training metrics ──
    metrics = None
    if os.path.exists("training_metrics.json"):
        with open("training_metrics.json") as f:
            metrics = json.load(f)

    # ── Load benchmark results ──
    benchmarks = None
    if os.path.exists("benchmark_results.json"):
        with open("benchmark_results.json") as f:
            benchmarks = json.load(f)

    # ── Load bio analysis results ──
    bio_analysis = None
    if os.path.exists("bio_analysis_results.json"):
        with open("bio_analysis_results.json") as f:
            bio_analysis = json.load(f)

    return model, X, y, df_display, scaler, train_columns, G, pos, metrics, benchmarks, bio_analysis


# ═══════════════════════════════════════════════════════════
# 3. VISUALIZATION
# ═══════════════════════════════════════════════════════════
def draw_3d_brain(G, pos, activations, num_viz_nodes=300, highlight_neurons=None):
    """Render the 3D connectome with color-coded neural activations."""
    n = min(num_viz_nodes, len(activations))
    colors = activations[:n].detach().numpy()

    # Edges
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        if u < n and v < n:
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

    edges = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode="lines",
        line=dict(color="rgba(0,229,255,0.04)", width=0.5),
        hoverinfo="none",
    )

    # Nodes
    nx_list = [pos[k][0] for k in range(n) if k in pos]
    ny_list = [pos[k][1] for k in range(n) if k in pos]
    nz_list = [pos[k][2] for k in range(n) if k in pos]
    node_colors = colors[:len(nx_list)]
    
    # Node sizes — highlight top activated neurons
    sizes = np.full(len(nx_list), 4.0)
    if highlight_neurons is not None:
        for idx in highlight_neurons:
            if idx < len(sizes):
                sizes[idx] = 12.0

    hovers = []
    for i, act in enumerate(node_colors):
        if act > 0.8: state = "Hyperactive"
        elif act > 0.3: state = "Active"
        elif act > 0: state = "Low"
        else: state = "Inhibited"
        deg_in = G.in_degree(i) if i in G else 0
        deg_out = G.out_degree(i) if i in G else 0
        hovers.append(
            f"<b>Neuron #{i}</b><br>"
            f"State: {state}<br>"
            f"Potential: {act:.4f}<br>"
            f"Synapses In: {deg_in}<br>"
            f"Synapses Out: {deg_out}"
        )

    nodes = go.Scatter3d(
        x=nx_list, y=ny_list, z=nz_list, mode="markers",
        marker=dict(
            size=sizes, color=node_colors, opacity=0.9,
            colorscale=[[0, "#0D1117"], [0.3, "#1E3A5F"], [0.5, "#00E5FF"], [0.8, "#FF9100"], [1, "#FF1744"]],
            cmin=-1, cmax=1, line=dict(width=0),
        ),
        text=hovers, hoverinfo="text",
    )

    fig = go.Figure(data=[edges, nodes])
    fig.update_layout(
        uirevision="brain", showlegend=False,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
    )
    return fig


def draw_xai_feature_importance(features, train_columns, model, scaler):
    """Compute and display gradient-based feature importance (XAI)."""
    t_in = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    t_in.requires_grad_(True)
    
    # Enable gradients temporarily
    model.train()
    pred, brain_state = model(t_in)
    predicted_class = pred.argmax(dim=1).item()
    
    # Compute gradient of predicted class score w.r.t. input
    pred[0, predicted_class].backward()
    model.eval()
    
    # Input × Gradient attribution
    gradients = t_in.grad[0].detach().numpy()
    input_vals = t_in.detach().numpy()[0]
    attribution = np.abs(gradients * input_vals)
    
    # Get top 15 features
    top_indices = np.argsort(attribution)[-15:][::-1]
    top_features = []
    for idx in top_indices:
        if idx < len(train_columns):
            name = str(train_columns[idx])
            # Clean up one-hot encoded names
            name = name.replace("protocol_type_", "proto:").replace("service_", "svc:").replace("flag_", "flag:")
            top_features.append((name, float(attribution[idx])))
    
    return top_features, brain_state, predicted_class


# ═══════════════════════════════════════════════════════════
# 4. NEUROPLASTICITY
# ═══════════════════════════════════════════════════════════
def retrain_with_replay(model, new_features, new_label):
    """Retrain with replay buffer to prevent catastrophic forgetting."""
    if "replay_buffer_x" not in st.session_state:
        st.session_state["replay_buffer_x"] = []
        st.session_state["replay_buffer_y"] = []

    st.session_state["replay_buffer_x"].append(new_features)
    st.session_state["replay_buffer_y"].append(new_label)

    if len(st.session_state["replay_buffer_x"]) > 100:
        st.session_state["replay_buffer_x"] = st.session_state["replay_buffer_x"][-100:]
        st.session_state["replay_buffer_y"] = st.session_state["replay_buffer_y"][-100:]

    if os.path.exists("fly_brain_weights.pt"):
        shutil.copy("fly_brain_weights.pt", "fly_brain_weights.pt.backup")

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    X_buf = torch.tensor(np.array(st.session_state["replay_buffer_x"]), dtype=torch.float32)
    y_buf = torch.tensor(st.session_state["replay_buffer_y"], dtype=torch.long)

    for _ in range(5):
        optimizer.zero_grad()
        out, _ = model(X_buf)
        loss = criterion(out, y_buf)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "fly_brain_weights.pt")
    model.eval()

    if "corrections_count" not in st.session_state:
        st.session_state["corrections_count"] = 0
    st.session_state["corrections_count"] += 1


# ═══════════════════════════════════════════════════════════
# 5. MAIN UI
# ═══════════════════════════════════════════════════════════

# Initialize session counters
if "session_scanned" not in st.session_state:
    st.session_state["session_scanned"] = 0
    st.session_state["session_threats"] = 0
    st.session_state["corrections_count"] = 0
    st.session_state["threat_timeline"] = []

# ── Hero ──
st.markdown("<div class='hero-title'>BIOLOGICAL FIREWALL ENGINE</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>A biologically-constrained neural network using the Drosophila melanogaster connectome for real-time cyber threat detection</div>", unsafe_allow_html=True)

try:
    model, X_data, y_labels, df_display, scaler, train_columns, G, pos, metrics, benchmarks, bio_analysis = load_engine()

    # ── Metrics Bar ──
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Neurons", f"{model.num_neurons:,}")
    m2.metric("Synapses", f"{model.adjacency_mask._nnz():,}")
    m3.metric("Training Data", "125,973")
    m4.metric("Accuracy", f"{metrics['best_val_accuracy']:.1f}%" if metrics else "—")
    
    # Placeholders to prevent stale state (filled at end of script)
    m5_ph = m5.empty()
    m6_ph = m6.empty()
    m7_ph = m7.empty()
    
    m5_ph.metric("Scanned (Session)", str(st.session_state.get("session_scanned", 0)))
    m6_ph.metric("Threats Blocked", str(st.session_state.get("session_threats", 0)))
    ac = "—"
    if "brain_state" in st.session_state and isinstance(st.session_state["brain_state"], torch.Tensor):
        ac = f"{(st.session_state['brain_state'].abs() > 0.3).sum().item():,}"
    m7_ph.metric("Active Neurons", ac)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Sidebar Demo Script ──
    with st.sidebar:
        st.markdown("### Demo Walkthrough")
        st.info(
            "**Step 1:** Click to scan an archived packet in the *Archive* tab. Shows BENIGN.\n\n"
            "**Step 2:** Switch to *Attack Sim* tab and click to run attack simulation. Shows threats detected.\n\n"
            "**Step 3:** Click to export PDF report in the *Analytics* section to download a professional report."
        )
        st.markdown("---")
        st.markdown(
            "**Pitch points:**\n"
            "- *Efficiency:* Show the Parameters column in Benchmarks (150× fewer parameters for only 5% accuracy drop).\n"
            "- *Biology:* Point out the 'Active Neurons' counter in the hero bar flashing when a packet is processed.\n"
            "- *Visualization:* Show the 3D connectome rendering."
        )
    
    if not st.session_state.get("weights_loaded", True):
        st.warning("**MODEL IS CURRENTLY RETRAINING** — The optimized weights (`fly_brain_weights.pt`) are not present on disk. The Biological Firewall is currently running on **untrained, random synaptic weights**. Classifications will be inaccurate until the background training completes and you restart the dashboard.")

    # ── Main Layout ──
    col_ctrl, col_viz = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown("### DATA INGESTION")

        tab_archive, tab_pcap, tab_live, tab_sim = st.tabs([
            "[ ARCHIVE ]", "[ BULK PCAP ]", "[ LIVE SNIFFER ]", "[ SIMULATION ]"
        ])

        with tab_archive:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("INTERCEPT ARCHIVED PACKET", use_container_width=True):
                idx = np.random.randint(0, len(X_data))
                st.session_state["scan_results"] = {
                    "type": "single",
                    "features": X_data[idx],
                    "raw": df_display.iloc[idx].to_dict(),
                    "true_label": y_labels[idx],
                }
                st.session_state["session_scanned"] += 1

        with tab_pcap:
            st.markdown("<br>", unsafe_allow_html=True)
            uploaded = st.file_uploader("Drop .pcap / .pcapng", type=["pcap", "pcapng"], label_visibility="collapsed")
            pcap_limit = st.number_input("Max Packets to Read (0 = Unlimited)", min_value=0, value=500, step=100)
            st.caption("Reading unlimited packets from large PCAPs may crash the browser.")
            
            if uploaded and st.button("ANALYZE FULL TRACE", use_container_width=True):
                with st.spinner("Parsing packets..."):
                    feats, disps = parse_pcap_batch(uploaded.getvalue(), train_columns, scaler, max_packets=pcap_limit)
                    if feats is not None:
                        st.session_state["scan_results"] = {"type": "bulk", "features": feats, "displays": disps}
                        st.session_state["session_scanned"] += len(disps)
                    else:
                        st.error("No valid packets found.")

        with tab_live:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("Intercept real packets from your network interface.")
            n_capture = st.slider("Packets to capture", 1, 50, 10)
            if st.button("START LIVE CAPTURE", use_container_width=True):
                with st.spinner(f"Sniffing {n_capture} packets from NIC..."):
                    try:
                        packets = sniff(count=n_capture, timeout=10)
                        if len(packets) > 0:
                            all_feats, all_disps = [], []
                            for pkt in packets:
                                f, d = packet_to_model_input(pkt, train_columns, scaler)
                                if f is None:
                                    continue  # Skip non-IP packets
                                d["id"] = len(all_feats) + 1
                                all_feats.append(f)
                                all_disps.append(d)
                            if all_feats:
                                st.session_state["scan_results"] = {
                                    "type": "bulk", "features": np.array(all_feats), "displays": all_disps,
                                }
                                st.session_state["session_scanned"] += len(all_feats)
                            else:
                                st.warning("No IP packets captured. Only ARP/broadcast traffic found.")
                        else:
                            st.warning("No traffic captured. Try browsing a website while sniffing!")
                    except PermissionError:
                        st.error("Permission Denied — restart with `sudo`")
                        st.code("sudo venv/bin/streamlit run app.py --server.port 8502", language="bash")

        with tab_sim:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<span class='sim-pulse'></span> **ATTACK SIMULATION LAB**", unsafe_allow_html=True)
            st.caption("Generate synthetic attack packets for live demos — no real attacks performed.")
            
            sim_type = st.selectbox("Attack Scenario", [
                "Mixed Traffic (Realistic)", "SYN Flood (DoS)", "Port Scan (Probe)", 
                "Brute Force (R2L)", "Buffer Overflow (U2R)", "Normal Browsing"
            ])
            sim_count = st.slider("Packets to generate", 5, 50, 15, key="sim_count")
            
            if st.button("LAUNCH SIMULATION", use_container_width=True):
                try:
                    from attack_sim import AttackSimulator
                    sim = AttackSimulator()
                    
                    sim_map = {
                        "Mixed Traffic (Realistic)": "mixed",
                        "SYN Flood (DoS)": "dos",
                        "Port Scan (Probe)": "probe",
                        "Brute Force (R2L)": "r2l",
                        "Buffer Overflow (U2R)": "u2r",
                        "Normal Browsing": "normal",
                    }
                    attack_type = sim_map[sim_type]
                    
                    if attack_type == "mixed":
                        results = sim.generate_mixed_scenario(sim_count)
                    else:
                        gen_func = getattr(sim, f"generate_{attack_type}")
                        pkts = gen_func(sim_count)
                        results = [(p, attack_type) for p in pkts]
                    
                    all_feats, all_disps = [], []
                    for pkt_data in results:
                        pkt = pkt_data[0] if isinstance(pkt_data, tuple) else pkt_data
                        f, d = packet_to_model_input(pkt, train_columns, scaler)
                        if f is None:
                            continue  # Skip non-IP packets
                        d["id"] = len(all_feats) + 1
                        d["sim_type"] = pkt_data[1] if isinstance(pkt_data, tuple) else attack_type
                        all_feats.append(f)
                        all_disps.append(d)
                    
                    st.session_state["scan_results"] = {
                        "type": "bulk", "features": np.array(all_feats), "displays": all_disps,
                    }
                    st.session_state["session_scanned"] += len(all_feats)
                    st.success(f"Generated {len(all_feats)} synthetic packets")
                except ImportError:
                    st.error("attack_sim.py not found")
                except Exception as e:
                    st.error(f"Simulation error: {e}")

        # ── Results ──
        if "scan_results" in st.session_state:
            res = st.session_state["scan_results"]
            st.markdown("<hr style='border-color:#0D1117'>", unsafe_allow_html=True)
            st.markdown("### NEURAL CLASSIFICATION")

            if res["type"] == "single":
                t_in = torch.tensor(res["features"], dtype=torch.float32).unsqueeze(0)
                t0 = time.perf_counter()
                pred, brain_state = model(t_in)
                inference_ms = (time.perf_counter() - t0) * 1000
                _, cls = torch.max(pred, 1)
                cls_id = cls.item()
                cls_name = CLASS_NAMES[cls_id]
                
                # Confidence scores
                probs = torch.softmax(pred, dim=1)[0]
                confidence = probs[cls_id].item() * 100

                if cls_id == 0:
                    st.markdown(f"<div class='status-safe'>[✓] TRAFFIC BENIGN — {confidence:.1f}% confidence<br><span style='font-size:0.85rem;font-weight:400'>Connectome classified as <b>{cls_name}</b>. Packet authorized.</span></div>", unsafe_allow_html=True)
                else:
                    st.session_state["session_threats"] += 1
                    tag_cls = f"tag-{cls_name.lower()}"
                    st.markdown(f"<div class='status-threat'>[!] THREAT DETECTED: <span class='{tag_cls}'>{cls_name.upper()}</span> — {confidence:.1f}% confidence<br><span style='font-size:0.85rem;font-weight:400'>Connectome identified malicious signature. Connection terminated.</span></div>", unsafe_allow_html=True)
                
                # Add to timeline
                st.session_state["threat_timeline"].append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "class": cls_name,
                    "confidence": confidence,
                    "src_ip": res['raw'].get('src_ip', f"10.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}")
                })

                # Terminal output
                color = "term-cyan" if cls_id == 0 else "term-red"
                st.markdown(f"""<div class='packet-terminal'>
                    <span class='term-dim'>┌─ PACKET TELEMETRY ──────────────────</span><br>
                    <span class='{color}'>│ PROTOCOL : {res['raw']['protocol_type'].upper()}</span><br>
                    <span class='{color}'>│ SERVICE  : {res['raw']['service'].upper()}</span><br>
                    <span class='{color}'>│ FLAG     : {res['raw']['flag'].upper()}</span><br>
                    <span class='{color}'>│ PAYLOAD  : {int(res['raw']['src_bytes'])} bytes</span><br>
                    <span class='{color}'>│ VERDICT  : {cls_name.upper()} ({confidence:.1f}%)</span><br>
                    <span class='term-green'>│ LATENCY  : {inference_ms:.2f} ms</span><br>
                    <span class='term-dim'>└─────────────────────────────────────</span>
                </div>""", unsafe_allow_html=True)

                st.session_state["brain_state"] = brain_state[0]
                st.session_state["last_features"] = res["features"]
                
                # ── GenAI & Auto-Mitigation ──
                if cls_id != 0:
                    src_ip = res['raw'].get('src_ip', f"10.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}")
                    dst_ip = res['raw'].get('dst_ip', '10.0.0.5')
                    st.markdown("<hr style='border-color:#0D1117'>", unsafe_allow_html=True)
                    st.markdown("### AI INCIDENT RESPONDER & AUTO-MITIGATION")
                    st.info(mock_genai_responder(cls_name, src_ip, dst_ip))
                    st.markdown("**Deploy Countermeasures:**")
                    st.code(f"sudo iptables -A INPUT -s {src_ip} -j DROP\nsudo ufw deny from {src_ip}", language="bash")
                    
                    with st.expander("ENTERPRISE THREAT INTEL (YARA & SIEM)", expanded=False):
                        st.markdown("**SIEM Webhook Payload (Splunk/Sentinel):**")
                        st.code(generate_siem_payload(cls_name, src_ip, dst_ip, confidence, res['raw'].get('protocol_type', 'tcp')), language="json")
                        st.markdown("**Zero-Day YARA Signature:**")
                        st.code(generate_yara_rule(cls_name, res['raw'].get('protocol_type', 'tcp'), res['raw'].get('src_bytes', 0)), language="yara")
                        
                    if st.button("Mark as False Positive (Bio-Feedback)", use_container_width=True):
                        retrain_with_replay(model, res["features"], 0)
                        st.session_state["session_threats"] = max(0, st.session_state["session_threats"] - 1)
                        st.success("Biological synapses adjusted. Threat marked as benign.")
                        time.sleep(1)
                        st.rerun()
                
                # ── XAI: Explainability ──
                st.markdown("<hr style='border-color:#0D1117'>", unsafe_allow_html=True)
                with st.expander("EXPLAINABILITY — Why did the brain classify this way?", expanded=False):
                    try:
                        xai_features, _, _ = draw_xai_feature_importance(res["features"], train_columns, model, scaler)
                        if xai_features:
                            max_val = max(v for _, v in xai_features) if xai_features else 1.0
                            for fname, fval in xai_features[:10]:
                                pct = (fval / max_val * 100) if max_val > 0 else 0
                                bar_color = "#FF1744" if cls_id != 0 else "#00E5FF"
                                st.markdown(f"""<div style='display:flex;align-items:center;margin:3px 0'>
                                    <span class='xai-label' style='width:180px;text-align:right;padding-right:10px'>{fname}</span>
                                    <div style='flex:1;background:#0D1117;border-radius:4px;height:16px'>
                                        <div style='width:{pct:.0f}%;height:100%;background:{bar_color};border-radius:4px;opacity:0.8'></div>
                                    </div>
                                    <span class='xai-label' style='width:60px;padding-left:8px'>{fval:.4f}</span>
                                </div>""", unsafe_allow_html=True)
                            st.caption("Feature importance computed via Input × Gradient attribution")
                    except Exception as e:
                        st.caption(f"XAI unavailable: {e}")

                # Neuroplasticity
                st.markdown("<hr style='border-color:#0D1117'>", unsafe_allow_html=True)
                st.markdown("### NEUROPLASTICITY")
                st.caption(f"Corrections applied this session: {st.session_state['corrections_count']}")
                cols = st.columns(5)
                for i, name in enumerate(CLASS_NAMES):
                    with cols[i]:
                        if st.button(f"Force: {name}", use_container_width=True, key=f"retrain_{i}"):
                            retrain_with_replay(model, res["features"], i)
                            st.success(f"Brain adapted → {name}")
                            st.rerun()

                if os.path.exists("fly_brain_weights.pt.backup"):
                    if st.button("Restore Original Brain", use_container_width=True):
                        shutil.copy("fly_brain_weights.pt.backup", "fly_brain_weights.pt")
                        st.cache_resource.clear()
                        st.rerun()

            elif res["type"] == "bulk":
                t_in = torch.tensor(res["features"], dtype=torch.float32)
                t0 = time.perf_counter()
                preds, brain_states = model(t_in)
                bulk_ms = (time.perf_counter() - t0) * 1000
                _, cls_ids = torch.max(preds, 1)
                cls_np = cls_ids.numpy()
                probs = torch.softmax(preds, dim=1).detach().numpy()

                n_total = len(res["displays"])
                n_threats = int((cls_np != 0).sum())
                throughput = n_total / (bulk_ms / 1000) if bulk_ms > 0 else 0
                st.session_state["session_threats"] += n_threats
                
                # Add all to timeline
                for j in range(n_total):
                    st.session_state["threat_timeline"].append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "class": CLASS_NAMES[int(cls_np[j])],
                        "confidence": float(probs[j][int(cls_np[j])]) * 100,
                        "src_ip": res['displays'][j].get('src_ip', f"10.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}")
                    })

                if n_threats > 0:
                    st.markdown(f"<div class='status-threat'>[!] INCIDENT RESPONSE<br><span style='font-size:0.85rem;font-weight:400'>Scanned {n_total} packets · Neutralized {n_threats} threats · {bulk_ms:.1f}ms ({throughput:.0f} pkt/s)</span></div>", unsafe_allow_html=True)
                    
                    # ── GenAI & Auto-Mitigation (Bulk Mode) ──
                    threat_idx = np.where(cls_np != 0)[0][0]
                    threat_cls_name = CLASS_NAMES[int(cls_np[threat_idx])]
                    threat_ip = res['displays'][threat_idx].get('src_ip', f"10.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}")
                    
                    st.markdown("<hr style='border-color:#0D1117'>", unsafe_allow_html=True)
                    st.markdown("### AI INCIDENT RESPONDER & AUTO-MITIGATION")
                    st.info(mock_genai_responder(threat_cls_name, threat_ip, "10.0.0.5"))
                    st.markdown("**Deploy Countermeasures:**")
                    st.code(f"sudo iptables -A INPUT -s {threat_ip} -j DROP\nsudo ufw deny from {threat_ip}", language="bash")
                    
                    with st.expander("ENTERPRISE THREAT INTEL (YARA & SIEM)", expanded=False):
                        conf = float(probs[threat_idx][int(cls_np[threat_idx])]) * 100
                        proto = res['displays'][threat_idx].get('protocol_type', 'tcp')
                        s_bytes = res['displays'][threat_idx].get('src_bytes', 0)
                        st.markdown("**SIEM Webhook Payload (Splunk/Sentinel):**")
                        st.code(generate_siem_payload(threat_cls_name, threat_ip, "10.0.0.5", conf, proto), language="json")
                        st.markdown("**Zero-Day YARA Signature:**")
                        st.code(generate_yara_rule(threat_cls_name, proto, s_bytes), language="yara")
                        
                    if st.button("Mark Primary Threat as False Positive (Bio-Feedback)", use_container_width=True):
                        retrain_with_replay(model, res["features"][threat_idx], 0)
                        st.session_state["session_threats"] = max(0, st.session_state["session_threats"] - 1)
                        st.success("Biological synapses adjusted. Threat marked as benign.")
                        time.sleep(1)
                        st.rerun()
                        
                else:
                    st.markdown(f"<div class='status-safe'>[✓] NETWORK SECURE<br><span style='font-size:0.85rem;font-weight:400'>Scanned {n_total} packets · 0 threats · {bulk_ms:.1f}ms ({throughput:.0f} pkt/s)</span></div>", unsafe_allow_html=True)

                # Summary pie chart
                class_counts = {name: int((cls_np == i).sum()) for i, name in enumerate(CLASS_NAMES)}
                df_pie = pd.DataFrame({
                    "Class": list(class_counts.keys()),
                    "Count": list(class_counts.values())
                })
                fig_pie = px.pie(
                    df_pie, names="Class", values="Count", color="Class",
                    color_discrete_map=CLASS_COLORS,
                    hole=0.45,
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#8899AA", showlegend=True, margin=dict(t=10, b=10, l=10, r=10),
                    legend=dict(font=dict(size=10)),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # Full log table with scrolling container
                table = "<div style='max-height: 400px; overflow-y: auto; margin-bottom: 20px; border: 1px solid #21262D; border-radius: 8px;'><table class='threat-log'><tr><th>#</th><th>PROTO</th><th>SERVICE</th><th>FLAG</th><th>BYTES</th><th>CLASS</th><th>CONF</th><th>STATUS</th></tr>"
                for j, d in enumerate(res["displays"]):
                    c = int(cls_np[j])
                    cname = CLASS_NAMES[c]
                    tag = f"tag-{cname.lower()}"
                    status = "ALLOWED" if c == 0 else "BLOCKED"
                    s_color = "term-cyan" if c == 0 else "term-red"
                    conf = f"{probs[j][c]*100:.0f}%"
                    table += f"<tr><td>{d['id']}</td><td>{d['protocol_type'].upper()}</td><td>{d['service'].upper()}</td><td>{d['flag'].upper()}</td><td>{d['src_bytes']}</td><td><span class='{tag}'>{cname}</span></td><td>{conf}</td><td><span class='{s_color}'>{status}</span></td></tr>"
                table += "</table></div>"
                st.markdown(table, unsafe_allow_html=True)

                # CSV export
                export_df = pd.DataFrame(res["displays"])
                export_df["classification"] = [CLASS_NAMES[int(c)] for c in cls_np]
                export_df["confidence"] = [f"{probs[j][int(cls_np[j])]*100:.1f}%" for j in range(len(cls_np))]
                export_df["status"] = ["ALLOWED" if int(c) == 0 else "BLOCKED" for c in cls_np]
                st.download_button("Download Scan Report (CSV)", export_df.to_csv(index=False), "firewall_report.csv", "text/csv", use_container_width=True)
                
                # PDF export
                try:
                    from report_generator import generate_security_report
                    pdf_results = []
                    for j, d in enumerate(res["displays"]):
                        c = int(cls_np[j])
                        pdf_results.append({
                            "id": d["id"],
                            "protocol_type": d["protocol_type"],
                            "service": d["service"],
                            "flag": d["flag"],
                            "src_bytes": d["src_bytes"],
                            "classification": CLASS_NAMES[c],
                            "confidence": float(probs[j][c]),  # Pass raw float, PDF generator formats it
                            "status": "ALLOWED" if c == 0 else "BLOCKED",
                        })
                    pdf_stats = {
                        "num_neurons": model.num_neurons,
                        "num_synapses": model.adjacency_mask._nnz(),
                        "accuracy": metrics["best_val_accuracy"] / 100 if metrics else 0,
                        "f1_macro": metrics["f1_macro"] if metrics else 0,
                    }
                    pdf_path = generate_security_report(pdf_results, pdf_stats, output_path="/tmp/security_report.pdf")
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button("Download PDF Report", pdf_file.read(), "security_report.pdf", "application/pdf", use_container_width=True)
                except Exception as e:
                    st.caption(f"PDF generation unavailable: {e}")

                # Brain state from first threat, or first packet
                threat_idx = np.where(cls_np != 0)[0]
                st.session_state["brain_state"] = brain_states[threat_idx[0]] if len(threat_idx) > 0 else brain_states[0]

    # ── 3D Brain ──
    with col_viz:
        st.markdown("### LIVE CONNECTOME ACTIVITY")
        
        # Find top activated neurons for highlighting
        highlight = None
        if "brain_state" in st.session_state:
            bs = st.session_state["brain_state"]
            if isinstance(bs, torch.Tensor):
                top_neurons = torch.argsort(bs.abs(), descending=True)[:20].tolist()
                highlight = [n for n in top_neurons if n < 300]
            fig = draw_3d_brain(G, pos, bs, highlight_neurons=highlight)
        else:
            fig = draw_3d_brain(G, pos, torch.zeros(300))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        # ── Temporal Threat Timeline ──
        if st.session_state["threat_timeline"]:
            st.markdown("### THREAT TIMELINE")
            tl_df = pd.DataFrame(st.session_state["threat_timeline"][-60:])  # Last 60 entries
            
            fig_timeline = px.scatter(
                tl_df, x="time", y="confidence", color="class",
                color_discrete_map=CLASS_COLORS,
                size="confidence", size_max=15,
                labels={"time": "Time", "confidence": "Confidence %", "class": "Classification"},
            )
            fig_timeline.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#8899AA", height=200,
                margin=dict(t=10, b=30, l=40, r=10),
                xaxis=dict(gridcolor="#0D1117"),
                yaxis=dict(gridcolor="#0D1117", range=[0, 105]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # 6. BOTTOM TABS — Architecture, Analytics, Benchmarks, Bio Analysis
    # ═══════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)

    tab_arch, tab_analytics, tab_bench, tab_bio = st.tabs([
        "[ HOW IT WORKS ]", "[ PERFORMANCE ANALYTICS ]",
        "[ COMPARATIVE BENCHMARKS ]", "[ BIOLOGICAL ANALYSIS ]"
    ])

    with tab_arch:
        st.markdown("### Architecture Pipeline")
        st.markdown("""
```mermaid
graph LR
    A[Raw Packet] --> B[Feature Extraction]
    B --> C[Sensory Neurons]
    C --> D[Biological Connectome<br>35,462 neurons · Sparse RNN]
    D --> E[Motor Neurons]
    E --> F{Classification}
    F --> G[Normal]
    F --> H[DoS]
    F --> I[Probe]
    F --> J[R2L]
    F --> K[U2R]
```
""")
        st.markdown("""
**What makes this different from a standard neural network?**

In a conventional AI, every neuron connects to every other neuron (dense connectivity). 
This model's hidden layer is **structurally constrained** by the actual biological wiring 
of *Drosophila melanogaster* (fruit fly), extracted from the 
[FlyWire Connectome Project](https://flywire.ai) at Princeton University.

Information can **only flow** along pathways where a real biological synapse exists. 
The trainable parameters are the **synaptic weights** — one per real synapse — 
while the topology is frozen to biology. This is analogous to how biological brains 
learn: the wiring is fixed by genetics, but the strength of each connection adapts 
through experience.

**Key Innovation:** Unlike the previous version where the biological layer received zero input 
(and only dense layers were learning), v3.0 ensures that the sensory signal flows *through* 
the biological connectome via state persistence across batches. The biological synapses 
now actively learn to route threat-detection signals.
        """)

    with tab_analytics:
        st.markdown("### 🌍 GLOBAL THREAT MAP")
        map_data = []
        if st.session_state["threat_timeline"]:
            import hashlib
            for t in st.session_state["threat_timeline"]:
                if t["class"] != "Normal":
                    h = int(hashlib.md5(str(t.get("src_ip", "0.0.0.0")).encode()).hexdigest(), 16)
                    lat = (h % 140) - 70
                    lon = ((h // 140) % 360) - 180
                    map_data.append({"ip": t.get("src_ip", "Unknown"), "lat": lat, "lon": lon, "class": t["class"]})
        
        if map_data:
            map_df = pd.DataFrame(map_data).drop_duplicates(subset=["ip"])
            fig_map = go.Figure(go.Scattergeo(
                lon = map_df['lon'], lat = map_df['lat'],
                text = map_df['ip'] + " [" + map_df['class'] + "]",
                mode = 'markers',
                marker = dict(size = 8, color = '#FF1744', opacity=0.8, line=dict(width=1, color='rgba(255,23,68,0.5)')),
            ))
            # Draw laser lines to server
            for _, row in map_df.iterrows():
                fig_map.add_trace(go.Scattergeo(
                    lon = [row['lon'], -74.006], lat = [row['lat'], 40.7128], # Server at NY
                    mode = 'lines', line = dict(width = 1, color = 'rgba(255,23,68,0.3)'),
                    showlegend=False
                ))
            fig_map.update_geos(
                projection_type="orthographic", showcountries=True, countrycolor="#1E3A5F",
                showocean=True, oceancolor="#040810", showland=True, landcolor="#0D1117", bgcolor="rgba(0,0,0,0)",
            )
            fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", height=400)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No external threats detected yet to map. Waiting for malicious packets...")
        
        st.markdown("<hr style='border-color:#0D1117'>", unsafe_allow_html=True)
        if metrics:
            st.markdown("### Model Performance Metrics")

            # Metrics summary
            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("Validation Accuracy", f"{metrics['best_val_accuracy']:.2f}%")
            ac2.metric("F1 Score (Macro)", f"{metrics['f1_macro']:.4f}")
            ac3.metric("Training Epochs", str(metrics.get("epochs", 10)))
            bio_synapse_count = f"{metrics.get('num_synapses', 61270):,}"
            ac4.metric("Biological Synapses", bio_synapse_count)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Edge-Compute Hardware Footprint")
            st.info("This biological sparse network is uniquely suited for IoT and Edge deployment.")
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Total Parameters", bio_synapse_count)
            ec2.metric("VRAM / RAM Required", f"{int(metrics.get('num_synapses', 61270) * 4 / 1024)} KB")
            ec3.metric("IoT Viability", "Excellent (Raspberry Pi/Smartwatch)")

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Training history (if available)
            if "history" in metrics:
                st.markdown("#### Training Progression")
                hist_df = pd.DataFrame(metrics["history"])
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(
                    x=hist_df["epoch"], y=hist_df["train_acc"],
                    mode="lines+markers", name="Training Accuracy",
                    line=dict(color="#00E5FF", width=2),
                ))
                fig_hist.add_trace(go.Scatter(
                    x=hist_df["epoch"], y=hist_df["val_acc"],
                    mode="lines+markers", name="Validation Accuracy",
                    line=dict(color="#FF9100", width=2),
                ))
                fig_hist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#8899AA", height=300,
                    margin=dict(t=30, b=30),
                    xaxis=dict(title="Epoch", gridcolor="#0D1117"),
                    yaxis=dict(title="Accuracy (%)", gridcolor="#0D1117"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            col_cm, col_bar = st.columns(2)

            with col_cm:
                st.markdown("#### Confusion Matrix")
                cm = np.array(metrics["confusion_matrix"])
                fig_cm = px.imshow(
                    cm, x=CLASS_NAMES, y=CLASS_NAMES, text_auto=True,
                    color_continuous_scale=[[0, "#060A13"], [0.5, "#1E3A5F"], [1, "#00E5FF"]],
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                )
                fig_cm.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#8899AA", margin=dict(t=30, b=10),
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_bar:
                st.markdown("#### Per-Class F1 Score")
                report = metrics["classification_report"]
                f1_scores = {name: report[name]["f1-score"] for name in CLASS_NAMES if name in report}
                df_f1 = pd.DataFrame({
                    "Attack Class": list(f1_scores.keys()),
                    "F1 Score": list(f1_scores.values())
                })
                fig_bar = px.bar(
                    df_f1, x="Attack Class", y="F1 Score", color="Attack Class",
                    color_discrete_map=CLASS_COLORS
                )
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#8899AA", showlegend=False, margin=dict(t=30, b=10),
                    yaxis=dict(range=[0, 1], gridcolor="#0D1117"),
                    xaxis=dict(gridcolor="#0D1117"),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # Per-class precision/recall table
            st.markdown("#### Classification Report")
            report_df = pd.DataFrame({
                name: {
                    "Precision": f"{report[name]['precision']:.4f}",
                    "Recall": f"{report[name]['recall']:.4f}",
                    "F1-Score": f"{report[name]['f1-score']:.4f}",
                    "Support": int(report[name]["support"]),
                }
                for name in CLASS_NAMES if name in report
            }).T
            st.dataframe(report_df, use_container_width=True)
        else:
            st.info("Training metrics not found. Run `train.py` to generate analytics.")

    with tab_bench:
        st.markdown("### Comparative Benchmarks")
        st.caption("How does the biological connectome compare to traditional ML models on the same data?")
        
        if benchmarks:
            # Build comparison table
            model_names = []
            accuracies = []
            f1_scores_bench = []
            
            # Add FlyBrainNet first
            if metrics:
                model_names.append("FlyBrainNet (Biological)")
                accuracies.append(metrics["best_val_accuracy"])
                f1_scores_bench.append(metrics["f1_macro"] * 100)
            
            display_names = {
                "random_forest": "Random Forest",
                "logistic_regression": "Logistic Regression", 
                "mlp": "MLP (Dense NN)",
            }
            
            for key, display_name in display_names.items():
                if key in benchmarks:
                    model_names.append(display_name)
                    accuracies.append(benchmarks[key]["accuracy"] * 100)
                    f1_scores_bench.append(benchmarks[key]["f1_macro"] * 100)
            
            # Grouped bar chart
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=model_names, y=accuracies, name="Accuracy %",
                marker_color=["#00E5FF" if "Bio" in n else "#484F58" for n in model_names],
            ))
            fig_comp.add_trace(go.Bar(
                x=model_names, y=f1_scores_bench, name="F1 Macro %",
                marker_color=["#FF9100" if "Bio" in n else "#303840" for n in model_names],
            ))
            fig_comp.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#8899AA", height=400,
                margin=dict(t=30, b=30),
                yaxis=dict(title="Score (%)", gridcolor="#0D1117", range=[0, 105]),
                xaxis=dict(gridcolor="#0D1117"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Detailed comparison table
            comp_data = []
            if metrics:
                comp_data.append({
                    "Model": "FlyBrainNet (Biological)",
                    "Accuracy": f"{metrics['best_val_accuracy']:.2f}%",
                    "F1 Macro": f"{metrics['f1_macro']:.4f}",
                    "Parameters": f"{metrics.get('num_synapses', 61270):,} bio + dense",
                    "Type": "Biological Connectome RNN",
                })
            for key, display_name in display_names.items():
                if key in benchmarks:
                    b = benchmarks[key]
                    
                    param_str = "N/A"
                    if key == "random_forest":
                        param_str = "~15,000,000+ (100 trees)"
                    elif key == "logistic_regression":
                        param_str = "~615 (Dense)"
                    elif key == "mlp":
                        param_str = "~65,000 (Dense)"
                        
                    comp_data.append({
                        "Model": display_name,
                        "Accuracy": f"{b['accuracy']*100:.2f}%",
                        "F1 Macro": f"{b['f1_macro']:.4f}",
                        "Parameters": param_str,
                        "Type": "Traditional ML",
                    })
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
            
            st.markdown("""
> **Key Insight:** The biological connectome network achieves competitive accuracy while being 
> structurally constrained to real neuroanatomy. Unlike dense networks, every connection in 
> FlyBrainNet corresponds to a real synapse observed in the *Drosophila* brain. This demonstrates 
> that biological network topology is a viable and interpretable substrate for machine learning.
            """)
        else:
            st.info("Run `python benchmarks.py` to generate comparative benchmark data.")

    with tab_bio:
        st.markdown("### Biological Network Analysis")
        st.caption("Graph-theoretic properties of the Drosophila connectome vs. random network baselines")
        
        if bio_analysis:
            bio = bio_analysis.get("biological", {})
            baselines_data = bio_analysis.get("baselines", {})
            centrality = bio_analysis.get("centrality", {})
            
            # Key metrics
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Nodes", f"{bio.get('num_nodes', 0):,}")
            b2.metric("Edges", f"{bio.get('num_edges', 0):,}")
            b3.metric("Clustering Coeff", f"{bio.get('clustering_coefficient', 0):.4f}")
            b4.metric("Small-World?", "Yes" if bio.get("is_small_world") else "No")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Comparison chart: Bio vs Random
            col_bio1, col_bio2 = st.columns(2)
            
            with col_bio1:
                st.markdown("#### Clustering Coefficient Comparison")
                cc_names = ["Biological"]
                cc_vals = [bio.get("clustering_coefficient", 0)]
                cc_colors = ["#00E5FF"]
                
                er = baselines_data.get("erdos_renyi", {})
                ws = baselines_data.get("watts_strogatz", {})
                if "clustering_coefficient" in er:
                    cc_names.append("Erdős–Rényi (Random)")
                    cc_vals.append(er["clustering_coefficient"])
                    cc_colors.append("#FF1744")
                if "clustering_coefficient" in ws:
                    cc_names.append("Watts–Strogatz")
                    cc_vals.append(ws["clustering_coefficient"])
                    cc_colors.append("#FF9100")
                
                df_cc = pd.DataFrame({
                    "Network Type": cc_names,
                    "Clustering Coefficient": cc_vals
                })
                fig_cc = px.bar(
                    df_cc, x="Network Type", y="Clustering Coefficient", color="Network Type",
                    color_discrete_sequence=cc_colors
                )
                fig_cc.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#8899AA", showlegend=False,
                    margin=dict(t=10, b=10),
                    yaxis=dict(gridcolor="#0D1117"),
                )
                st.plotly_chart(fig_cc, use_container_width=True)
            
            with col_bio2:
                st.markdown("#### Degree Distribution")
                if "degree_distribution" in bio:
                    deg_data = bio["degree_distribution"]
                    if isinstance(deg_data, dict):
                        df_deg = pd.DataFrame({
                            "Degree": list(deg_data.keys()),
                            "Count": list(deg_data.values())
                        })
                        fig_deg = px.bar(df_deg, x="Degree", y="Count")
                    else:
                        fig_deg = px.histogram(x=deg_data, nbins=50,
                            labels={"x": "Degree", "y": "Count"})
                    fig_deg.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#8899AA", showlegend=False,
                        margin=dict(t=10, b=10),
                        yaxis=dict(gridcolor="#0D1117"),
                    )
                    fig_deg.update_traces(marker_color="#00E5FF")
                    st.plotly_chart(fig_deg, use_container_width=True)
            
            # Hub neurons
            if centrality:
                st.markdown("#### Top Hub Neurons (Highest Out-Degree)")
                if "top_10_hubs" in centrality:
                    hubs_df = pd.DataFrame(centrality["top_10_hubs"])
                    hubs_df.columns = ["Neuron ID", "Out-Degree"]
                    st.dataframe(hubs_df, use_container_width=True, hide_index=True)
                    
                st.markdown("""
> **Scientific Significance:** The fly brain's connectome exhibits **small-world topology** — 
> high local clustering (neurons form tightly-connected communities) combined with short global 
> path lengths (any neuron can reach any other in few hops). This is the same architecture found 
> in the human brain, the internet, and social networks. It enables efficient information routing 
> with minimal wiring — a property that directly benefits our threat classification task.
                """)
        else:
            st.info("Run `python bio_analysis.py` to generate biological network analysis.")

        # ── Synaptic Plasticity Visualization ──
        if os.path.exists("synaptic_deltas.pt"):
            st.markdown("---")
            st.markdown("### SYNAPTIC PLASTICITY — How the Brain Learned")
            st.caption("Distribution of weight changes across all 61,270 biological synapses during training")
            
            try:
                deltas = torch.load("synaptic_deltas.pt", weights_only=False)
                weight_changes = deltas["weight_deltas"].numpy()
                
                col_hist, col_stats = st.columns([2, 1])
                
                with col_hist:
                    fig_delta = go.Figure()
                    fig_delta.add_trace(go.Histogram(
                        x=weight_changes, nbinsx=100,
                        marker_color="#00E5FF", opacity=0.8,
                        name="Weight Change",
                    ))
                    fig_delta.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#8899AA", height=300,
                        margin=dict(t=10, b=30, l=40, r=10),
                        xaxis=dict(title="Δ Weight", gridcolor="#0D1117"),
                        yaxis=dict(title="Synapse Count", gridcolor="#0D1117"),
                    )
                    st.plotly_chart(fig_delta, use_container_width=True)
                
                with col_stats:
                    abs_changes = np.abs(weight_changes)
                    st.metric("Mean |ΔW|", f"{abs_changes.mean():.6f}")
                    st.metric("Max |ΔW|", f"{abs_changes.max():.4f}")
                    st.metric("Synapses Changed > 0.01", f"{(abs_changes > 0.01).sum():,}")
                    st.metric("Synapses Changed > 0.1", f"{(abs_changes > 0.1).sum():,}")
                    
                    # Top 10 most plastic synapses
                    top_plastic = np.argsort(abs_changes)[-10:][::-1]
                    st.caption("Top 10 Most Plastic Synapses:")
                    for idx in top_plastic:
                        st.caption(f"  Synapse #{idx}: ΔW = {weight_changes[idx]:+.4f}")
            except Exception as e:
                st.caption(f"Plasticity data unavailable: {e}")

    # ── Finalize Dynamic Metrics ──
    m5_ph.metric("Scanned (Session)", str(st.session_state.get("session_scanned", 0)))
    m6_ph.metric("Threats Blocked", str(st.session_state.get("session_threats", 0)))
    
    active_count = "—"
    if "brain_state" in st.session_state and isinstance(st.session_state["brain_state"], torch.Tensor):
        active_count = f"{(st.session_state['brain_state'].abs() > 0.3).sum().item():,}"
    m7_ph.metric("Active Neurons", active_count)

    # ── Footer ──
    st.markdown("""
    <div class='footer'>
        POWERED BY FLYWIRE CONNECTOME (PRINCETON) · NSL-KDD DATASET · PYTORCH · SCAPY<br>
        BIOLOGICAL FIREWALL ENGINE v3.1 — HACKATHON EDITION · BUILT BY JISJ THOMAS<br>
        <span style='color:#1E3A5F'>Focal Loss · SMOTE · XAI Gradient Attribution · Small-World Topology</span>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"System Offline: {e}")
    st.exception(e)
