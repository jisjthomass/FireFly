"""
packet_engine.py — Deep Packet Feature Extraction
===================================================
Centralizes all raw-packet-to-feature-vector conversion logic.
Supports both Scapy packet objects and raw PCAP/PCAPNG files.
Maps packets to the NSL-KDD feature schema for model inference.
"""

import tempfile
import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any

from scapy.all import TCP, UDP, ICMP, IP, ARP, DNS


# ── Port-to-service mapping (NSL-KDD convention) ──
PORT_SERVICE_MAP = {
    20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    53: "domain_u", 79: "finger", 80: "http", 110: "pop_3", 111: "sunrpc",
    113: "auth", 119: "nntp", 123: "ntp_u", 139: "netbios_ssn",
    143: "imap4", 389: "ldap", 443: "http_443s", 513: "login", 514: "shell",
    515: "printer", 530: "courier", 540: "uucp", 993: "imap4",
    995: "pop_3", 1080: "socks", 3306: "sql_net", 5432: "sql_net",
    6667: "IRC", 8080: "http",
}

# ── TCP flag decoding (NSL-KDD flag field) ──
def _decode_tcp_flags(pkt) -> str:
    """Map Scapy TCP flags to NSL-KDD flag strings."""
    if TCP not in pkt:
        return "SF"  # default for non-TCP

    flags = str(pkt[TCP].flags)

    if flags == "S":
        return "S0"         # SYN sent, no reply
    elif flags == "SA":
        return "S1"         # SYN-ACK (connection partially open)
    elif flags == "A":
        return "SF"         # ACK (normal established)
    elif flags == "FA" or flags == "F":
        return "SF"         # FIN-ACK (normal close)
    elif flags == "R" or flags == "RA":
        return "REJ"        # RST (connection rejected)
    elif flags == "PA":
        return "SF"         # PSH-ACK (data transfer, normal)
    elif "R" in flags:
        return "RSTO"       # Reset with data
    elif "S" in flags and "A" not in flags:
        return "S0"
    else:
        return "SF"


def _identify_service(pkt) -> str:
    """Identify the service from port numbers and payload inspection."""
    if TCP in pkt:
        dport = pkt[TCP].dport
        sport = pkt[TCP].sport
    elif UDP in pkt:
        dport = pkt[UDP].dport
        sport = pkt[UDP].sport
    else:
        return "other"

    # Check destination port first, then source
    if dport in PORT_SERVICE_MAP:
        return PORT_SERVICE_MAP[dport]
    if sport in PORT_SERVICE_MAP:
        return PORT_SERVICE_MAP[sport]

    # DNS detection via payload
    if DNS in pkt:
        return "domain_u" if UDP in pkt else "domain"

    return "private"


def extract_packet_features(pkt) -> Optional[Dict[str, Any]]:
    """
    Extract NSL-KDD-compatible features from a single Scapy packet.

    Returns a dict with all 38 numeric features plus protocol_type,
    service, and flag as strings, or None if the packet has no IP layer.
    """
    # Skip non-IP packets (ARP, STP, LLDP, etc.)
    if not pkt.haslayer(IP):
        return None

    features: Dict[str, Any] = {
        "duration": 0,
        "src_bytes": 0,
        "dst_bytes": 0,
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 0,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": 1,
        "srv_count": 1,
        "serror_rate": 0.0,
        "srv_serror_rate": 0.0,
        "rerror_rate": 0.0,
        "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0,
        "dst_host_count": 1,
        "dst_host_srv_count": 1,
        "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.0,
        "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0,
    }

    # ── Protocol ──
    if TCP in pkt:
        protocol = "tcp"
    elif UDP in pkt:
        protocol = "udp"
    elif ICMP in pkt:
        protocol = "icmp"
    else:
        protocol = "tcp"  # default fallback

    # ── Payload size ──
    if IP in pkt:
        features["src_bytes"] = len(pkt[IP].payload)
        # Check for land attack (src == dst)
        if pkt[IP].src == pkt[IP].dst:
            features["land"] = 1

    # ── Urgent pointer ──
    if TCP in pkt and pkt[TCP].urgptr > 0:
        features["urgent"] = 1

    # ── Wrong fragment ──
    if IP in pkt and pkt[IP].flags.MF:
        features["wrong_fragment"] = 1

    # ── Service & Flag ──
    service = _identify_service(pkt)
    flag = _decode_tcp_flags(pkt)

    # Flag-based rate estimation
    if flag in ("S0", "S1", "S2", "S3"):
        features["serror_rate"] = 1.0
        features["srv_serror_rate"] = 1.0
        features["dst_host_serror_rate"] = 1.0
        features["dst_host_srv_serror_rate"] = 1.0
    elif flag in ("REJ", "RSTO", "RSTOS0", "RSTR"):
        features["rerror_rate"] = 1.0
        features["srv_rerror_rate"] = 1.0
        features["dst_host_rerror_rate"] = 1.0
        features["dst_host_srv_rerror_rate"] = 1.0

    features["protocol_type"] = protocol
    features["service"] = service
    features["flag"] = flag

    return features


def packet_to_model_input(
    pkt,
    train_columns: pd.Index,
    scaler,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Convert a single Scapy packet into a scaled feature vector
    aligned with the training schema.

    Returns:
        features: (n_features,) numpy array ready for the model.
        display:  dict with human-readable packet info for the UI.
    """
    raw = extract_packet_features(pkt)
    if raw is None:
        return None, None

    from scapy.layers.inet import IP
    src_ip = pkt[IP].src if IP in pkt else f"10.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}"
    dst_ip = pkt[IP].dst if IP in pkt else "192.168.1.100"

    display = {
        "protocol_type": raw["protocol_type"],
        "service": raw["service"],
        "flag": raw["flag"],
        "src_bytes": raw["src_bytes"],
        "dst_bytes": raw["dst_bytes"],
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "label": "Live Capture",
    }

    # One-hot encode and align to training columns
    df = pd.DataFrame([{k: v for k, v in raw.items()}])
    df_encoded = pd.get_dummies(
        df, columns=["protocol_type", "service", "flag"]
    )
    for col in train_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[train_columns]

    features = scaler.transform(df_encoded)[0]
    return features, display


def parse_pcap_batch(
    file_bytes: bytes,
    train_columns: pd.Index,
    scaler,
    max_packets: int = 500,
) -> Tuple[Optional[np.ndarray], Optional[List[Dict[str, Any]]]]:
    """
    Parse a PCAP/PCAPNG file and convert up to `max_packets` into
    model-ready feature vectors.

    Uses PcapReader (generator) to avoid loading the full file into RAM.

    Returns:
        features:  (n_packets, n_features) numpy array, or None.
        displays:  list of display dicts, or None.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    all_features: List[np.ndarray] = []
    all_displays: List[Dict[str, Any]] = []

    try:
        from scapy.all import sniff
        
        def process_pkt(pkt):
            feat, disp = packet_to_model_input(pkt, train_columns, scaler)
            if feat is not None:
                disp["id"] = len(all_features) + 1
                all_features.append(feat)
                all_displays.append(disp)
        
        def stop_check(pkt):
            if max_packets > 0 and len(all_features) >= max_packets:
                return True
            return False

        sniff(offline=tmp_path, prn=process_pkt, stop_filter=stop_check, store=False)
        
    except Exception as e:
        import sys
        print(f"[PacketEngine] PCAP parse error: {e}", file=sys.stderr)
    finally:
        os.remove(tmp_path)

    if not all_features:
        return None, None

    return np.array(all_features), all_displays
