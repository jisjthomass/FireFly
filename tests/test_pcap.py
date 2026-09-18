import sys
import os
import pandas as pd
import pickle
import torch
sys.path.append(os.path.abspath('.'))
from packet_engine import parse_pcap_batch
import glob

# Try to find the exact pcapng the user uploaded (it's likely in streamlit's tmp or we can just try to see if there's any pcapng)
pcaps = glob.glob("*.pcapng")
print("Found pcaps:", pcaps)

train_columns = torch.load("feature_columns.pt", weights_only=False)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

for pcap in pcaps:
    print(f"Testing {pcap}")
    with open(pcap, "rb") as f:
        file_bytes = f.read()
    feats, disps = parse_pcap_batch(file_bytes, train_columns, scaler, max_packets=0)
    if feats is None:
        print("Returned None!")
    else:
        print(f"Parsed {len(feats)} packets successfully.")
