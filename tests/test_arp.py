import sys, os, pickle, torch
sys.path.append(os.path.abspath('.'))
from packet_engine import packet_to_model_input
from scapy.all import ARP, Ether, IP, TCP, UDP

train_columns = torch.load("feature_columns.pt", weights_only=False)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("Parsing ARP...")
try:
    feat, disp = packet_to_model_input(ARP(), train_columns, scaler)
    print("Success")
except Exception as e:
    print(f"Error on ARP: {e}")

print("Parsing Ether...")
try:
    feat, disp = packet_to_model_input(Ether(), train_columns, scaler)
    print("Success")
except Exception as e:
    print(f"Error on Ether: {e}")

print("Parsing valid IP/TCP...")
try:
    feat, disp = packet_to_model_input(IP()/TCP(), train_columns, scaler)
    print("Success")
except Exception as e:
    print(f"Error on IP/TCP: {e}")

