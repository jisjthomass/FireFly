import pandas as pd
import urllib.request
import os

# NSL-KDD is a standard benchmark dataset for Network Intrusion Detection
url_train = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
url_test = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty_level"
]

def prepare_dataset():
    print("Downloading NSL-KDD Cybersecurity Dataset...")
    urllib.request.urlretrieve(url_train, "KDDTrain.txt")
    
    df = pd.read_csv("KDDTrain.txt", names=columns)
    
    # We map any attack type to '1' (Malicious) and 'normal' to '0' (Benign)
    df['is_malicious'] = (df['label'] != 'normal').astype(int)
    
    print(f"Downloaded {len(df)} network packets.")
    print(f"Benign Packets: {len(df[df['is_malicious'] == 0])}")
    print(f"Malicious Packets (Attacks): {len(df[df['is_malicious'] == 1])}")
    
    # Let's peek at the first malicious packet
    print("\nSample Malicious Packet (Cyber Attack):")
    print(df[df['is_malicious'] == 1].iloc[0][['protocol_type', 'service', 'flag', 'src_bytes', 'label']])
    
    return df

if __name__ == "__main__":
    prepare_dataset()
