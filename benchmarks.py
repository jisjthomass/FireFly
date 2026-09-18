import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ATTACK_CATEGORIES = {
    "normal": 0,
    "back": 1, "land": 1, "neptune": 1, "pod": 1, "smurf": 1,
    "teardrop": 1, "mailbomb": 1, "apache2": 1, "processtable": 1, "udpstorm": 1,
    "ipsweep": 2, "nmap": 2, "portsweep": 2, "satan": 2, "mscan": 2, "saint": 2,
    "ftp_write": 3, "guess_passwd": 3, "imap": 3, "multihop": 3, "phf": 3,
    "spy": 3, "warezclient": 3, "warezmaster": 3, "xlock": 3, "xsnoop": 3,
    "snmpguess": 3, "snmpgetattack": 3, "httptunnel": 3, "sendmail": 3, "named": 3, "worm": 3,
    "buffer_overflow": 4, "loadmodule": 4, "perl": 4, "rootkit": 4, "xterm": 4, "ps": 4, "sqlattack": 4,
}
CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
COLUMNS = [
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

def load_and_preprocess_data(filepath="KDDTrain.txt"):
    df = pd.read_csv(filepath, names=COLUMNS)
    df["label_cat"] = df["label"].map(lambda x: ATTACK_CATEGORIES.get(x, 0))
    
    # Drop original label and difficulty_level
    df = df.drop(["label", "difficulty_level"], axis=1)
    
    # One-hot encode categorical features
    categorical_cols = ["protocol_type", "service", "flag"]
    df = pd.get_dummies(df, columns=categorical_cols)
    
    X = df.drop("label_cat", axis=1)
    y = df["label_cat"]
    
    return X, y

def main():
    print("Loading data...")
    X, y = load_and_preprocess_data("KDDTrain.txt")
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            "accuracy": acc,
            "f1_macro": f1,
            "classification_report": report,
            "confusion_matrix": cm
        }
        
    print("\nSaving results...")
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nComparison Table:")
    print("-" * 50)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'F1-Macro':<10}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<25} | {metrics['accuracy']:.4f}     | {metrics['f1_macro']:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main()
