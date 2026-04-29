import argparse

parser = argparse.ArgumentParser()

# Selection
parser.add_argument('-model', type=str, default="RF")
parser.add_argument('-dataset', type=str, default="KDD99")
parser.add_argument('-top_k', type=int, default=0) 

# Model Hyperparameter

# Phase에 따라 fs, top_n의 계수 조절 가능 
parser.add_argument('-phase', type=int, default=0) #1, 2, 3
parser.add_argument('-shap', type=int, default=1)

args = parser.parse_args()

dataset_configs = {
    "CIC2018": {
        "path": "CIC2018/attack_data/",
        "normal": "CIC_normal.csv",
        "anomaly_pattern": "CIC_anomaly_{}.csv",
        "labels": ['DDOS attack-HOIC', 'DDoS attacks-LOIC-HTTP', 'DoS attacks-Hulk', 'Bot',
                    'FTP-BruteForce', 'SSH-Bruteforce', 'Infiltration', 'DoS attacks-SlowHTTPTest',
                    'DoS attacks-GoldenEye', 'DoS attacks-Slowloris', 'DDOS attack-LOIC-UDP',
                    'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection', 'normal'],
        "anomaly_range": range(0, 14)
    },
    "InSDN": {
        "path": "InSDN/attack_data/",
        "normal": "InSDN_normal.csv",
        "anomaly_pattern": "InSDN_anomaly_{}.csv",
        "labels": ['BFA', 'BOTNET', 'DDOS', 'DOS', 'Probe', 'U2R', 'Web-Attack', 'normal'],
        "anomaly_range": range(0, 7)
    },
    "KDD99": {
        "path": "KDD99/attack_data/",
        "normal": "KDD99_normal.csv",
        "anomaly_pattern": "KDD99_anomaly_{}.csv",
        "labels": ['back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep', 'land', 'loadmodule', 
                    'multihop', 'neptune', 'nmap', 'perl', 'phf', 'pod', 'portsweep', 'rootkit', 'satan', 'spy', 'teardrop',
                    'warezclient', 'warezmaster', 'normal'],
        "anomaly_range": range(0, 21)
    },
    "UNSW_NB15": {
        "path": "UNSW_NB15/attack_data/",
        "normal": "UNSW_NB15_normal.csv",
        "anomaly_pattern": "UNSW_NB15_anomaly_{}.csv",
        "labels": ['analysis', 'backdoor', 'dos', 'exploits', 'fuzzers', 'generic', 'reconnaissance', 'shellcode', 'worms', 'normal'],
        "anomaly_range": range(0, 9)
    }
}

