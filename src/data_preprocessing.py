import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import pickle
import warnings
warnings.filterwarnings("ignore")


ATTACK_TYPES = ["normal", "ddos", "data_exfiltration", "botnet", "port_scan"]

def create_synthetic_iot_data(
    n_samples: int = 12000,
    n_devices: int = 60,
    anomaly_ratio: float = 0.15,
    multiclass: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)

    n_normal   = int(n_samples * (1 - anomaly_ratio))
    n_anomaly  = n_samples - n_normal
    attack_per = n_anomaly // 4

    normal = pd.DataFrame({
        "src_ip":             [f"192.168.{np.random.randint(0,5)}.{np.random.randint(1,255)}"
                               for _ in range(n_normal)],
        "dst_ip":             [f"192.168.{np.random.randint(0,5)}.{np.random.randint(1,255)}"
                               for _ in range(n_normal)],
        "src_device":         np.random.randint(0, n_devices, n_normal),
        "dst_device":         np.random.randint(0, n_devices, n_normal),
        "src_port":           np.random.randint(1024, 65535, n_normal),
        "dst_port":           np.random.choice([80, 443, 22, 1883, 8883, 8080, 53], n_normal),
        "protocol":           np.random.choice(["TCP", "UDP", "ICMP", "HTTP", "MQTT"], n_normal,
                                               p=[0.4, 0.3, 0.1, 0.1, 0.1]),
        "packet_count":       np.random.poisson(45, n_normal),
        "byte_count":         np.random.poisson(4500, n_normal),
        "duration":           np.random.exponential(8, n_normal),
        "packet_size":        np.random.normal(480, 120, n_normal).clip(50, 1500),
        "inter_arrival_time": np.random.exponential(1.5, n_normal),
        "tcp_flags":          np.random.randint(0, 64, n_normal),
        "label":              "normal",
        "attack_type":        "normal",
    })

    def _base_block(n, attack_name):
        return pd.DataFrame({
            "src_ip":             [f"10.0.{np.random.randint(0,3)}.{np.random.randint(1,255)}"
                                   for _ in range(n)],
            "dst_ip":             [f"192.168.{np.random.randint(0,5)}.{np.random.randint(1,255)}"
                                   for _ in range(n)],
            "src_device":         np.random.randint(0, n_devices, n),
            "dst_device":         np.random.randint(0, n_devices, n),
            "src_port":           np.random.randint(1024, 65535, n),
            "dst_port":           np.random.choice([80, 443, 22, 53], n),
            "protocol":           "TCP",
            "tcp_flags":          np.random.randint(0, 64, n),
            "label":              "anomaly",
            "attack_type":        attack_name,
        })

    ddos = _base_block(attack_per, "ddos")
    ddos["packet_count"]       = np.random.poisson(800, attack_per)
    ddos["byte_count"]         = ddos["packet_count"] * np.random.uniform(60, 80, attack_per)
    ddos["duration"]           = np.random.uniform(0.1, 1.0, attack_per)
    ddos["packet_size"]        = np.random.normal(70, 10, attack_per).clip(50, 120)
    ddos["inter_arrival_time"] = np.random.exponential(0.05, attack_per)

    exfil = _base_block(attack_per, "data_exfiltration")
    exfil["packet_count"]       = np.random.poisson(120, attack_per)
    exfil["byte_count"]         = np.random.poisson(90000, attack_per)
    exfil["duration"]           = np.random.uniform(30, 200, attack_per)
    exfil["packet_size"]        = np.random.normal(1400, 50, attack_per).clip(1000, 1500)
    exfil["inter_arrival_time"] = np.random.exponential(3.0, attack_per)

    botnet = _base_block(attack_per, "botnet")
    botnet["packet_count"]       = np.random.poisson(180, attack_per)
    botnet["byte_count"]         = np.random.poisson(18000, attack_per)
    botnet["duration"]           = np.random.uniform(5, 20, attack_per)
    botnet["packet_size"]        = np.random.normal(200, 20, attack_per).clip(100, 300)
    botnet["inter_arrival_time"] = np.random.normal(0.5, 0.05, attack_per).clip(0.3, 0.7)

    scan = _base_block(attack_per + (n_anomaly - 4 * attack_per), "port_scan")
    n_scan = len(scan)
    scan["packet_count"]       = np.random.poisson(3, n_scan)
    scan["byte_count"]         = np.random.poisson(180, n_scan)
    scan["duration"]           = np.random.uniform(0.001, 0.05, n_scan)
    scan["packet_size"]        = np.random.normal(60, 5, n_scan).clip(40, 80)
    scan["inter_arrival_time"] = np.random.exponential(0.01, n_scan)

    df = pd.concat([normal, ddos, exfil, botnet, scan], ignore_index=True)

    df["bytes_per_packet"]    = df["byte_count"] / (df["packet_count"] + 1)
    df["packets_per_second"]  = df["packet_count"] / (df["duration"].clip(lower=0.001) + 0.001)
    df["bytes_per_second"]    = df["byte_count"] / (df["duration"].clip(lower=0.001) + 0.001)
    df["payload_ratio"]       = df["byte_count"] / (df["packet_size"] * df["packet_count"] + 1)

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].clip(lower=0)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if not multiclass:
        df["target"] = (df["label"] != "normal").astype(int)
    else:
        le = LabelEncoder()
        df["target"] = le.fit_transform(df["attack_type"])
        df.attrs["label_encoder"] = le
        df.attrs["class_names"] = le.classes_.tolist()

    print(f"\n[OK] Synthetic dataset: {len(df):,} samples | {n_devices} devices")
    vc = df["attack_type"].value_counts()
    for k, v in vc.items():
        print(f"     {k:<22} {v:>6} ({100*v/len(df):5.1f}%)")

    return df


def load_bot_iot(path: str, multiclass: bool = False, sample_frac: float = 1.0) -> pd.DataFrame:
    print(f"[INFO] Loading BoT-IoT from {path}...")
    df = pd.read_csv(path)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    rename = {
        "saddr": "src_ip", "daddr": "dst_ip",
        "sport": "src_port", "dport": "dst_port",
        "proto": "protocol", "pkts": "packet_count",
        "bytes": "byte_count", "dur": "duration",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    if "packet_size" not in df.columns:
        df["packet_size"] = df["byte_count"] / (df["packet_count"] + 1)
    if "inter_arrival_time" not in df.columns:
        df["inter_arrival_time"] = df["duration"] / (df["packet_count"] + 1)

    if multiclass and "category" in df.columns:
        le = LabelEncoder()
        df["target"] = le.fit_transform(df["category"].fillna("normal").astype(str))
        df.attrs["class_names"] = le.classes_.tolist()
    else:
        df["target"] = (df["label"].astype(str).str.lower() != "0").astype(int)

    _add_derived_features(df)
    print(f"[OK] BoT-IoT loaded: {len(df):,} rows  |  anomaly rate: "
          f"{100*df['target'].mean():.1f}%")
    return df


def load_ton_iot(path: str, multiclass: bool = False, sample_frac: float = 0.10) -> pd.DataFrame:
    print(f"[INFO] Loading ToN-IoT from {path} (sample={sample_frac:.0%})...")
    df = pd.read_csv(path)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    rename = {
        "src_ip": "src_ip", "dst_ip": "dst_ip",
        "src_port": "src_port", "dst_port": "dst_port",
        "proto": "protocol", "pkts": "packet_count",
        "bytes": "byte_count", "duration": "duration",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    if "packet_size" not in df.columns:
        df["packet_size"] = df.get("byte_count", pd.Series(np.zeros(len(df)))) / \
                            (df.get("packet_count", pd.Series(np.ones(len(df)))) + 1)
    if "inter_arrival_time" not in df.columns:
        df["inter_arrival_time"] = 0.0

    if multiclass and "type" in df.columns:
        le = LabelEncoder()
        df["target"] = le.fit_transform(df["type"].fillna("normal").astype(str))
        df.attrs["class_names"] = le.classes_.tolist()
    else:
        df["target"] = df.get("label", pd.Series(np.zeros(len(df)))).astype(int)

    _add_derived_features(df)
    print(f"[OK] ToN-IoT loaded: {len(df):,} rows  |  anomaly rate: "
          f"{100*df['target'].mean():.1f}%")
    return df


def _add_derived_features(df: pd.DataFrame):
    if "bytes_per_packet" not in df.columns:
        df["bytes_per_packet"] = df.get("byte_count", 0) / \
                                 (df.get("packet_count", 1).replace(0, 1) + 1)
    if "packets_per_second" not in df.columns:
        dur = df.get("duration", pd.Series(np.ones(len(df)))).replace(0, 0.001)
        df["packets_per_second"] = df.get("packet_count", 0) / (dur + 0.001)
    if "bytes_per_second" not in df.columns:
        dur = df.get("duration", pd.Series(np.ones(len(df)))).replace(0, 0.001)
        df["bytes_per_second"] = df.get("byte_count", 0) / (dur + 0.001)
    if "payload_ratio" not in df.columns:
        df["payload_ratio"] = df.get("byte_count", 0) / \
                              (df.get("packet_size", 1).replace(0, 1) *
                               df.get("packet_count", 1).replace(0, 1) + 1)


FEATURE_COLS = [
    "src_port", "dst_port",
    "packet_count", "byte_count", "duration",
    "packet_size", "inter_arrival_time", "tcp_flags",
    "bytes_per_packet", "packets_per_second",
    "bytes_per_second", "payload_ratio",
]

PROTOCOL_COL = "protocol"


class IoTDataPreprocessor:

    def __init__(self):
        self.scaler          = StandardScaler()
        self.protocol_enc    = LabelEncoder()
        self.feature_columns: list = []
        self._fitted         = False

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        df = self._clean(df)
        df = self._encode_protocol(df, fit=True)
        feat_cols  = self._build_feature_list(df)
        features   = df[feat_cols].copy()
        features   = pd.DataFrame(
            self.scaler.fit_transform(features), columns=feat_cols
        )
        self.feature_columns = feat_cols
        self._fitted = True
        labels = df["target"].values.astype(int)
        self._print_stats(features, labels)
        return features, labels

    def transform(self, df: pd.DataFrame) -> tuple:
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        df = self._clean(df)
        df = self._encode_protocol(df, fit=False)
        features = df[self.feature_columns].copy()
        features = pd.DataFrame(
            self.scaler.transform(features), columns=self.feature_columns
        )
        labels = df["target"].values.astype(int)
        return features, labels

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "protocol_enc": self.protocol_enc,
                "feature_columns": self.feature_columns,
            }, f)
        print(f"[OK] Preprocessor saved → {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.scaler          = state["scaler"]
        self.protocol_enc    = state["protocol_enc"]
        self.feature_columns = state["feature_columns"]
        self._fitted         = True
        print(f"[OK] Preprocessor loaded ← {path}")

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        num = df.select_dtypes(include=[np.number]).columns
        df[num] = df[num].fillna(df[num].median())
        cat = df.select_dtypes(include=["object"]).columns
        for c in cat:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
        return df

    def _encode_protocol(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if PROTOCOL_COL in df.columns:
            vals = df[PROTOCOL_COL].astype(str)
            if fit:
                self.protocol_enc.fit(vals)
            safe = vals.map(
                lambda x: x if x in self.protocol_enc.classes_ else self.protocol_enc.classes_[0]
            )
            df["protocol_enc"] = self.protocol_enc.transform(safe)
        return df

    @staticmethod
    def _build_feature_list(df: pd.DataFrame) -> list:
        cols = []
        if "protocol_enc" in df.columns:
            cols.append("protocol_enc")
        for c in FEATURE_COLS:
            if c in df.columns:
                cols.append(c)
        return cols

    @staticmethod
    def _print_stats(features: pd.DataFrame, labels: np.ndarray):
        counts = np.bincount(labels)
        print(f"\n[OK] Preprocessing complete")
        print(f"     Feature dim : {features.shape[1]}")
        print(f"     Samples     : {len(labels):,}")
        for i, c in enumerate(counts):
            name = "Normal" if i == 0 else f"Class-{i}"
            print(f"     {name:<12}: {c:>6,}  ({100*c/len(labels):5.1f}%)")
