import pandas as pd
import numpy as np

def preprocess_data(df):
    df = df.copy()

    df["age"] = 2024 - pd.to_datetime(df["dob"]).dt.year
    df["hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour

    df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)

    df["distance"] = np.sqrt(
        (df["lat"] - df["merch_lat"])**2 +
        (df["long"] - df["merch_long"])**2
    )

    df["txn_count"] = df.groupby("cc_num")["amt"].transform("count")
    df["avg_amt"] = df.groupby("cc_num")["amt"].transform("mean")
    df["amt_dev"] = df["amt"] - df["avg_amt"]

    df["is_far_location"] = (df["distance"] > df["distance"].mean()).astype(int)

    df["amt_ratio"] = df["amt"] / (df["avg_amt"] + 1)
    df["is_high_amt"] = (df["amt_ratio"] > 2).astype(int)

    df["amt_spike"] = df.groupby("cc_num")["amt"].diff().fillna(0)
    df["txn_velocity"] = df.groupby("cc_num")["unix_time"].diff().fillna(0)

    df["small_txn_flag"] = (df["amt"] < 50).astype(int)
    df["amt_log"] = np.log1p(df["amt"])

    drop_cols = [
        "first","last","street","city","state","job",
        "trans_num","merchant","category","dob",
        "trans_date_trans_time"
    ]

    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    df = df.select_dtypes(include=["int64","float64"])
    df.fillna(0, inplace=True)

    return df