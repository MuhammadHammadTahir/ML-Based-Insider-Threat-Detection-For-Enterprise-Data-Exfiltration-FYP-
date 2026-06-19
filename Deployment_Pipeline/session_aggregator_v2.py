# session_aggregator.py
# FYP - Insider Threat Detection System
# Session Aggregation Module with Auto-checkpoint (No Duplicates)

from elasticsearch import Elasticsearch
import pandas as pd
import json
import os
import shutil
import time
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# CONFIG

ES_HOST         = "https://localhost:9200"
ES_USER         = "elastic"
ES_PASSWORD     = "AG=6l4+qzo0r9+saUpgu"
INDEX           = "logs-generic-default*"
INTERNAL_DOMAIN = "dtaa.com"
CLOUD_KEYWORDS  = ['dropbox', 'drive.google', 'mega',
                   'box.com', 'onedrive', 'wikileaks', 'pastebin']

CHECKPOINT_FILE = "/home/server/fyp_pipeline/last_checkpoint.txt"
OUTPUT_CSV      = "/home/server/session_aggregator_backup/Test_session.csv"
SLEEP_INTERVAL  = 60  # seconds between each run

# CHECKPOINT FUNCTIONS

def get_last_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return f.read().strip()
    return "2010-01-01T00:00:00"

def save_checkpoint(timestamp):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(timestamp))

# CONNECT TO ELASTICSEARCH

def connect_es():
    es = Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER, ES_PASSWORD),
        verify_certs=False
    )
    if es.ping():
        print("\n Connected to Elasticsearch")
    else:
        print("\n Connection failed")
        exit()
    return es

# FETCH ONLY NEW LOGS (after checkpoint)

def fetch_logs(es):
    print("\nFetching logs from Elasticsearch...")

    last_checkpoint = get_last_checkpoint()
    print(f"  Fetching logs after: {last_checkpoint}")

    result = es.search(
        index=INDEX,
        body={
            "size": 10000,
            "query": {
                "bool": {
                    "must": [
                        {"exists": {"field": "user"}},
                        {"range": {
                            "@timestamp": {
                                "gt": last_checkpoint
                            }
                        }}
                    ]
                }
            },
            "sort": [{"@timestamp": "asc"}]
        }
    )

    logon_raw  = []
    device_raw = []
    file_raw   = []
    email_raw  = []
    http_raw   = []
    latest_timestamp = last_checkpoint

    for hit in result["hits"]["hits"]:
        doc = hit["_source"]
        if "user" not in doc:
            continue

        ts = doc.get("@timestamp", "")
        if ts > latest_timestamp:
            latest_timestamp = ts

        activity = doc.get("activity", "")
        if activity in ["Logon", "Logoff"]:
            logon_raw.append(doc)
        elif activity in ["Connect", "Disconnect"]:
            device_raw.append(doc)
        elif "filename" in doc:
            file_raw.append(doc)
        elif "to" in doc:
            email_raw.append(doc)
        elif "url" in doc:
            http_raw.append(doc)

    print(f"  Logon events  : {len(logon_raw)}")
    print(f"  Device events : {len(device_raw)}")
    print(f"  File events   : {len(file_raw)}")
    print(f"  Email events  : {len(email_raw)}")
    print(f"  HTTP events   : {len(http_raw)}")

    if latest_timestamp != last_checkpoint:
        save_checkpoint(latest_timestamp)
        print(f"  âœ“ Checkpoint updated to: {latest_timestamp}")
    else:
        print("  No new logs found since last checkpoint")

    return logon_raw, device_raw, file_raw, email_raw, http_raw

# CONVERT TO DATAFRAME

def to_df(records):
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "@timestamp" in df.columns:
        df["@timestamp"] = pd.to_datetime(df["@timestamp"], errors="coerce", utc=True)
        df = df.rename(columns={"@timestamp": "date"})
    return df

# SESSION AGGREGATION

def aggregate_sessions(logon_raw, device_raw, file_raw, email_raw, http_raw):
    print("\nBuilding DataFrames...")

    logon_df  = to_df(logon_raw)
    device_df = to_df(device_raw)
    file_df   = to_df(file_raw)
    email_df  = to_df(email_raw)
    http_df   = to_df(http_raw)

    if logon_df.empty:
        print("  No logon events skipping session building")
        return pd.DataFrame()

    logon_df = logon_df.sort_values(["user", "date"])

    for df in [email_df, http_df, file_df, device_df]:
        if not df.empty and "date" in df.columns:
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

    def get_activity(df, user, start, end):
        if df.empty:
            return pd.DataFrame()
        try:
            sliced = df[start:end]
            if "user" in sliced.columns:
                return sliced[sliced["user"] == user]
            return sliced
        except:
            return pd.DataFrame()

    sessions = []
    grouped  = logon_df.groupby("user")

    print("Aggregating sessions...")
    for user, group in grouped:
        group = group.reset_index(drop=True)
        i = 0
        while i < len(group):
            if group.loc[i, "activity"] == "Logon":
                start_time = group.loc[i, "date"]
                end_time   = None

                j = i + 1
                while j < len(group):
                    if group.loc[j, "activity"] == "Logoff":
                        end_time = group.loc[j, "date"]
                        i = j
                        break
                    elif group.loc[j, "activity"] == "Logon":
                        end_time = group.loc[j, "date"]
                        i = j - 1
                        break
                    j += 1

                if end_time is None:
                    end_time = start_time + pd.Timedelta(hours=8)
                    i = len(group)

                sess_email  = get_activity(email_df,  user, start_time, end_time)
                sess_http   = get_activity(http_df,   user, start_time, end_time)
                sess_file   = get_activity(file_df,   user, start_time, end_time)
                sess_device = get_activity(device_df, user, start_time, end_time)

                # Time Features
                duration      = (end_time - start_time).total_seconds()
                is_weekend    = 1 if start_time.weekday() >= 5 else 0
                is_after_hour = 1 if (start_time.hour >= 19 or start_time.hour < 7) else 0

                # Email Features
                n_emails           = len(sess_email)
                n_ext_emails       = 0
                n_attachments      = 0
                total_email_size   = 0
                email_content_text = ""
                if not sess_email.empty:
                    if "to" in sess_email.columns:
                        n_ext_emails = sess_email[
                            ~sess_email["to"].astype(str).str.contains(INTERNAL_DOMAIN, na=False)
                        ].shape[0]
                    if "attachments" in sess_email.columns:
                        n_attachments = sess_email["attachments"].apply(
                            lambda x: 0 if (pd.isna(x) or str(x).strip() in ["nan","None",""])
                            else len(str(x).split(";"))
                        ).sum()
                    if "size" in sess_email.columns:
                        total_email_size = pd.to_numeric(
                            sess_email["size"], errors="coerce"
                        ).fillna(0).sum()
                    if "content" in sess_email.columns:
                        email_content_text = " ".join(
                            sess_email["content"].dropna().astype(str)
                        )

                # HTTP Features
                n_http            = len(sess_http)
                cloud_uploads     = 0
                http_url_text     = ""
                http_content_text = ""
                if not sess_http.empty:
                    if "url" in sess_http.columns:
                        cloud_uploads = sess_http["url"].apply(
                            lambda x: 1 if any(
                                k in str(x).lower() for k in CLOUD_KEYWORDS
                            ) else 0
                        ).sum()
                        http_url_text = " ".join(
                            sess_http["url"].dropna().astype(str)
                        )
                    if "content" in sess_http.columns:
                        http_content_text = " ".join(
                            sess_http["content"].dropna().astype(str)
                        )

                # File Features
                n_file_copies     = len(sess_file)
                n_file_to_usb     = 0
                file_names_text   = ""
                file_content_text = ""
                if not sess_file.empty:
                    if "to_removable_media" in sess_file.columns:
                        n_file_to_usb = sess_file["to_removable_media"].astype(str).str.contains(
                            "True", case=False
                        ).sum()
                    if "filename" in sess_file.columns:
                        file_names_text = " ".join(
                            sess_file["filename"].dropna().astype(str)
                        )
                    if "content" in sess_file.columns:
                        file_content_text = " ".join(
                            sess_file["content"].dropna().astype(str)
                        )

                # Device Features
                n_usb_connects = 0
                if not sess_device.empty and "activity" in sess_device.columns:
                    n_usb_connects = sess_device["activity"].str.contains(
                        "Connect", case=False, na=False
                    ).sum()

                sessions.append({
                    "user"               : user,
                    "start"              : start_time,
                    "end"                : end_time,
                    "duration"           : duration,
                    "is_weekend"         : is_weekend,
                    "is_after_hour"      : is_after_hour,
                    "emails_count"       : n_emails,
                    "ext_emails_count"   : n_ext_emails,
                    "attachments_count"  : int(n_attachments),
                    "total_email_size"   : int(total_email_size),
                    "email_content_text" : email_content_text,
                    "http_count"         : n_http,
                    "cloud_uploads_count": int(cloud_uploads),
                    "http_url_text"      : http_url_text,
                    "http_content_text"  : http_content_text,
                    "usb_connects_count" : int(n_usb_connects),
                    "file_copies_count"  : n_file_copies,
                    "file_to_usb_count"  : int(n_file_to_usb),
                    "file_names_text"    : file_names_text,
                    "file_content_text"  : file_content_text,
                  #  "label"              : 0
                })
            i += 1

    sessions_df = pd.DataFrame(sessions)
    print(f"  \n  {len(sessions_df)} new sessions built")
    return sessions_df

# SAVE TO CSV (append, no duplicates)

def save_sessions(sessions_df):
    if sessions_df.empty:
        print("  No new sessions to save")
        return

    file_exists = os.path.exists(OUTPUT_CSV)
    sessions_df.to_csv(
        OUTPUT_CSV,
        mode="a",
        header=not file_exists,
        index=False
    )
    print(f"  \n {len(sessions_df)} sessions appended to Test_sessions.csv")
    
    src  = "/home/server/session_aggregator_backup/Test_sessions.csv"
    dest = "/home/server/fyp_pipeline/Test_sessions.csv"
    try:
        if os.path.exists(src):
            shutil.copy2(src, dest)
            print(f"\n Sessions Built Successfully")
        else:
            print(f"\n Sessions not Built")
    except Exception as e:
        print(f"  \n  Error: Sessions not Built")




# MAIN LOOP

if __name__ == "__main__":
    print("=" * 50)
    print("  FYP Session Aggregator - Starting")
    print(f"  Interval: every {SLEEP_INTERVAL} seconds")
    print("=" * 50)

    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running cycle...")

            es = connect_es()

            logon_raw, device_raw, file_raw, email_raw, http_raw = fetch_logs(es)

            total_new = sum([
                len(logon_raw), len(device_raw),
                len(file_raw), len(email_raw), len(http_raw)
            ])

            if total_new == 0:
                print("  No new logs - skipping aggregation")
            else:
                sessions_df = aggregate_sessions(
                    logon_raw, device_raw,
                    file_raw, email_raw, http_raw
                )
                save_sessions(sessions_df)

        except Exception as e:
            print(f"  \n Error: {e}")

        print(f"  Sleeping {SLEEP_INTERVAL}s...")
        time.sleep(SLEEP_INTERVAL)
