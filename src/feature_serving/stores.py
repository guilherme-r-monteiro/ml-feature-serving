import json
import sqlite3
from pathlib import Path

import pandas as pd

from feature_serving.features import FEATURE_NAMES, compute_features, feature_version, snapshot_hash

ONLINE_SCHEMA = """
CREATE TABLE IF NOT EXISTS online_features (
    client_id INTEGER NOT NULL, as_of INTEGER NOT NULL, version TEXT NOT NULL,
    snapshot TEXT NOT NULL, values_json TEXT NOT NULL,
    PRIMARY KEY (client_id, as_of)
);
CREATE TABLE IF NOT EXISTS prediction_log (
    request_id TEXT PRIMARY KEY, client_id INTEGER, as_of INTEGER, model TEXT, model_version TEXT,
    feature_version TEXT, snapshot TEXT, score REAL
);
"""


def open_database(path: Path | str) -> sqlite3.Connection:
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.executescript(ONLINE_SCHEMA)
    return connection


def materialize(connection: sqlite3.Connection, events: pd.DataFrame, clients: pd.DataFrame, as_of: int) -> int:
    by_client = {client: frame for client, frame in events[events["month"] <= as_of].groupby("client_id")}
    rows = []
    for client_id, frame in by_client.items():
        values = compute_features(frame, clients.loc[[client_id]], as_of).iloc[0].to_dict()
        rows.append((int(client_id), as_of, feature_version(), snapshot_hash(values), json.dumps(values)))
    connection.executemany("INSERT OR REPLACE INTO online_features VALUES (?,?,?,?,?)", rows)
    connection.commit()
    return len(rows)


def read_online(connection: sqlite3.Connection, client_id: int, as_of: int) -> tuple[dict[str, float], str, str] | None:
    row = connection.execute(
        "SELECT values_json, version, snapshot FROM online_features WHERE client_id=? AND as_of=?", (client_id, as_of)
    ).fetchone()
    return (json.loads(row[0]), row[1], row[2]) if row else None


def online_frame(connection: sqlite3.Connection, as_of: int) -> pd.DataFrame:
    rows = connection.execute("SELECT client_id, values_json FROM online_features WHERE as_of=?", (as_of,)).fetchall()
    return pd.DataFrame({cid: json.loads(v) for cid, v in rows}).T[list(FEATURE_NAMES)].astype(float).sort_index()
