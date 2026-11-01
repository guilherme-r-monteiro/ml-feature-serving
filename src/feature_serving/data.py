from pathlib import Path

import pandas as pd

MONTHS = 6
STATIC_COLUMNS = ("LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE")
STATUS_COLUMNS = ("PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_0")
BILL_COLUMNS = tuple(f"BILL_AMT{n}" for n in range(MONTHS, 0, -1))
PAYMENT_COLUMNS = tuple(f"PAY_AMT{n}" for n in range(MONTHS, 0, -1))


def load_clients(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    return frame.rename(columns={"ID": "client_id"})[["client_id", *STATIC_COLUMNS]].set_index("client_id")


def load_events(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path).rename(columns={"ID": "client_id"})
    parts = []
    for month in range(1, MONTHS + 1):
        parts.append(
            pd.DataFrame(
                {
                    "client_id": frame["client_id"],
                    "month": month,
                    "pay_status": frame[STATUS_COLUMNS[month - 1]],
                    "bill_amount": frame[BILL_COLUMNS[month - 1]],
                    "payment_amount": frame[PAYMENT_COLUMNS[month - 1]],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)
