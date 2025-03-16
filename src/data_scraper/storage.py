from __future__ import annotations

import sqlite3

import pandas as pd


class Storage:
    DB_FILE = "market_data.db"

    @staticmethod
    def save_to_db(data: pd.DataFrame, table_name: str):
        """Saves data to a local SQLite database."""
        conn = sqlite3.connect(Storage.DB_FILE)
        data.to_sql(table_name, conn, if_exists="replace", index=True)
        conn.close()

    @staticmethod
    def load_from_db(table_name: str):
        """Loads data from the SQLite database."""
        conn = sqlite3.connect(Storage.DB_FILE)
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn, index_col="index")
        conn.close()
        return df
