from pathlib import Path
from typing import Optional, Dict, Any
import awswrangler as wr
import pandas as pd
import redshift_connector
from global_catalog.common.db_credentials import get_redshift_local_connection_params

class RedShiftRepo:
    def __init__(self):
        self.credentials = get_redshift_local_connection_params()

    def get_conn(self):
        return redshift_connector.connect(
            host=self.credentials["host"],
            port=int(self.credentials["port"]),
            database=self.credentials["database"],
            user=self.credentials["user"],
            password=self.credentials["password"],
            ssl=True,
        )

    def read_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        conn = self.get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("select 1")
                cur.fetchall()
            return pd.read_sql_query(sql, conn, params=params or {})
        finally:
            try:
                conn.close()
            except:
                pass

    def read_sql_or_snapshot(
        self,
        sql: str,
        snapshot_csv: str,
        use_snapshot: bool,
        save_snapshot: bool = True,
        s3_snapshot_uri: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        snap_path = Path(snapshot_csv)
        if use_snapshot and snap_path.exists():
            return pd.read_csv(snap_path)
        df = self.read_sql(sql, params=params)
        if save_snapshot:
            snap_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(snap_path, index=False)
            if s3_snapshot_uri:
                wr.s3.to_csv(df=df, path=s3_snapshot_uri)
        return df

    def get_core_data(self, table: str, columns: Optional[list[str]] = None, filters: Optional[str] = None) -> pd.DataFrame:
        conn = self.get_conn()
        cols = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols} FROM {table}"
        if filters:
            query += f" WHERE {filters}"
        return wr.redshift.read_sql_query(query, con=conn)
