"""CSV 資料匯入 SQLite 資料庫模組。

將 Kaggle Supermarket Sales CSV 檔案匯入 SQLite，
並建立正確的欄位型別與索引。
"""

import os
import sqlite3
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import datetime

class SalesRecord(BaseModel):
    invoice_id: str = Field(alias="Invoice ID")
    branch: str = Field(alias="Branch")
    city: str = Field(alias="City")
    customer_type: str = Field(alias="Customer type")
    gender: str = Field(alias="Gender")
    product_line: str = Field(alias="Product line")
    unit_price: float = Field(alias="Unit price")
    quantity: int = Field(alias="Quantity")
    tax5: float = Field(alias="Tax 5%")
    sales: float = Field(alias="Sales")
    date: datetime.date = Field(alias="Date")
    time: datetime.time = Field(alias="Time")
    payment: str = Field(alias="Payment")
    cogs: float = Field(alias="cogs")
    gross_margin_percentage: float = Field(alias="gross margin percentage")
    gross_income: float = Field(alias="gross income")
    rating: float = Field(alias="Rating")

    @field_validator("date", mode="before")
    def validate_date(cls, v):
        return datetime.datetime.strptime(v, "%m/%d/%Y").date()

    @field_validator("time", mode="before")
    def validate_time(cls, v):
        return datetime.datetime.strptime(v, "%I:%M:%S %p").time()



def init_database(csv_path: str = None, db_path: str = None) -> str:
    """將 SuperMarket Analysis CSV 檔案匯入 SQLite 資料庫。

    Args:
        csv_path: CSV 檔案路徑。若為 None，使用預設路徑 data/SuperMarket Analysis.csv。
        db_path: SQLite 資料庫檔案路徑。若為 None，使用預設路徑 data/supermarket.db。

    Process:
        1. 讀取 CSV 檔案為 pandas DataFrame
        2. 清理欄位名稱（移除空白、統一命名）
        3. 轉換資料型別（日期、數值）
        4. 建立 SQLite 資料庫與 sales 資料表
        5. 將 DataFrame 寫入資料庫
        6. 建立常用查詢的索引

    Returns:
        str: 建立的資料庫檔案路徑。

    Raises:
        FileNotFoundError: CSV 檔案不存在。
        ValueError: CSV 檔案格式不正確或欄位缺失。
    """
    project_root = Path(__file__).parent.parent.parent
    env_csv_path = Path(project_root) / os.getenv("CSV_PATH")
    env_db_path = Path(project_root) / os.getenv("DATABASE_PATH")

    if csv_path is None:
        csv_path = env_csv_path
    else:
        csv_path = Path(csv_path)

    if db_path is None:
        db_path = env_db_path
    else:
        db_path = Path(db_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV 檔案不存在: {csv_path}\n"
            f"請從 Kaggle 下載: https://www.kaggle.com/datasets/faresashraf1001/supermarket-sales"
        )

    # 讀取 CSV
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 將資料轉換為 Pydantic 模型以驗證資料 (適合在小場景下使用，若資料量大可考慮直接使用 DataFrame)
    records = [SalesRecord(**row) for row in df.to_dict(orient="records")]
    df = pd.DataFrame([record.model_dump() for record in records])

    # 建立資料庫目錄
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 若資料庫已存在，先刪除重建
    if db_path.exists():
        os.remove(db_path)

    # 寫入 SQLite
    conn = sqlite3.connect(str(db_path))
    try:
        df.to_sql("sales", conn, index=False, if_exists="replace")

        # 建立索引以加速常用查詢
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_branch ON sales(branch)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_city ON sales(city)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_product_line ON sales(product_line)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON sales(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_type ON sales(customer_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_gender ON sales(gender)")
        conn.commit()

        # 驗證匯入結果
        row_count = cursor.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
        col_count = len(df.columns)
        print(f"  資料庫建立成功: {db_path}")
        print(f"  匯入筆數: {row_count} 筆")
        print(f"  欄位數量: {col_count} 個")

    finally:
        conn.close()

    return str(db_path)


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
        init_database()
    except FileNotFoundError as e:
        print(f"[錯誤] {e}")
    except Exception as e:
        print(f"[錯誤] 資料庫初始化失敗: {e}")