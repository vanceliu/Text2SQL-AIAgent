"""SalesRecord Pydantic model 的單元測試。

測試 CSV 資料匯入時的型別驗證與格式轉換。
"""

import pytest
import datetime
from pydantic import ValidationError

from db.init_db import SalesRecord


class TestSalesRecordValidation:
    """測試 SalesRecord model 的欄位驗證。"""

    @pytest.fixture
    def valid_row(self):
        """一筆合法的 CSV 資料（模擬 df.to_dict 的結果）。"""
        return {
            "Invoice ID": "750-67-8428",
            "Branch": "Alex",
            "City": "Yangon",
            "Customer type": "Member",
            "Gender": "Female",
            "Product line": "Health and beauty",
            "Unit price": 74.69,
            "Quantity": 7,
            "Tax 5%": 26.1415,
            "Sales": 548.9715,
            "Date": "1/5/2019",
            "Time": "1:08:00 PM",
            "Payment": "Ewallet",
            "cogs": 522.83,
            "gross margin percentage": 4.761904762,
            "gross income": 26.1415,
            "Rating": 9.1,
        }

    def test_valid_record(self, valid_row):
        """合法資料應成功建立 SalesRecord。"""
        record = SalesRecord(**valid_row)
        assert record.invoice_id == "750-67-8428"
        assert record.branch == "Alex"
        assert record.unit_price == 74.69
        assert record.quantity == 7

    def test_date_parsing(self, valid_row):
        """日期欄位應正確解析 M/D/YYYY 格式。"""
        record = SalesRecord(**valid_row)
        assert record.date == datetime.date(2019, 1, 5)

    def test_time_parsing(self, valid_row):
        """時間欄位應正確解析 12 小時制格式。"""
        record = SalesRecord(**valid_row)
        assert record.time == datetime.time(13, 8, 0)

    def test_date_various_formats(self, valid_row):
        """測試不同日期格式。"""
        valid_row["Date"] = "12/31/2019"
        record = SalesRecord(**valid_row)
        assert record.date == datetime.date(2019, 12, 31)

    def test_time_am(self, valid_row):
        """AM 時間應正確解析。"""
        valid_row["Time"] = "10:29:00 AM"
        record = SalesRecord(**valid_row)
        assert record.time == datetime.time(10, 29, 0)

    def test_invalid_date_raises(self, valid_row):
        """無效日期格式應拋出 ValidationError。"""
        valid_row["Date"] = "2019-01-05"
        with pytest.raises(ValidationError):
            SalesRecord(**valid_row)

    def test_invalid_time_raises(self, valid_row):
        """無效時間格式應拋出 ValidationError。"""
        valid_row["Time"] = "13:08"
        with pytest.raises(ValidationError):
            SalesRecord(**valid_row)

    def test_missing_field_raises(self, valid_row):
        """缺少必要欄位應拋出 ValidationError。"""
        del valid_row["Branch"]
        with pytest.raises(ValidationError):
            SalesRecord(**valid_row)

    def test_invalid_numeric_type(self, valid_row):
        """非數值的 unit_price 應拋出 ValidationError。"""
        valid_row["Unit price"] = "not_a_number"
        with pytest.raises(ValidationError):
            SalesRecord(**valid_row)

    def test_model_dump_keys(self, valid_row):
        """model_dump 應回傳 snake_case 欄位名稱。"""
        record = SalesRecord(**valid_row)
        dumped = record.model_dump()
        assert "invoice_id" in dumped
        assert "product_line" in dumped
        assert "Invoice ID" not in dumped
