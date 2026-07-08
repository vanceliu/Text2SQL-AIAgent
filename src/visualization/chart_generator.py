"""圖表生成模組。

根據查詢結果和指定的圖表類型，使用 matplotlib 生成對應的圖表，
儲存為圖片檔案並回傳路徑。
"""

from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Noto Sans CJK TC", "Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False


def generate_chart(
    data: list[dict[str, Any]],
    chart_type: str,
    intent: Optional[dict] = None,
) -> Optional[str]:
    """根據查詢結果生成圖表並儲存為圖片檔案。

    Args:
        data: SQL 查詢結果，list of dict 格式。
              每個 dict 代表一列資料，key 為欄位名稱。
        chart_type: 圖表類型，"bar" | "line" | "pie"。
        intent: 意圖解析結果字典，用於推斷圖表標題。

    Process:
        1. 從 data 中辨識分類欄位（第一個字串型欄位）和數值欄位（第一個數值型欄位）
        2. 根據 chart_type 呼叫對應的繪圖函式
        3. 設定中文標題、座標軸標籤
        4. 儲存為 PNG 圖片至 output/ 目錄
        5. 回傳圖片檔案路徑

    Returns:
        Optional[str]: 成功時回傳圖片檔案路徑（如 "output/chart_001.png"），
                       失敗時回傳 None。
    """
    if not data:
        return None

    output_dir = Path(__file__).parent.parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_charts = list(output_dir.glob("chart_*.png"))
    chart_num = len(existing_charts) + 1
    output_path = output_dir / f"chart_{chart_num:03d}.png"

    columns = list(data[0].keys())
    label_col, value_col = _identify_columns(data, columns)

    if not label_col or not value_col:
        return None

    labels = [str(row.get(label_col, "")) for row in data]
    values = [float(row.get(value_col, 0)) for row in data]

    title = _generate_title(intent, label_col, value_col)

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar":
            _draw_bar_chart(ax, labels, values, title, label_col, value_col)
        elif chart_type == "line":
            _draw_line_chart(ax, labels, values, title, label_col, value_col)
        elif chart_type == "pie":
            _draw_pie_chart(ax, labels, values, title)
        else:
            _draw_bar_chart(ax, labels, values, title, label_col, value_col)

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        return str(output_path)

    except Exception:
        plt.close("all")
        return None


def _identify_columns(data: list[dict], columns: list[str]) -> tuple[Optional[str], Optional[str]]:
    """從資料中辨識分類欄位和數值欄位。

    Args:
        data: 查詢結果資料。
        columns: 欄位名稱列表。

    Process:
        1. 遍歷欄位，第一個值為字串型的作為分類欄位
        2. 第一個值為數值型的作為數值欄位
        3. 若只有兩個欄位，直接分配

    Returns:
        tuple[Optional[str], Optional[str]]: (分類欄位名, 數值欄位名)，
        找不到時對應位置為 None。
    """
    if len(columns) < 2:
        if len(columns) == 1:
            return None, columns[0]
        return None, None

    first_row = data[0]
    label_col = None
    value_col = None

    for col in columns:
        val = first_row.get(col)
        if val is None:
            continue
        if isinstance(val, str) and label_col is None:
            label_col = col
        elif isinstance(val, (int, float)) and value_col is None:
            value_col = col

    if label_col is None:
        label_col = columns[0]
    if value_col is None or value_col == label_col:
        for col in columns:
            if col != label_col:
                value_col = col
                break

    return label_col, value_col


def _generate_title(intent: Optional[dict], label_col: str, value_col: str) -> str:
    """根據意圖資訊生成圖表標題。

    Args:
        intent: 意圖解析結果，可能為 None。
        label_col: 分類欄位名稱。
        value_col: 數值欄位名稱。

    Process:
        若有 intent 中的 description，使用它作為標題；
        否則用「{value_col} by {label_col}」格式。

    Returns:
        str: 圖表標題文字。
    """
    if intent and intent.get("description"):
        return intent["description"]
    return f"{value_col} by {label_col}"


def _draw_bar_chart(ax, labels: list[str], values: list[float],
                    title: str, x_label: str, y_label: str) -> None:
    """繪製長條圖。

    Args:
        ax: matplotlib Axes 物件。
        labels: X 軸標籤列表。
        values: Y 軸數值列表。
        title: 圖表標題。
        x_label: X 軸標籤名稱。
        y_label: Y 軸標籤名稱。

    Process:
        繪製垂直長條圖，設定顏色漸層、標題和座標軸標籤，
        並在每個長條上方標註數值。

    Returns:
        None（直接修改 ax 物件）。
    """
    colors = plt.cm.Set3([i / len(labels) for i in range(len(labels))])
    bars = ax.bar(labels, values, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=14, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,.2f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=45, ha="right")


def _draw_line_chart(ax, labels: list[str], values: list[float],
                     title: str, x_label: str, y_label: str) -> None:
    """繪製折線圖。

    Args:
        ax: matplotlib Axes 物件。
        labels: X 軸標籤列表（通常為時間序列）。
        values: Y 軸數值列表。
        title: 圖表標題。
        x_label: X 軸標籤名稱。
        y_label: Y 軸標籤名稱。

    Process:
        繪製折線圖並標記資料點，設定標題和座標軸標籤。

    Returns:
        None（直接修改 ax 物件）。
    """
    ax.plot(labels, values, marker="o", linewidth=2, markersize=6, color="#2196F3")
    ax.fill_between(range(len(labels)), values, alpha=0.1, color="#2196F3")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, alpha=0.3)


def _draw_pie_chart(ax, labels: list[str], values: list[float], title: str) -> None:
    """繪製圓餅圖。

    Args:
        ax: matplotlib Axes 物件。
        labels: 各切片的標籤列表。
        values: 各切片的數值列表。
        title: 圖表標題。

    Process:
        繪製圓餅圖，顯示百分比標籤，突出最大切片。

    Returns:
        None（直接修改 ax 物件）。
    """
    max_idx = values.index(max(values))
    explode = [0.05 if i == max_idx else 0 for i in range(len(values))]

    colors = plt.cm.Set3([i / len(labels) for i in range(len(labels))])

    ax.pie(
        values, labels=labels, explode=explode, colors=colors,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10},
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
