# -*- coding: utf-8 -*-
"""
真实数据 RFM 分层 + 流转矩阵 —— E:\\archive_project\\Sales_Data.csv (97.5万笔真实订单)
====================================================================================
【口径文档】——真实工作中，这段要先写下来跟数据/业务方对齐：
  1. 分析对象：2023-01-01 ~ 2024-12-31 期间有购买的用户（143,509 人）
  2. 有效订单：OrderStatus == 'Delivered'。Cancelled(未成交)/Returned(已退货) 均不计入 F/M
  3. 金额口径：NetSales（实付货值，不含 GST 和运费）
  4. 分析基准日：2024-12-31（=数据中最大订单日期，不是"今天"！）
  5. 指标定义：
       R   = 基准日 - 最近一次有效购买日期（天数）
       F90 = 近 90 天有效订单数
       M180 = 近 180 天实付金额（"现在"的价值）
       M365 = 近 365 天实付金额（"曾经"的价值，用于抓流失大户）
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = "E:/archive_project/Sales_Data.csv"
ASOF = pd.Timestamp("2024-12-31")          # 分析基准日
COLS = ["OrderDate", "CustomerID", "OrderStatus", "NetSales", "CustomerSegment"]

# ---------- 1. 读数 + 口径过滤 ----------
df = pd.read_csv(DATA, usecols=COLS, parse_dates=["OrderDate"], date_format="%d-%m-%Y")
n0 = len(df)
df = df[df["OrderStatus"] == "Delivered"].copy()
print(f"[1] 原始 {n0:,} 笔 → 有效订单 {len(df):,} 笔"
      f"（剔除 Cancelled/Returned 共 {n0-len(df):,} 笔）")
print(f"    用户数 {df['CustomerID'].nunique():,}，日期范围 "
      f"{df['OrderDate'].min().date()} → {df['OrderDate'].max().date()}")

# ---------- 2. 分层规则（输入任意基准日即可复用——流转矩阵全靠这一点） ----------
def layer_of(orders: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """对截至 asof 的订单，给每个用户打分层。阈值取整后的业务数字。"""
    g = orders.groupby("CustomerID")
    last, first = g["OrderDate"].max(), g["OrderDate"].min()
    w90, w180 = asof - pd.Timedelta(days=90), asof - pd.Timedelta(days=180)
    x = pd.DataFrame({
        "R":    (asof - last).dt.days,
        "M180": orders[orders["OrderDate"] > w180].groupby("CustomerID")["NetSales"].sum(),
        "M365": orders[orders["OrderDate"] > asof - pd.Timedelta(days=365)]["NetSales"]
                .groupby(orders["CustomerID"]).sum(),
        "new_age": (asof - first).dt.days,
    })
    # 分层规则：从上到下先命中先归类；每条线都能翻译成人话
    conds = [
        x["new_age"] <= 30,                                   # 首单不满30天 → 新客
        (x["R"] <= 60) & (x["M180"] >= 15000),                # 近期还在且近半年花得多
        (x["R"] > 60) & (x["M365"] >= 15000),                 # 花得多但2个月没来
        (x["R"] <= 60) & (x["M180"] >= 5000),                 # 中等消费、近期活跃
        (x["R"] > 60) & (x["M365"] >= 5000),                  # 中等消费、在流失
        x["R"] <= 90,                                         # 买得少但3个月内还来
    ]
    x["layer"] = np.select(conds, [
        "新客孵化", "高价值活跃", "高价值预警", "中坚力量", "中坚预警", "低频活跃",
    ], default="沉睡流失")
    return x

x = layer_of(df, ASOF)

# ---------- 3. 常识验收：先看分布和分层结果合不合业务直觉 ----------
print("\n[3] 基准日分布体检：")
print(f"    R 中位数 {x['R'].median():.0f} 天 | M180 中位数 ₹{x['M180'].median():,.0f}"
      f" | M365 中位数 ₹{x['M365'].median():,.0f}")
for q in [0.8, 0.85, 0.9]:
    print(f"    M180 {int(q*100)} 分位 ₹{x['M180'].quantile(q):,.0f}"
          f" | M365 {int(q*100)} 分位 ₹{x['M365'].quantile(q):,.0f}")

result = x.reset_index().rename(columns={"index": "CustomerID"})
total_m = result["M180"].sum()
summary = (result.groupby("layer")
           .agg(人数=("CustomerID", "size"), M180总额=("M180", "sum"))
           .assign(人数占比=lambda t: (t["人数"] / len(result)).round(3),
                   金额占比=lambda t: (t["M180总额"] / total_m).round(3))
           .sort_values("M180总额", ascending=False))
print("\n[4] 分层结果（2024-12-31 快照）：")
print(summary[["人数", "人数占比", "金额占比"]].to_string())

# ---------- 5. 属性交叉：每个分层里都是什么人 ----------
seg = df.drop_duplicates("CustomerID").set_index("CustomerID")["CustomerSegment"]
cross = pd.crosstab(result["layer"], result["CustomerID"].map(seg), normalize="index").round(2)
print("\n[5] 分层 × 身份构成（行归一）：")
print(cross.to_string())

# ---------- 6. 流转矩阵：上半年快照 vs 年末快照 ----------
# 教学点：分层不是拍一次快照就完，用户会流动。看"从哪来、到哪去"比静态分布更有预警价值。
MID = pd.Timestamp("2024-06-30")
prev = layer_of(df[df["OrderDate"] <= MID], MID)[["layer"]].rename(columns={"layer": "半年前"})
now = x[["layer"]].rename(columns={"layer": "年末"})
flow = prev.join(now, how="outer")
flow_order = ["新客孵化", "高价值活跃", "高价值预警", "中坚力量",
              "中坚预警", "低频活跃", "沉睡流失"]
flow["半年前"] = flow["半年前"].fillna("新用户(上半年无购买)")   # 下半年才有首单的人
# 口径要点：'真流失'要单独判——下半年一笔有效订单都没有的人直接判流失，
# 不能靠分层规则兜底（否则会被'沉睡流失'层吸收，看不到真实流失率）
h2_users = df.loc[df["OrderDate"] > MID, "CustomerID"].unique()
flow["年末"] = np.where(np.isin(flow.index, h2_users), flow["年末"], "已流失(下半年无购买)")
order_full = ["新用户(上半年无购买)"] + flow_order
print("\n[6] 流转矩阵（行=2024-06-30 所在层，列=2024-12-31 所在层，单位：人）：")
mat = pd.crosstab(flow["半年前"], flow["年末"])
mat = mat.reindex(index=order_full, columns=order_full + ["已流失(下半年无购买)"]).fillna(0).astype(int)
print(mat.to_string())

# ---------- 7. 从流转矩阵算经营指标（所有比率=对矩阵做除法） ----------
row_tot = mat.sum(axis=1)                                # 每层"半年前"的总人数（行合计）
kpi = pd.DataFrame({
    "半年前人数": row_tot,
    "流失率": (mat["已流失(下半年无购买)"] / row_tot).round(3),
})
# 沉睡层的自然回流率：行总计里，既没流失、也没原地沉睡的，就是回来买的人
sl = mat.loc["沉睡流失"]
kpi.loc["沉睡流失", "自然回流率"] = round(
    1 - (sl["已流失(下半年无购买)"] + sl["沉睡流失"]) / row_tot["沉睡流失"], 3)
# 预警层自行回春率：没做任何干预，自己又买回高价值的比例
kpi.loc["高价值预警", "自行回春率"] = round(
    mat.loc["高价值预警", "高价值活跃"] / row_tot["高价值预警"], 3)
print("\n[7] 各层流转KPI（就是拿上一张矩阵逐行做除法）：")
print(kpi.to_string())

top = mat["高价值活跃"]                                    # 年末"高价值活跃"这列=该层的来源
print("\n    年末高价值活跃层从哪来（列内占比）：")
print((top / top.sum()).round(3).sort_values(ascending=False).to_string())
print(f"    → 原地留住的只有 {mat.loc['高价值活跃','高价值活跃']/top.sum():.1%}，"
      f"其余都是半年内从别的层升上来的")

# ---------- 8. 导出 ----------
out = Path(__file__).resolve().parent / "rfm_real_result.csv"  # 跟着脚本走：输出到脚本所在文件夹
result.merge(df.drop_duplicates("CustomerID")[["CustomerID", "CustomerSegment"]],
             on="CustomerID").to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n[完成] 用户级明细：{out}")
