# -*- coding: utf-8 -*-
"""
流失阈值的科学定法：不拍脑袋，也不拿 R 中位数凑数
====================================================
原理：一个用户"多久没买算流失"，应该挂在他的**自然复购节奏**上：
  1. 算每个用户相邻两单的间隔天数（复购间隔）
  2. 取分布的中位数 / P75 作为"一个正常周期"
  3. 流失阈值 = 周期 × 2~3 —— 超过一两个完整周期没回来，才算真流失

为什么不能用 R 中位数（77天）？因为那是"全体用户距最后一单多久"的快照，
里面混着活跃用户和已经流失的用户——它不是复购周期，拿它当阈值
会把正处在正常购买间隔中的老客误判成流失。
"""
import pandas as pd
#读取数据
DATA = "E:/archive_project/Sales_Data.csv"
df = pd.read_csv(DATA, usecols=["CustomerID", "OrderDate", "OrderStatus"])
#筛选：送达订单
df = df[df["OrderStatus"] == "Delivered"]                    # 口径：只算送达
#转变日期格式
df["OrderDate"] = pd.to_datetime(df["OrderDate"], format="%d-%m-%Y")
df = df.sort_values(["CustomerID", "OrderDate"])

# 复购间隔：同一用户按时间排序后，相邻两单 diff() 出天数
gaps = df.groupby("CustomerID")["OrderDate"].diff().dt.days.dropna()

print(f"复购间隔样本：{len(gaps):,} 个（每个至少买过2单的用户贡献）")
print(f"  中位数 {gaps.median():.0f} 天 | 均值 {gaps.mean():.0f} 天 | P75 {gaps.quantile(0.75):.0f} 天")
print()
med = gaps.median()
print(f"→ 流失阈值建议 = 复购周期中位数 × 2~3 = {med*2:.0f} ~ {med*3:.0f} 天")
print(f"→ 自洽检查：分层里的'预警线'（R>60）≈ {60/med:.1f} 个周期，作为早期预警合理；")
print(f"  流失线必须明显大于预警线，形成阶梯：预警(60) < 流失({med*2:.0f}~{med*3:.0f})")
