# -*- coding: utf-8 -*-
"""
RFM 分层教学脚本：完整复刻真实工作流程（v2：修正新客误判 + 业务规则清洗）
=========================================
数据说明（诚实声明）：
  - 用户属性表：来自 E:\\archive_project\\Customer_Details.csv（你的真实文件）
  - 订单表：模拟生成（教学用）。真实工作中它由数据团队按取数需求单提供，
    到时候把本文件第 2 步删掉、读真表即可，后续代码不变。

流程（对应真实项目的 6 步）：
  1. 读用户表
  2. 准备订单数据（这里是模拟）
  3. 清洗异常（业务规则，而不是纯统计切点）
  4. 计算 R / F / M（口径写在注释里，真实工作要沉淀成口径文档）
  5. 用分位数看分布 -> 取整成业务数字定规则
  6. 分层 + 每层配动作 + 属性交叉分析
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_DATE = date(2026, 11, 5)     # 分析基准日，实际就是"今天"
USER_CSV = sys.argv[1]                # 用户属性表路径（命令行传入）

rng = np.random.default_rng(42)
T = pd.Timestamp(ANALYSIS_DATE)
D = lambda days: T - pd.Timedelta(days=days)

# ============ 1. 读用户属性表，抽一个可操作的样本 ============
users = pd.read_csv(USER_CSV, usecols=["CustomerID", "SignupDate", "Segment", "Region"])
users = users.sample(n=20000, random_state=42).reset_index(drop=True)
print(f"[1] 用户表读取成功，抽样 {len(users):,} 人做教学演示")

# ============ 2. 模拟订单表（真实工作中：换成数据团队给的订单表） ============
# 先给每个用户暗中分配一种"真实行为类型"，再按类型生成订单
types = rng.choice(
    ["core", "mid", "low", "new", "hv_churn", "mid_churn", "dormant", "fraud"],
    size=len(users),
    p=[0.08, 0.17, 0.20, 0.05, 0.03, 0.05, 0.415, 0.005],
)

def gen_orders(t):
    """按行为类型生成订单列表 [(购买日期, 金额)]
    注意：老用户类型(core/mid/low)的订单窗口从31天前开始——
    保证他们的"首单"不落在近30天，否则会被误判成新客。"""
    ticket = {  # 客单价 ~ 对数正态分布 (中位数, 波动)
        "core": (3300, 0.4), "mid": (2000, 0.4), "low": (1300, 0.4),
        "new": (1100, 0.4), "hv_churn": (4000, 0.5), "mid_churn": (2500, 0.5),
        "dormant": (1000, 0.5), "fraud": (45000, 0.1),
    }[t]
    def mk(n_order, dmin, dmax):
        days = rng.integers(dmin, dmax, size=n_order)
        amts = rng.lognormal(np.log(ticket[0]), ticket[1], size=n_order)
        return [(D(int(d)), round(float(a), 2)) for d, a in zip(days, amts)]

    if t == "core":      return mk(int(rng.poisson(8)) + 2, 31, 120)  # 高频活跃
    if t == "mid":       return mk(int(rng.poisson(2)) + 1, 31, 150)  # 中坚
    if t == "low":       return mk(int(rng.poisson(1)) + 1, 31, 60)   # 低频活跃
    if t == "new":       return mk(int(rng.integers(1, 3)), 1, 25)    # 新客
    if t == "hv_churn":  return mk(12, 100, 300)                      # 曾经大户，最近不来
    if t == "mid_churn": return mk(4, 70, 250)                        # 中坚开始流失
    if t == "fraud":     return mk(15, 1, 20)                         # 羊毛党/囤货商
    return mk(int(rng.integers(0, 3)), 200, 500)                      # 沉睡(可能一次没买)

rows = [(uid, d, amt)
        for uid, t in zip(users["CustomerID"], types)
        for d, amt in gen_orders(t)]
orders = pd.DataFrame(rows, columns=["CustomerID", "order_date", "amount"])
print(f"[2] 订单表就绪：{len(orders):,} 笔订单（模拟数据）")

# ============ 3. 清洗异常：用业务规则，不用纯统计切点 ============
# 教学点：单笔 ₹25,000 的零售订单不符合业务常识（正常客单价的中位数才 ₹1,500 左右），
# 这种线写进口径文档时有业务说服力；纯分位数切点反而说不清"为什么是这个数"。
abnormal = orders[orders["amount"] > 25000]
n_bad_users = abnormal["CustomerID"].nunique()
orders = orders[orders["amount"] <= 25000]
m_user = orders.groupby("CustomerID")["amount"].sum()
mean_before_full = None  # 对比：不清洗时的均值已不可考，这里只展示清洗后
print(f"[3] 清洗：单笔金额>₹25,000 的异常订单 {len(abnormal):,} 笔，涉及 {n_bad_users} 人")
print(f"    → 这些人转入风控黑名单流程，不参与分层（清洗也是口径的一部分！）")

# ============ 4. 计算 R / F / M（口径！口径！口径！） ============
#   R   = 距最近一次购买的天数（观察窗口 365 天）
#   F90 = 近 90 天订单数
#   M180 = 近 180 天实付金额（"现在值多少钱"）
#   M365 = 近 365 天累计实付（"曾经值多少钱"，用来抓已经不来的大户）
g = orders.groupby("CustomerID")
last, first = g["order_date"].max(), g["order_date"].min()
win90, win180 = T - pd.Timedelta(days=90), T - pd.Timedelta(days=180)

rfm = pd.DataFrame({
    "R":    (T - last).dt.days,
    "F90":  orders[orders["order_date"] > win90].groupby("CustomerID").size(),
    "M180": orders[orders["order_date"] > win180].groupby("CustomerID")["amount"].sum(),
    "M365": g["amount"].sum(),
    "first_order_days": (T - first).dt.days,
}).reindex(users["CustomerID"])
rfm["F90"] = rfm["F90"].fillna(0)
rfm["M180"] = rfm["M180"].fillna(0)
rfm["M365"] = rfm["M365"].fillna(0)

# ============ 5. 先看分布，再定规则（分位数 -> 取整） ============
print("\n[5] M180 的分布——看清楚为什么不能用平均值：")
s = rfm["M180"]
print(f"    均值 ₹{s.mean():,.0f}  vs  中位数 ₹{s.median():,.0f}  ← 均值被少数大户拉高")
for q in [0.5, 0.8, 0.85, 0.9, 0.95]:
    print(f"    {int(q*100):>2} 分位: ₹{s.quantile(q):>9,.0f}")
print("    → 规则线要参考分位数，但取整成业务数字（稳定、好沟通）")
print(f"    → 所以'高价值线'定在 ₹10,000（约等于85分位取整，覆盖最值钱的~15%用户）")

# ============ 6. 规则分层（每条线都能翻译成人话，从上到下先命中先归类） ============
conds = [
    rfm["R"].isna(),                                  # 365天窗口内一笔都没买
    rfm["first_order_days"] <= 30,                    # 首单不满30天
    (rfm["R"] <= 45) & (rfm["M180"] >= 10000),        # 花得多且近期还在
    (rfm["R"] > 45) & (rfm["M365"] >= 10000),         # 花得多但45天+没来(含彻底沉默的大户)
    (rfm["R"] <= 45) & (rfm["M180"] >= 3000),         # 消费中等、近期活跃
    (rfm["R"] > 45) & (rfm["M365"] >= 3000),          # 消费中等、在流失
    rfm["R"] <= 60,                                   # 买得少但近期还来
]
choices = ["365天未购", "新客孵化", "高价值活跃", "高价值预警",
           "中坚力量", "中坚预警", "低频活跃"]
rfm["layer"] = np.select(conds, choices, default="沉睡流失")

ACTIONS = {
    "365天未购": "真实场景下含'注册未购'：大额首购券+转化实验，主攻首单破冰",
    "新客孵化": "首购后7天黄金期推二单券，品类引导",
    "高价值活跃": "不发券(纯损毛利)！专属客服/会员权益/新品优先购",
    "高价值预警": "预算重仓：大额召回券+短信/push，必须做AB实验",
    "中坚力量": "提频提额：品类交叉推荐、满减任务、小额券",
    "中坚预警": "中额召回券+push唤回",
    "低频活跃": "低成本触达：push/弹窗、凑单券，别上短信",
    "沉睡流失": "只在大促档期批量短信，严控成本，ROI为负则放弃",
}
rfm["建议动作"] = rfm["layer"].map(ACTIONS)

# ============ 7. 输出：分层结果 + 金额贡献 + 属性交叉 ============
result = users.merge(rfm.reset_index(), on="CustomerID")
total_m = result["M180"].sum()
summary = (result.groupby("layer")
           .agg(人数=("CustomerID", "size"), M180总额=("M180", "sum"))
           .assign(人数占比=lambda x: (x["人数"] / len(result)).round(3),
                   金额占比=lambda x: (x["M180总额"] / total_m).round(3))
           .sort_values("M180总额", ascending=False))
print("\n[7] 分层结果总览（先做常识验收：排序符合业务直觉吗？）：")
print(summary[["人数", "人数占比", "金额占比"]].to_string())

print("\n    交叉分析：每个分层里都是什么身份的人？（行=分层内身份构成）")
cross = pd.crosstab(result["layer"], result["Segment"], normalize="index").round(2)
print(cross.to_string())
print("    （模拟数据的身份是随机分配的所以看不出规律；真实数据里这一步经常出现")
print("      '学生集中在低频层''游戏玩家流失率最高'这类洞察，用来决定对谁卖什么）")

out = Path(__file__).resolve().parent / "rfm_result.csv"  # 跟着脚本走：输出到脚本所在文件夹
result.to_csv(out, index=False, encoding="utf-8-sig")   # utf-8-sig：Excel打开不乱码
print(f"\n[完成] 用户级明细已导出：{out}（可用 Excel 打开筛选查看）")
