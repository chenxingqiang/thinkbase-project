# 学位论文：保真度门控的递归自改进序列推荐框架

本目录是一部完整的硕士学位论文工程：实验代码、论文源码、数字注入
管线三部分组成一个**可复现闭环**——论文中的每一个实验数字都由
脚本从实验产物 JSON 自动注入，正文不出现手写结果。

## 目录结构

```
thesis/
├── main.tex               # GB/T 7713.1 论文主文件 (xelatex + ctex)
├── refs.bib               # 参考文献（全部真实可查证）
├── chapters/              # 各章源文件
│   ├── numbers.tex        #   实验数字宏（脚本生成，勿手改）
│   └── ...
├── figures/               # TikZ/pgfplots 图（rounds/drift 由脚本生成）
├── scripts/
│   ├── fill_numbers.py    # results/*.json -> chapters/numbers.tex
│   └── make_figures.py    # results/*.json -> figures/{rounds,drift}.tex
└── experiments/
    ├── run_experiments.py # 全量实验驱动（一条命令复现全文数字）
    ├── rsix/              # 实验包（数据/模型/指标/门控/生成/编排）
    ├── tests/test_rsix.py # 实验包单元测试（确定性/正确性/预算）
    ├── data/u.data        # MovieLens-100K（不入库）
    └── results/           # 实验产物 JSON（含 ELL 审计轨迹）
```

## 复现流程

```bash
# 1. 数据：MovieLens-100K (ml-100k.zip -> u.data 放入 experiments/data/)
# 2. 实验（CPU，约 8 小时；--quick 为 30 分钟冒烟）
cd thesis/experiments
python run_experiments.py

# 3. 数字注入 + 图表生成
python ../scripts/fill_numbers.py
python ../scripts/make_figures.py

# 4. 编译论文（需要 xelatex + ctex；使用 fandol 字体集）
latexmk -xelatex main.tex    # 或 xelatex main.tex ×3 遍

# 5. 单元测试
python -m pytest tests/test_rsix.py -q
```

## 可信性设计（诚实性红线）

- **种子固定**：模型初始化、扰动视图、批次顺序全部由单一种子
  派生；同种子两次运行结果逐位一致（`test_gru4rec_deterministic`）。
- **全量排序评估**：无负采样，测试物品与全部未交互物品竞争
  （`rank_of_target`）。
- **数字不落地正文**：正文引用 `\OursNdcg` 等宏，宏值由
  `fill_numbers.py` 从 results JSON 生成。
- **每条实验为指令流**：每轮 生成→门控→混合→重训→评估 各为一条
  可审计指令，并导出 ELL 轨迹存 `results/rsi_gated_trace.jsonl`。
- **失败模式先行**：naive 自训练的崩塌（动机实验）与 gated 的稳定
  性在同一协议下同种子对比。
- **局限如实报告**：论文第 4/5 章明确列出增益上界、生成开销、
  预算刻画力与单数据集验证的局限。
