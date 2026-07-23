# Peptide OmniPanel V31 中文批量网站升级说明

> 日期：2026-07-23
> 状态：**已部署并通过公网浏览器端到端测试**
> 公网临时地址：<https://3ea53ebc88ba6b8cb5.gradio.live>
> 本地地址：<http://127.0.0.1:7860>

## 1. 本轮升级结果

网站已从“左侧输入、右侧结果”的单页面改成两个独立工作页面：

```text
① 序列输入页
  ├─ 单条序列
  ├─ 多条序列
  ├─ FASTA 文本
  ├─ FASTA/FA/FAA/TXT/CSV/TSV 文件
  └─ 提交批次
       ↓ 自动切换
② 预测面板
  ├─ 右上角实时任务进展
  ├─ 每条序列的任务状态与 0/18 → 18/18 进度
  ├─ 已完成序列选择器
  ├─ 当前序列 18 端点卡片和完整结果
  ├─ 批量汇总表
  └─ JSON/TSV/ZIP 下载
```

用户提交以后，网页默认自动跳转到“② 预测面板”，不需要再手动寻找结果页面。预测面板右上角固定显示已完成任务数、总任务数、当前序列、进度条和完成百分比。

## 2. 中文化范围

除不可避免的技术专名、端点缩写和标准单位外，界面文字均改为中文，包括：

- 页面、按钮、选项卡、文件上传和输入说明；
- 任务状态、任务进度、失败信息和批次摘要；
- 输出类型：`概率 / 排序分数 / 连续值回归`；
- 特征路线：传统肽序列特征、ESM2 序列表征、二维结构特征、多模态融合；
- 证据通道和 18 个端点的条件合同；
- `RESEARCH ONLY` → `仅供科研`；
- `LOW CONFIDENCE` → `低置信度`；
- `NOT VALIDATED` → `未通过独立外部验证`；
- 下载、证据限制和修饰敏感性说明。

保留的专名包括 ESM2、ADMET、PK、LogD、F、T₁/₂、PPB、CL、Vd、BBB、HC50、FASTA、CSV、TSV、HELM、SMILES、Morgan、RDKit、JSON、SHA、OOF、ROC-AUC、MAE、RMSE 等。

## 3. 输入合同

### 3.1 文本输入

支持以下四种形式。

单条序列：

```text
GIGKFLHSAKKFGKAFVGEIMNS
```

每行一条：

```text
GIGKFLHSAKKFGKAFVGEIMNS
RRWWRF
ACDEFGHIK
```

名称 + Tab + 序列：

```text
候选肽A	GIGKFLHSAKKFGKAFVGEIMNS
候选肽B	RRWWRF
```

多记录 FASTA：

```fasta
>候选肽A
GIGKFLHSAKKFGKAFVGEIMNS
>候选肽B
RRWWRF
```

### 3.2 文件上传

支持：

```text
.fasta  .fa  .faa  .txt  .csv  .tsv
```

CSV/TSV 会自动识别以下序列列名：

```text
sequence, seq, peptide, peptide_sequence,
amino_acid_sequence, 序列, 肽序列, 氨基酸序列
```

名称列支持：

```text
id, name, identifier, sequence_id, peptide_id,
名称, 编号, 序列编号
```

### 3.3 安全边界

- 一个批次最多 32 条有效序列；
- 上传文件最大 1 MB；
- 仅接受 UTF-8 文本；
- 每条序列最多 1,024 aa；
- 仅接受 20 种标准氨基酸字母；
- 无效行不会伪装成成功任务，会在任务表中列为“输入无效”并显示原因；
- 文本和文件可以同时提交；
- 同一批次中完全相同的序列会复用已计算结果，但仍保留独立任务记录。

## 4. 任务进度设计

预测面板包含两层进度。

### 4.1 批次级进度

右上角任务卡显示：

- 已完成任务数 / 有效任务总数；
- 当前正在处理的序列名称；
- 当前处理阶段；
- 总体完成百分比；
- 完成后变为“本批次已执行完毕”。

### 4.2 序列级任务表

每条输入序列对应一个任务，状态包括：

```text
排队中 → 预测中 → 已完成
                 ↘ 失败
输入不合法 → 输入无效
```

任务表字段为：

```text
任务、序列名称、来源、长度、状态、进度、用时、信息
```

模型目前一次性生成一条序列的共同 ESM2/传统/二维结构特征，再依次运行 18 个本地预测头。因此单条任务在界面中显示“正在提取 ESM2 特征”，完成后从 `0 / 18` 更新为 `18 / 18`，不会伪造不存在的逐端点耗时。

## 5. 批量结果查看

每完成一条序列，该序列会立即出现在“选择已完成的肽序列”下拉框中。用户不必等整个批次结束即可查看首条结果。

预测面板提供：

1. 端点概览：18 张端点卡片；
2. 当前序列完整结果：值、单位、OOF、证据、特征路线、条件合同；
3. 分类/排序图：只显示可放在 0–1 轴上的概率或排序分数；
4. 批量汇总表：每条完成序列产生 18 行；
5. 证据与限制；
6. 当前序列原始 JSON；
7. 批量下载。

批量下载包括：

- JSON：批次、任务、拒绝输入、全部模型结果和警告；
- TSV：可直接用于 Excel、R 或 Python 的扁平汇总表；
- ZIP：批量 JSON、批量 TSV，以及每条成功序列的独立 JSON。

## 6. 真实验证

### 6.1 自动测试

```text
13 passed
ruff: pass
py_compile: pass
```

覆盖：

- 单条和多条文本；
- 名称 + Tab + 序列；
- 多记录/跨行 FASTA；
- 中文列名 CSV；
- peptide_sequence 列 TSV；
- 无效氨基酸拒绝；
- 32 条批次边界；
- 中文结果字段；
- 18 行单序列结果和 36 行双序列汇总；
- JSON/TSV/ZIP 导出；
- 序列切换、任务表和进度面板。

### 6.2 真实 RTX 5080 双序列批次

```text
任务数：2
成功结果：2
每条端点数：[18, 18]
每条已验证端点数：[0, 0]
批量汇总行数：36
真实任务用时：[2.78 秒, 0.36 秒]
导出文件：JSON + TSV + ZIP
状态：pass
```

证据：

```text
.omx/web_validation/v31_batch_real_smoke.log
```

### 6.3 公网浏览器端到端验证

使用 Windows Chrome + Selenium 在真实公网地址完成：

1. 打开网站；
2. 输入两条 FASTA 序列；
3. 点击提交；
4. 自动切换到“② 预测面板”；
5. 等待 2/2 完成；
6. 检查任务表；
7. 检查 18 张端点卡片；
8. 检查中文“仅供科研”披露。

结果：

```text
selected_tabs: [② 预测面板, 端点概览]
endpoint_card_count: 18
contains_batch_complete: true
contains_chinese_research_only: true
status: pass
```

截图：

```text
.omx/web_validation/v31_batch_input_page.png
.omx/web_validation/v31_batch_prediction_page.png
```

### 6.4 公网文件上传验证

另使用真实 Chrome 通过公网页面上传 `public_upload_test.fasta`，网站正确识别 `uploaded_A` 和 `uploaded_B`，自动切换到预测面板，并在任务表中显示来源文件名。结果：`pass`。

```text
.omx/web_validation/selenium_file_upload_ui.log
```

## 7. 当前服务

```text
公网：https://3ea53ebc88ba6b8cb5.gradio.live
本地：http://127.0.0.1:7860
tmux：peptide_v31_web_20260723
日志：.omx/logs/peptide_v31_web_20260723.log
```

公网链接为无认证临时链接，约一周有效；停止本机、WSL、tmux 或隧道后会立即失效。未公开或敏感序列不应通过无认证公网链接提交。

机器可读部署记录：

```text
docs/peptide_omnipanel_v31_web_deployment_20260723.json
```
