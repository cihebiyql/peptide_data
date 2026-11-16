#!/usr/bin/env python3
"""OmniGrid V32 web app — 多肽性质全景预测平台（苹果风·科研简洁设计，V31 设计语言延续）。

单分子: 序列和/或 SMILES → 55 端点(3-seed) + 机制锚定半衰期。
批量  : 上传 txt/csv/tsv/xlsx 或粘贴 → 逐分子预测 → 预览 + CSV 下载。
ESM2-650M (cuda:1, 第二块 RTX 4090) + node23 omnigrid 推理服务(经 node1 反向隧道)。
internal_research_only。
"""
import csv, io, json, os, re, tempfile, time, urllib.parse, urllib.request
import gradio as gr
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

SNAP = '/var/www/peptide-omnipanel/esm2_650m'
DEVICE = os.environ.get('OMNIGRID_DEVICE', 'cuda:1')
API = 'http://127.0.0.1:8901'
MAX_BATCH = 200

SEMAGLUTIDE_SMILES = 'CC[C@H](C)[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCCCNC(=O)COCCOCCNC(=O)COCCOCCNC(=O)CC[C@@H](NC(=O)CCCCCCCCCCCCCCCCC(=O)O)C(=O)O)NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](CCC(N)=O)NC(=O)CNC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@H](CO)NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](NC(=O)CNC(=O)[C@H](CCC(=O)O)NC(=O)C(C)(C)NC(=O)[C@@H](N)Cc1cnc[nH]1)[C@@H](C)O)[C@@H](C)O)C(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CC(C)C)C(=O)N[C@H](C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)O)C(C)C'
TIRZEPATIDE_SMILES = 'CC[C@H](C)[C@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](C)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](C)NC(=O)[C@@H](NC(=O)[C@H](CCCCNC(=O)CCCCCCCCCCCCCCCCCCC(=O)N[C@@H](CCC(=O)NCCOCCOCC(=O)NCCOCCOCC(=O)O)C(=O)O)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CC(C)C)NC(=O)C(C)(C)NC(=O)[C@@H](NC(=O)[C@H](CO)NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](NC(=O)CNC(=O)[C@H](CCC(=O)O)NC(=O)C(C)(C)NC(=O)[C@@H](N)Cc1ccc(O)cc1)[C@@H](C)O)[C@@H](C)O)[C@@H](C)CC)[C@@H](C)CC)C(C)C)C(=O)N[C@@H](C)C(=O)NCC(=O)NCC(=O)N1CCC[C@H]1C(=O)N[C@@H](CO)C(=O)N[C@@H](CO)C(=O)NCC(=O)N[C@@H](C)C(=O)N1CCC[C@H]1C(=O)N1CCC[C@H]1C(=O)N1CCC[C@H]1C(=O)N[C@@H](CO)C(N)=O'

CN = {
    'half_life': '半衰期 T½', 'PPB': '血浆蛋白结合', 'F': '绝对生物利用度', 'Cmax': '峰浓度',
    'LogD7.4': '分布系数 LogD(7.4)', 'solubility': '溶解度', 'PAMPA': 'PAMPA 被动渗透',
    'Caco2': 'Caco-2 渗透', 'MDCK': 'MDCK 渗透', 'RRCK': 'RRCK 渗透', 'Papp': '表观渗透 Papp',
    'permeability': '渗透性(汇总)', 'BBB': '血脑屏障透过', 'MIC': 'MIC 最小抑菌浓度',
    'MIC50': 'MIC50', 'MIC90': 'MIC90', 'MBC': 'MBC 最小杀菌浓度', 'MFC': 'MFC 最小杀真菌浓度',
    'EC50': 'EC50', 'IC50': 'IC50', 'IC50_inhibitor': 'IC50(抑制剂)', 'Ki': 'Ki', 'Kd': 'Kd',
    'Km': 'Km', 'AC50': 'AC50', 'GI50': 'GI50', 'LC_lemma': 'LC(lemna)', 'LC50': 'LC50',
    'LD50': 'LD50', 'CC50': 'CC50', 'cytotoxicity_CC50': '细胞毒性 CC50',
    'cytotoxicity_EC50': '细胞毒性 EC50', 'cytotoxicity_IC50': '细胞毒性 IC50',
    'cytotoxicity_LC50': '细胞毒性 LC50', 'hemolysis': '溶血', 'hemolysis_EC50': '溶血 EC50',
    'hemolysis_HC10': '溶血 HC10', 'hemolysis_HC50': '溶血 HC50', 'hemolysis_LC50': '溶血 LC50',
    'hemolysis_MHC': '溶血 MHC', 'hemolysis_MHC10': '溶血 MHC10', 'hemolysis_SOURCE_DEFINED_BINARY': '溶血(来源定义)',
    'MHC': 'MHC 溶血', 'MHC10': 'MHC10 溶血', 'HC10': 'HC10 溶血', 'HC50': 'HC50 溶血',
    'overall_peptide_toxicity': '综合毒性', 'CL_per_kg': '清除率 / 体重', 'CL_per_mg_protein': '清除率 / 蛋白',
    'CL_renal_per_kg': '肾清除率 / 体重', 'Vdss_per_kg': '稳态分布容积 / 体重',
}
CAT_OF = {}
for _ep in ['half_life','CL_per_kg','CL_per_mg_protein','CL_renal_per_kg','Vdss_per_kg','PPB','F','Cmax','LogD7.4']:
    CAT_OF[_ep] = '药代 PK'
for _ep in ['PAMPA','Caco2','MDCK','RRCK','Papp','permeability','BBB']:
    CAT_OF[_ep] = '渗透性'
for _ep in ['MIC','MIC50','MIC90','MBC','MFC','EC50','IC50','IC50_inhibitor','Ki','Kd','Km','AC50','GI50','LC_lemma','LC50']:
    CAT_OF[_ep] = '抗菌·抑制活性'
for _ep in ['hemolysis','hemolysis_EC50','hemolysis_HC10','hemolysis_HC50','hemolysis_LC50','hemolysis_MHC',
            'hemolysis_MHC10','hemolysis_SOURCE_DEFINED_BINARY','MHC','MHC10','HC10','HC50','CC50','LD50',
            'cytotoxicity_CC50','cytotoxicity_EC50','cytotoxicity_IC50','cytotoxicity_LC50','overall_peptide_toxicity']:
    CAT_OF[_ep] = '毒性·溶血'
CAT_OF['solubility'] = '理化·其他'
CAT_ORDER = ['药代 PK', '渗透性', '抗菌·抑制活性', '毒性·溶血', '理化·其他']
# 低饱和科研配色：冷(青/蓝) + 暖(琥珀/玫红) 结合
CHIP = {'药代 PK': ('#e4f4ef', '#0c6b5c'), '渗透性': ('#e2eff7', '#175d80'),
        '抗菌·抑制活性': ('#faf1e3', '#8a5a14'), '毒性·溶血': ('#faecec', '#8d4444'),
        '理化·其他': ('#eef1f4', '#4f5f6e')}

# ---------------- ESM2 ----------------
print('loading ESM2-650M on', DEVICE, flush=True)
_tok = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)
_model = AutoModel.from_pretrained(SNAP, local_files_only=True, dtype=torch.bfloat16).eval().to(DEVICE)
print('esm2 ready | vram %.1f GB' % (torch.cuda.memory_allocated(DEVICE) / 1e9), flush=True)

@torch.no_grad()
def esm2_embed(seq):
    """last_hidden_state 残基均值（排除全部特殊/填充 token）——与训练合同一致。"""
    enc = _tok(seq, return_tensors='pt', truncation=True, max_length=1022).to(DEVICE)
    hidden = _model(**enc).last_hidden_state[0].float()
    mask = enc['attention_mask'][0].bool()
    L = int(mask.sum()); m = mask.clone(); m[0] = False
    if L > 1: m[L - 1] = False
    return hidden[m].mean(0).cpu().numpy()

# ---------------- API ----------------
def call_api(payload, timeout=900, retries=2):
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(API + '/predict', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except Exception as ex:
            last = ex; time.sleep(1.5)
    raise RuntimeError(f'推理服务暂不可达: {last}')

def build_payload(seq, smi, species):
    payload = {'sequence': seq, 'smiles': smi}
    if species:
        payload['covariates'] = {'rbc_species_context:human': 1.0 if species.startswith('人') else 0.0,
                                 'rbc_species_context:horse': 1.0 if species.startswith('马') else 0.0,
                                 'rbc_species_context:mouse': 1.0 if species.startswith('鼠') else 0.0}
    if seq:
        payload['esm2'] = esm2_embed(seq).tolist()
    return payload

def parse_rows(r, name=''):
    """service 响应 → 展示行列表 [(类别, 端点名, 值, 单位, seeds±SD, 训练n, CV/说明, is_anchor)]"""
    out, anchor_note = [], ''
    for p in r.get('predictions', []):
        ep = p['endpoint']
        if ep == 'half_life_anchor':
            rng = p.get('range_h') or []
            rng_s = f"{rng[0]}–{rng[1]}" if rng else '—'
            out.append(['药代 PK', '⭐ 半衰期 T½ · 机制锚定', rng_s, 'h（文献锚定区间）',
                        '类锚定', p.get('acylation_class', ''), '', True])
            anchor_note = p.get('note', '')
            continue
        cat = CAT_OF.get(ep, '理化·其他')
        name_ep = CN.get(ep, ep)
        if ep == 'half_life':
            name_ep = '半衰期 T½ · ≥1h 概率' if p['kind'] == 'classification' else '半衰期 T½ · 回归'
        val = p.get('value')
        val_s = f'{val:.4g}' if isinstance(val, (int, float)) else '外推越界'
        cv = p.get('best_cv')
        cv_s = f'{cv:.3f}' if isinstance(cv, (int, float)) else '—'
        seeds_s = f"{p['n_seeds']}±{p['sd_seeds']:.3g}" if p.get('n_seeds', 0) > 1 else '1'
        ntr = str(p.get('n_train_records') or '—')
        out.append([cat, name_ep, val_s, p.get('unit', ''), seeds_s, ntr, cv_s, False])
    rank = {c: i for i, c in enumerate(CAT_ORDER)}
    out.sort(key=lambda x: (0 if x[7] else 1, rank.get(x[0], 9), x[1]))
    return out, anchor_note

def table_html(rows, max_rows=400, sticky=True):
    head = ''.join(f'<th>{h}</th>' for h in ['类别', '端点', '预测值', '单位', 'seeds ± SD', '训练 n', 'CV'])
    body = []
    for x in rows[:max_rows]:
        cat, ep, val, unit, seeds, ntr, cv, is_anchor = x
        bg, fg = CHIP.get(cat, CHIP['理化·其他'])
        style = 'class="anchor-row"' if is_anchor else ''
        body.append(f'<tr {style}><td><span class="chip" style="background:{bg};color:{fg}">{cat}</span></td>'
                    f'<td class="ep">{ep}</td><td class="val">{val}</td><td class="unit">{unit}</td>'
                    f'<td>{seeds}</td><td>{ntr}</td><td>{cv}</td></tr>')
    more = f'<tr><td colspan="7" class="more">… 仅预览前 {max_rows} 行，完整结果见 CSV 下载</td></tr>' if len(rows) > max_rows else ''
    return (f'<div class="res-wrap"><table class="res-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}{more}</tbody></table></div>')

def summarize(r, dt, anchor_note):
    skipped = r.get('skipped', [])
    pend = [s['endpoint'] for s in skipped if s['reason'] == 'conditional_endpoint_pending']
    miss = [s['endpoint'] for s in skipped if s['reason'] != 'conditional_endpoint_pending']
    html = (f'<div class="cards">'
            f'<div class="card"><span>可用端点</span><strong>{r["n_ok"]}</strong></div>'
            f'<div class="card"><span>推理耗时</span><strong>{dt:.1f}s</strong></div>'
            f'<div class="card"><span>条件端点待部署</span><strong>{len(pend)}</strong></div>'
            f'<div class="card"><span>输入不适用</span><strong>{len(miss)}</strong></div></div>')
    if anchor_note:
        html += (f'<div class="assumption"><b>⭐ 机制锚定半衰期</b> — {anchor_note}<br>'
                 f'纯 ML 半衰期模型训练集缺少长效脂化类似物，直接预测会低估约 10–20 倍；'
                 f'机制层从 SMILES 检测脂肪酰化类型并锚定到已发表同类药物 FDA 标签值，两演示药物标签未参与锚定。</div>')
    return html

# ---------------- 单分子 ----------------
def run_predict(sequence, smiles, rbc_species, progress=gr.Progress()):
    seq = re.sub(r'\s+', '', (sequence or '').upper())
    smi = (smiles or '').strip()
    if not seq and not smi:
        raise gr.Error('请至少输入氨基酸序列或 SMILES 之一（修饰肽建议提供完整 SMILES）')
    if seq and not re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{1,1022}', seq):
        raise gr.Error('序列只能包含 20 种标准氨基酸；修饰信息请通过 SMILES 提供')
    if seq:
        progress(0.25, desc='ESM2-650M GPU 嵌入 (cuda:1)…')
    payload = build_payload(seq, smi, rbc_species)
    progress(0.6, desc='node23 全端点推理…')
    t0 = time.time()
    r = call_api(payload)
    dt = time.time() - t0
    if not r.get('n_ok'):
        raise gr.Error('无法识别输入：既不是 20 种标准氨基酸组成的序列，也不是可解析的 SMILES。请检查输入内容。')
    rows, note = parse_rows(r)
    return table_html(rows), summarize(r, dt, note)

# ---------------- 批量 ----------------
def looks_like_smiles(s):
    return bool(re.search(r'[=#$%\[\]@]|Cl|Br|\(.*\)', s)) and re.search(r'[CNOPS]', s)

def parse_batch(file, pasted):
    """txt/csv/tsv(/xlsx 若装 openpyxl) 或粘贴文本 → [{name, sequence, smiles}]"""
    text, ext = pasted or '', ''
    if file is not None:
        raw = open(file, 'rb').read()
        ext = (file.name or '').lower().rsplit('.', 1)[-1] if file.name else ''
        if ext == 'xlsx':
            try:
                import openpyxl
                wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True)
                ws = wb.active
                lines = ['\t'.join('' if c is None else str(c) for c in row) for row in ws.iter_rows(values_only=True)]
                text = '\n'.join(lines); ext = 'tsv'
            except ImportError:
                raise gr.Error('服务器暂不支持 xlsx（缺 openpyxl），请另存为 CSV/TSV/TXT 上传')
        else:
            for enc in ('utf-8-sig', 'utf-8', 'gbk'):
                try: text = raw.decode(enc); break
                except UnicodeDecodeError: continue
    if not text.strip():
        raise gr.Error('请上传文件或粘贴内容（每行一个分子）')
    ext = ext or 'txt'
    delim = ',' if ext == 'csv' else '\t'
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header = None
    first = lines[0]
    cols0 = [c.strip().lower() for c in re.split(delim if ext != 'txt' else r'[,\t]', first)]
    if any(k in ' '.join(cols0) for k in ('sequence', 'seq', 'smiles', '名称', 'name')):
        header = cols0
        lines = lines[1:]
        if ext == 'txt': delim = ',' if ',' in first else '\t'
    def find_col(keys):
        for i, c in enumerate(header):
            if any(k in c for k in keys): return i
        return None
    out = []
    fasta_name = ''
    for ln in lines[:MAX_BATCH]:
        if ln.startswith('>'):          # FASTA header
            fasta_name = ln[1:].strip().split()[0] if len(ln) > 1 else ''
            continue
        parts = [p.strip() for p in re.split(delim if ext != 'txt' else r'[,\t]', ln)] if (ext != 'txt' or delim) else [ln]
        parts = [p for p in parts if p != ''] or [ln]
        seq = smi = ''
        if header is not None:
            i_seq, i_smi, i_name = find_col(('sequence', 'seq', '序列')), find_col(('smiles', '结构')), find_col(('name', 'id', '名称', '编号'))
            name = parts[i_name] if i_name is not None and i_name < len(parts) else f'M{len(out)+1}'
            seq = re.sub(r'\s+', '', parts[i_seq]).upper() if i_seq is not None and i_seq < len(parts) else ''
            smi = parts[i_smi] if i_smi is not None and i_smi < len(parts) else ''
        else:
            name = fasta_name or f'M{len(out)+1}'
            for p in parts:
                pu = re.sub(r'\s+', '', p).upper()
                if re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{4,}', pu) and not seq:
                    seq = pu
                elif looks_like_smiles(p) and not smi:
                    smi = p
                elif len(parts) > 1:
                    name = p[:40]
        if seq or smi:
            out.append({'name': name, 'sequence': seq, 'smiles': smi})
    if not out:
        raise gr.Error('未解析到任何分子：请提供 每行一条序列/SMILES，或带表头的 CSV/TSV')
    return out

def batch_preview(file, pasted):
    rows = parse_batch(file, pasted)
    sample = ''.join(
        f'<tr><td>{i+1}</td><td class="ep">{r["name"][:28]}</td>'
        f'<td class="val">{r["sequence"][:38]}{"…" if len(r["sequence"])>38 else ""}</td>'
        f'<td class="val">{r["smiles"][:38]}{"…" if len(r["smiles"])>38 else ""}</td></tr>'
        for i, r in enumerate(rows[:12]))
    more = f'<tr><td colspan="4" class="more">… 共 {len(rows)} 个分子</td></tr>' if len(rows) > 12 else ''
    return (f'<div class="result-note">已解析 <b>{len(rows)}</b> 个分子（上限 {MAX_BATCH}）</div>'
            f'<div class="res-wrap"><table class="res-table"><thead><tr><th>#</th><th>名称</th>'
            f'<th>序列</th><th>SMILES</th></tr></thead><tbody>{sample}{more}</tbody></table></div>')

def run_batch(file, pasted, rbc_species, progress=gr.Progress()):
    mols = parse_batch(file, pasted)
    long_rows, stats, failed = [], [], []
    t_start = time.time()
    for i, m in enumerate(mols):
        progress(i / max(len(mols), 1), desc=f'[{i+1}/{len(mols)}] {m["name"]}')
        try:
            t0 = time.time()
            r = call_api(build_payload(m['sequence'], m['smiles'], rbc_species))
            rows, _ = parse_rows(r)
            for x in rows:
                long_rows.append([m['name'], m['sequence'][:80], m['smiles'][:120]] + x[:7])
            stats.append([m['name'], r['n_ok'], round(time.time()-t0, 1)])
        except Exception as ex:
            failed.append([m['name'], str(ex)[:120]])
    if not long_rows and not failed:
        raise gr.Error('批量推理全部失败，请检查输入或稍后重试')
    # CSV 落盘
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='_omnigrid.csv', delete=False,
                                      dir='/tmp', encoding='utf-8-sig', newline='')
    with tmp:
        w = csv.writer(tmp)
        w.writerow(['name', 'sequence', 'smiles', 'category', 'endpoint', 'value', 'unit', 'seeds_sd', 'n_train', 'cv'])
        for row in long_rows:
            w.writerow(row)
    n_ok = sum(s[1] for s in stats)
    html = (f'<div class="cards">'
            f'<div class="card"><span>分子数</span><strong>{len(mols)}</strong></div>'
            f'<div class="card"><span>预测总行数</span><strong>{len(long_rows)}</strong></div>'
            f'<div class="card"><span>失败分子</span><strong>{len(failed)}</strong></div>'
            f'<div class="card"><span>总耗时</span><strong>{time.time()-t_start:.0f}s</strong></div></div>')
    if failed:
        html += '<div class="assumption"><b>失败分子</b> — ' + '；'.join(f'{f[0]}: {f[1]}' for f in failed[:6]) + '</div>'
    # 简化预览：分子 × 端点（前400行）
    prev = ''.join(
        f'<tr><td class="ep">{r[0]}</td><td><span class="chip" style="background:{CHIP.get(r[3],CHIP["理化·其他"])[0]};color:{CHIP.get(r[3],CHIP["理化·其他"])[1]}">{r[3]}</span></td>'
        f'<td class="ep">{r[4]}</td><td class="val">{r[5]}</td><td class="unit">{r[6]}</td></tr>'
        for r in long_rows[:400])
    more = f'<tr><td colspan="5" class="more">… 共 {len(long_rows)} 行，下载 CSV 查看全部</td></tr>' if len(long_rows) > 400 else ''
    html += (f'<div class="res-wrap"><table class="res-table"><thead><tr><th>分子</th><th>类别</th>'
             f'<th>端点</th><th>预测值</th><th>单位</th></tr></thead><tbody>{prev}{more}</tbody></table></div>')
    return html, tmp.name

# ---------------- UI ----------------（admetSAR 3 风格完全重构）
CSS = """
:root { --ink:#333; --sub:#767676; --accent:#319ff1; --accent-dark:#1a7fd0; --bg:rgba(240,248,255,.72); --panel:#fbfdff; --line:#e0e6ec; }
.gradio-container { max-width:1400px !important; background:var(--bg) !important; color:var(--ink) !important;
  font-family:'Helvetica Neue',Helvetica,Arial,'PingFang SC','Microsoft YaHei',sans-serif !important; }
/* ===== 顶部导航 ===== */
.navbar { display:flex; align-items:center; justify-content:space-between; gap:16px;
  background:#fff; border:1px solid var(--line); border-radius:6px; padding:0 22px; height:56px;
  box-shadow:0 1px 4px rgba(0,0,0,.06); margin-bottom:0; }
.navbar .brand { font-size:21px; font-weight:700; color:var(--ink); letter-spacing:.01em; white-space:nowrap; }
.navbar .brand em { font-style:normal; color:var(--accent); }
.navbar .nav { display:flex; gap:4px; }
.navbar .nav span { padding:7px 16px; font-size:14.5px; color:var(--ink); border-bottom:3px solid transparent; cursor:default; }
.navbar .nav span.on { color:var(--accent); border-bottom-color:var(--accent); font-weight:700; }
/* ===== 搜索主区（admetSAR signature）===== */
.hero { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:34px 34px 26px;
  box-shadow:0 2px 10px rgba(30,80,140,.06); margin-top:16px; }
.hero h2 { text-align:center; margin:0 0 6px; font-size:30px; font-weight:700; color:var(--ink); }
.hero .sub { text-align:center; color:var(--sub); font-size:14px; margin:0 0 22px; }
.searchbar { display:flex; max-width:900px; margin:0 auto; }
.searchbar textarea, .searchbar input { flex:1; height:52px !important; background:#fff !important; color:var(--ink) !important;
  border:1px solid #ccc !important; border-right:none !important; border-radius:3px 0 0 3px !important;
  font-size:15px !important; padding:6px 14px !important; box-shadow:inset 0 1px 2px rgba(0,0,0,.05); resize:none;
  font-family:'Helvetica Neue',Helvetica,Arial,'PingFang SC',sans-serif !important; }
.searchbar textarea:focus { outline:none; border-color:var(--accent) !important; box-shadow:inset 0 1px 2px rgba(0,0,0,.05),0 0 0 3px rgba(49,159,241,.15) !important; }
#predict-button { width:112px !important; height:52px !important; background:var(--accent) !important; color:#fff !important;
  border:1px solid var(--accent) !important; border-radius:0 3px 3px 0 !important; font-size:15.5px !important; font-weight:700 !important; margin:0 !important; }
#predict-button:hover { background:var(--accent-dark) !important; }
.examples { text-align:center; margin-top:14px; font-size:13px; color:var(--sub); }
.examples b { color:var(--ink); font-weight:400; margin-right:6px; }
.examples button { background:#fff !important; border:1px solid #cfe3f2 !important; color:var(--accent-dark) !important;
  border-radius:14px !important; font-size:12.5px !important; padding:2px 12px !important; margin:2px 3px !important; }
.examples button:hover { background:#eef6fd !important; }
.notice { max-width:900px; margin:12px auto 0; font-size:12px; color:#8a6d3b; background:#fcf8e3; border:1px solid #faebcc;
  border-radius:4px; padding:7px 12px; text-align:center; }
/* ===== 统计条 ===== */
.stats { display:flex; justify-content:center; gap:56px; flex-wrap:wrap; padding:20px 10px 8px; }
.stats div { text-align:center; }
.stats b { display:block; font-size:26px; color:var(--accent); font-weight:700; }
.stats span { font-size:15px; color:var(--ink); }
/* ===== 面板与表格 ===== */
.page-panel, .task-panel { border:1px solid var(--line) !important; border-radius:6px !important; background:var(--panel) !important;
  box-shadow:0 2px 8px rgba(30,80,140,.05) !important; }
.page-panel textarea, .page-panel input { background:#fff !important; color:var(--ink) !important;
  border:1px solid #ccc !important; border-radius:3px !important; font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12.5px; }
.page-panel textarea:focus, .page-panel input:focus { border-color:var(--accent) !important; }
.sec-title { font-size:19px; font-weight:700; color:var(--ink); margin:2px 0 4px; padding-left:10px; border-left:4px solid var(--accent); }
.assumption { padding:11px 14px; border-left:4px solid #f0ad4e; background:#fcf8e3; border-radius:4px; color:#6d5a10; font-size:12.5px; margin:8px 0 12px; line-height:1.7; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin:12px 0; }
.cards .card { background:#fff; border:1px solid var(--line); border-top:3px solid var(--accent); border-radius:5px; padding:11px 14px; }
.cards .card span { display:block; color:var(--sub); font-size:11.5px; }
.cards .card strong { display:block; color:var(--ink); font-size:21px; margin-top:3px; }
.res-wrap { overflow:auto; max-height:540px; border:1px solid var(--line); border-radius:5px; background:#fff; }
.res-table { border-collapse:collapse; width:100%; font-size:12.5px; }
.res-table th { position:sticky; top:0; background:linear-gradient(180deg,#eaf4fc,#dff0fa); color:#20536f; font-weight:700; text-align:left;
  padding:8px 12px; border-bottom:2px solid #bcdcf2; z-index:2; white-space:nowrap; }
.res-table td { padding:7px 12px; border-bottom:1px solid #eef3f7; color:#3d3d3d; vertical-align:middle; }
.res-table tbody tr:nth-child(even) { background:#f7fbfe; }
.res-table tbody tr:hover { background:#eaf4fc; }
.res-table .chip { display:inline-block; padding:2px 9px; border-radius:3px; font-size:11px; font-weight:700; white-space:nowrap; }
.res-table .ep { font-weight:700; color:#262626; }
.res-table .val { font-family:ui-monospace,Menlo,Consolas,monospace; font-weight:700; color:#0a3d62; white-space:nowrap; }
.res-table .unit { color:#8b98a5; font-size:11.5px; white-space:nowrap; }
.res-table .anchor-row { background:#fff6df !important; }
.res-table .anchor-row td { border-bottom-color:#f0dfae; }
.res-table .more { color:#7b95a8; text-align:center; font-size:12px; padding:10px; }
.result-note { border:1px solid var(--line); background:#f5fafe; color:#4a5c68; border-radius:4px; padding:9px 12px; margin-bottom:12px; font-size:12.5px; }
/* ===== 空状态 ===== */
.empty-state { border:1px dashed #bcdcf2; background:#fff; border-radius:6px; padding:44px 26px; text-align:center; color:#5a6b78; }
.empty-state .empty-title { font-size:16px; font-weight:700; color:var(--accent-dark); margin-bottom:10px; }
.empty-state p { font-size:13px; line-height:1.9; margin:7px 0; }
.empty-state .legend { display:flex; gap:10px; justify-content:center; flex-wrap:wrap; margin:15px 0; line-height:2.3; }
.empty-state .hint { color:#93a5b1; font-size:12px; }
/* ===== 页脚 ===== */
.site-footer { background:var(--bg); border-top:1px solid #d5e3ef; padding:26px 34px 14px; margin-top:26px;
  display:grid; grid-template-columns:2fr 1.1fr 1.1fr; gap:28px; }
.site-footer h1 { font-size:20px; font-weight:700; color:#000; margin:0 0 10px; }
.site-footer p, .site-footer li { font-size:13px; line-height:1.9; margin:0; color:#4a4a4a; }
.site-footer a { color:var(--accent-dark); text-decoration:none; }
.site-footer ul { list-style:none; padding:0; margin:0; }
.footer-bottom { text-align:center; color:#9aa8b3; font-size:12px; padding:12px 0 2px; }
@media (max-width: 900px) { .site-footer { grid-template-columns:1fr; } .navbar .nav { display:none; } }
"""

JS_FORCE_LIGHT = """() => { try {
  document.documentElement.classList.remove('dark'); document.body.classList.remove('dark');
  document.querySelectorAll('.dark').forEach(e => e.classList.remove('dark'));
  localStorage.setItem('gradio_theme', 'light');
} catch (e) {} }"""

AA_RE = re.compile(r'[ACDEFGHIKLMNPQRSTVWY]{4,}')

def detect_input(text):
    """admetSAR 式单框输入：自动识别氨基酸序列或 SMILES。"""
    t = re.sub(r'\s+', '', (text or '').strip().upper())
    if not t:
        return '', ''
    if AA_RE.fullmatch(t):
        return t, ''
    return '', re.sub(r'\s+', '', (text or '').strip())

PLACEHOLDER = """
<div class="empty-state">
  <div class="empty-title">🧬 等待预测</div>
  <p>在上方输入框中粘贴<b>氨基酸序列</b>（如 HISEAVKIFGCKWKENWKWNSNC）或<b>SMILES</b>（修饰肽建议），
  点击「Predict」一次性返回 55 个端点预测。</p>
  <p class="legend">
    <span class="chip" style="background:#e4f4ef;color:#0c6b5c">药代 PK</span>
    <span class="chip" style="background:#e2eff7;color:#175d80">渗透性</span>
    <span class="chip" style="background:#faf1e3;color:#8a5a14">抗菌·抑制活性</span>
    <span class="chip" style="background:#faecec;color:#8d4444">毒性·溶血</span>
    <span class="chip" style="background:#eef1f4;color:#4f5f6e">理化·其他</span>
  </p>
  <p class="hint">序列 + SMILES 同时提供可用端点最多（51/55）· 平均 3–5 秒 · internal_research_only</p>
</div>
"""

STATS_HTML = """
<div class="stats">
  <div><b>58,932</b><span>compounds</span></div>
  <div><b>55</b><span>endpoints</span></div>
  <div><b>213,800</b><span>records</span></div>
  <div><b>170</b><span>models (3-seed)</span></div>
</div>
"""

def run_predict_auto(main_input, sequence, smiles, rbc_species, progress=gr.Progress()):
    """主输入框自动识别序列/SMILES；高级选项中的显式输入优先级更高。"""
    seq_a, smi_a = detect_input(main_input)
    seq = re.sub(r'\s+', '', (sequence or '').upper()) or seq_a
    smi = (smiles or '').strip() or smi_a
    return run_predict(seq, smi, rbc_species, progress)

with gr.Blocks(title='OmniPanel · 多肽性质全景预测', css=CSS, js=JS_FORCE_LIGHT,
               theme=gr.themes.Default(primary_hue=gr.themes.colors.sky, neutral_hue=gr.themes.colors.gray,
                                       font=[gr.themes.GoogleFont('Helvetica Neue'), 'Arial', 'PingFang SC', 'sans-serif'])) as demo:
    gr.HTML('<div class="navbar">'
            '<div class="brand">🧬 Omni<em>Panel</em></div>'
            '<div class="nav"><span class="on">Predict</span><span>Batch</span><span>Resource</span></div>'
            '</div>')
    with gr.Tabs() as tabs:
        # ================= Predict（单分子） =================
        with gr.Tab('Predict', id='tab_predict'):
            gr.HTML('<div class="hero">'
                    '<h2>多肽性质全景预测</h2>'
                    '<p class="sub">Peptide Omni-Panel · 氨基酸序列或完整 SMILES（含修饰）→ 55 个 ADMET / 活性端点 · 3-seed 一致模型 · ESM2-650M</p>'
                    '</div>')
            with gr.Row(elem_classes=['searchbar-wrap']):
                main_in = gr.Textbox(show_label=False, lines=1, max_lines=1, elem_classes=['searchbar'],
                                     placeholder='输入氨基酸序列（如 HISEAVKIFGCKWKENWKWNSNC）或 SMILES，自动识别 · 修饰肽请给完整 SMILES')
                go = gr.Button('Predict', elem_id='predict-button')
            gr.HTML('<div class="examples"><b>Examples:</b></div>')
            with gr.Row():
                b_amp = gr.Button('示例抗菌肽 (序列)', size='sm')
                b_sem = gr.Button('司美格鲁肽 (SMILES)', size='sm')
                b_tir = gr.Button('替尔泊肽 (SMILES)', size='sm')
            gr.HTML('<div class="notice">⚠️ internal_research_only — 本平台仅供内部科研使用，不可用于临床或监管决策。</div>')
            with gr.Accordion('⚙️ 高级选项（序列与 SMILES 分别输入 / RBC 物种）', open=False, elem_classes=['page-panel']):
                with gr.Row():
                    with gr.Column():
                        seq_in = gr.Textbox(label='氨基酸序列（仅 20 种标准 AA）', lines=2,
                                            placeholder='留空则使用上方主输入框', elem_classes=['page-panel'])
                        smi_in = gr.Textbox(label='SMILES（完整共价结构）', lines=2,
                                            placeholder='留空则使用上方主输入框', elem_classes=['page-panel'])
                    with gr.Column():
                        rbc = gr.Radio(['人 (human)', '马 (horse)', '鼠 (mouse)'], value='人 (human)',
                                       label='溶血端点 RBC 物种（MHC10 / HC50）')
                        gr.HTML('<div style="font-size:12px;color:#767676;line-height:1.8">'
                                '· 裸序列 → ESM2/序列特征（约 13 个端点）<br>'
                                '· 仅 SMILES → 结构特征（约 43 个端点）<br>'
                                '· 两者都给 → 全部约 51 个端点</div>')
            gr.HTML(STATS_HTML)
            summary = gr.HTML(value=PLACEHOLDER)
            result = gr.HTML()
        # ================= Batch =================
        with gr.Tab('Batch', id='tab_batch'):
            gr.HTML('<div class="hero"><h2>批量预测</h2>'
                    '<p class="sub">上传 txt / csv / tsv / xlsx / FASTA 文件，或直接粘贴（每行一个分子，自动识别序列 / SMILES）· 上限 200 个</p></div>')
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Column(elem_classes=['page-panel']):
                        up = gr.File(label='上传分子文件（.txt / .csv / .tsv / .xlsx / .fasta）',
                                     file_types=['.txt', '.csv', '.tsv', '.xlsx', '.fasta'], type='filepath')
                        paste = gr.Textbox(label='或直接粘贴（支持 FASTA：>名称 换行 序列）', lines=6,
                                           placeholder='HISEAVKIFGCKWKENWKWNSNC\n>peptide1\nGIGKFLHSAKKFGKAFVGEIMNS')
                        rbc_b = gr.Radio(['人 (human)', '马 (horse)', '鼠 (mouse)'], value='人 (human)',
                                         label='溶血端点 RBC 物种')
                        b_prev = gr.Button('📋 解析预览', size='sm')
                        go_b = gr.Button('🚀 批量预测', variant='primary', elem_id='predict-button')
                with gr.Column(scale=7):
                    preview = gr.HTML()
            batch_out = gr.HTML()
            csv_out = gr.File(label='📥 下载完整结果 CSV')
        # ================= Resource =================
        with gr.Tab('Resource', id='tab_resource'):
            gr.HTML('<div class="hero"><h2>模型与说明</h2></div>')
            gr.HTML("""
<div style="max-width:1000px;margin:0 auto;font-size:13.5px;line-height:2;color:#3d3d3d;">
<p><b style="color:#1a7fd0">模型体系</b> — 55 个端点 × 3 个随机种子（42/2026/3407）已保存优胜模型；每端点独立 Optuna(128) 搜索 × 8 个模型族
（XGBoost / LightGBM / HistGB / ExtraTrees / 随机森林 / SVR / Ridge / ElasticNet）；17 组冻结化学表示（Morgan r2/r3、原子对、拓扑扭转、MACCS、
RDKit 描述符、EState、ErG、Pattern、Layered、路径指纹、序列组成、二肽组成、序列扩展、ESM2-650M 等）。
特征名/顺序与训练合同逐列一致（SHA256 字节级校验）；血浆蛋白结合端点端到端复现值 0.9934 / 0.9933 与冻结报告一致。</p>
<p><b style="color:#1a7fd0">⭐ 机制锚定半衰期</b> — 纯 ML 半衰期模型训练集缺少长效脂化类似物，对周给药药物低估约 10–20 倍。平台从 SMILES 检测
脂肪二羧酸/单酰化签名并锚定到已发表同类药物 FDA 标签值：C≥16 二羧酸 → 120–168h（dulaglutide / efpeglenatide / albiglutide）；
C12–15 → ~25h（degludec）；C14–16 单酰 → 5–13h（detemir / liraglutide）。两演示药物标签未参与锚定（12/12 外部验证全中）。</p>
<p><b style="color:#1a7fd0">数据与边界</b> — 新旧来源登记 1,289,057 行、213,800 条清晰标签；6 个条件端点（AC50 / 绝对清除率 / 肾清除率 / Cmax / Vd）
依赖实验条件协变量，部署中。全部模型 internal_research_only；开发 OOF 指标不等同独立外部验证。</p>
</div>
""")
            gr.HTML(STATS_HTML)
    gr.HTML("""
<div class="site-footer">
  <div>
    <h1>ABOUT</h1>
    <p>OmniPanel：面向多肽药物研发的全端点性质预测平台。序列 / 修饰 SMILES 一次输入，
    同时返回 55 个端点（药代 · 渗透 · 抗菌活性 · 毒性溶血 · 理化），含机制锚定半衰期层。internal_research_only。</p>
  </div>
  <div>
    <h1>HELP CENTER</h1>
    <ul>
      <li><a href="https://lmmd.ecust.edu.cn/" target="_blank">&gt; LMMD Group</a></li>
      <li><a href="https://pharmacy.ecust.edu.cn/" target="_blank">&gt; 华东理工大学药学院</a></li>
      <li><a href="https://www.ecust.edu.cn/" target="_blank">&gt; 华东理工大学</a></li>
    </ul>
  </div>
  <div>
    <h1>DEV TOOL</h1>
    <ul>
      <li>&gt; Python · PyTorch · Transformers</li>
      <li>&gt; ESM2-650M · 17 组化学表示</li>
      <li>&gt; LightGBM · XGBoost · Optuna</li>
      <li>&gt; node2 GPU + node23 推理集群</li>
    </ul>
  </div>
</div>
<div class="footer-bottom">OmniPanel · 55 endpoints · 3-seed consensus · internal_research_only</div>
""")

    b_sem.click(lambda: SEMAGLUTIDE_SMILES, None, main_in)
    b_tir.click(lambda: TIRZEPATIDE_SMILES, None, main_in)
    b_amp.click(lambda: 'HISEAVKIFGCKWKENWKWNSNC', None, main_in)
    go.click(run_predict_auto, [main_in, seq_in, smi_in, rbc], [result, summary], api_name='predict_all')
    b_prev.click(batch_preview, [up, paste], preview, api_name='batch_preview')
    go_b.click(run_batch, [up, paste, rbc_b], [batch_out, csv_out], api_name='run_batch')

if __name__ == '__main__':
    demo.queue(max_size=64, default_concurrency_limit=8).launch(
        server_name='127.0.0.1', server_port=7862,
        root_path='/peptide-omnipanel', show_error=True,
        allowed_paths=['/tmp'])
