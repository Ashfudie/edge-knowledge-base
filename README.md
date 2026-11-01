# Edge Knowledge Base · 端侧知识库

一个**完全在你的浏览器里**运行的迷你知识库：文档导入后在本地完成向量化（不调用任何云端推理 API），向量、原文与元数据存入 [seekdb](https://github.com/oceanbase/seekdb)，检索时由 seekdb 在数据库内完成**向量 + 全文混合检索**（RRF 融合），返回单一排序列表。

```
导入文档 → 浏览器本地向量化（分块 · 批处理 · 量化模型） → 写入 seekdb → 混合检索 → 排序结果
```

## 特性

- **端侧推理**：Transformers.js + ONNX Runtime Web，文档与查询的 embedding 全部在用户设备计算；模型与运行时随应用分发，离线可用
- **WebGPU 优先，WASM 自动降级**：显式配置推理后端；环境不支持 WebGPU 时自动降级到 WASM 并在界面明确提示当前后端
- **量化模型**：默认 FP16（体积减半、检索质量零损失），可选 INT8 / FP32；内置模型体积与压缩比统计
- **库内混合检索**：全文（IK 分词）与向量（HNSW）两路检索在 seekdb 内用一条 SQL 完成 RRF 融合，客户端不做两路拼合
- **增量导入**：分批写入 + 索引刷新，新文档秒级可检索；每次导入自动做可检索性验证
- **多集合管理**：创建/切换/删除知识库集合，检索始终作用于当前集合
- **内置评测面板**：中英混合评测集上实测 Recall@K / nDCG@K 与分阶段延迟（查询向量化 / seekdb 检索 / 端到端），一键导出 JSON，切换精度/后端即可横向对比

## 快速开始

要求：**Node.js ≥ 22**，**Chrome / Edge 最新版**（WebGPU 需要较新版本；不支持时自动降级 WASM）。
已验证环境：macOS 15+（Apple Silicon）+ Chrome 156 + Node 26。seekdb 运行时通过 npm 分发（`@oceanbase/seekdb-node`），无需单独安装数据库。

```bash
# 1. 安装依赖
#    注意：上游 @seekdb/default-embed 的 peerDependencies 误带 workspace:* 协议，
#    必须加 --legacy-peer-deps（详见 FAQ）
npm install --legacy-peer-deps

# 2. 模型文件已随仓库提供（web/public/models，FP16+INT8）。
#    如需 FP32 基线（量化对比实验用，约 90MB）：
npm run download-models-fp32      # 默认从 hf-mirror.com 拉取，海外网络可 HF_ENDPOINT=https://huggingface.co

# 3. 构建前端并启动（一条命令拉起 HTTP 服务 + 本地 seekdb）
npm run build
npm start
# 打开 http://127.0.0.1:8787
```

开发模式（前端热更新，需另开终端跑 `npm start`）：

```bash
npm start          # 终端 1：后端 + seekdb（:8787）
npm run dev        # 终端 2：vite dev server（:5173，/api 代理到 8787）
```

首次启动会初始化本地 seekdb 数据目录（`data/seekdb/`），冷启动约 2-4 秒；之后复用数据目录。按 `Ctrl+C` 退出会同时关闭 seekdb 进程。

## 使用流程

1. 打开页面后，状态栏显示**当前推理后端**（WebGPU / WASM）与模型/精度/加载耗时/维度
2. **导入**：粘贴文本或选择 `.txt` / `.md` 文件 → 自动分块（默认 500 字符 / 重叠 50）→ 浏览器本地批向量化（默认 16 条/批）→ 连同向量、原文、元数据写入 seekdb → 自动索引刷新 → 内置"写入后可检索性"验证
3. **检索**：输入自然语言查询 → 浏览器本地计算查询向量 → seekdb 在库内执行全文 + 向量 + RRF 混合检索 → 展示排序结果、原文与延迟细分（查询向量化 / seekdb 检索 / 端到端）
4. **评测**：点击"载入评测集并运行"，在 60 条中英混合语料 + 36 条查询上实测 Recall@5/@10、nDCG@10、跨语言命中率与各阶段延迟；切换精度/后端再跑即可横向对比，结果可导出 JSON

## 架构

```
┌─────────────────────────── 浏览器（用户设备） ───────────────────────────┐
│  前端 (web/)                                                            │
│   • Transformers.js + ONNX Runtime Web                                  │
│     - device: webgpu（优先）→ wasm（自动降级，UI 明示）                    │
│     - dtype: fp16（默认）/ q8 (INT8) / fp32                              │
│   • 文档分块、批量向量化、进度与计时                                        │
│   • 查询向量化（本地计算，不经过服务端）                                    │
└──────────────┬───────────────────────────────────────────────────────────┘
               │ HTTP (JSON)：只有向量、原文、元数据与查询文本
┌──────────────▼───────────────────────────────────────────────────────────┐
│  Node 服务 (server/)    —— 不做任何 embedding 计算                        │
│   • @oceanbase/seekdb-node：拉起本地 seekdb server 进程（MySQL 协议）        │
│   • TCP↔UnixSocket 桥（seekdb-js ConnectionConfig 仅支持 host/port）        │
│   • seekdb-js（SeekdbClient）Collection API：                            │
│       collection.add / refresh_index / hybridSearch                    │
└──────────────┬───────────────────────────────────────────────────────────┘
               │ MySQL 协议（本机回环）
┌──────────────▼───────────────────────────────────────────────────────────┐
│  seekdb v1.4（嵌入式本地 server，数据目录 data/seekdb）                     │
│   • HNSW 向量索引（dim=512，cosine，m=16，ef_construction=200，ef_search=256）│
│   • IK 全文索引（中文分词）                                                │
│   • DBMS_HYBRID_SEARCH：全文 + 向量两路检索与 RRF 融合全部在库内完成          │
└───────────────────────────────────────────────────────────────────────────┘
```

设计要点：

- **隐私**：语料不出设备——浏览器算完向量后，只有向量与原文进入用户自建的本地数据库
- **混合检索在库内**：`collection.hybridSearch({ query: {whereDocument}, knn: {queryEmbeddings}, rank: {rrf} })` → `DBMS_HYBRID_SEARCH.GET_SQL` 生成 SQL 并在 seekdb 内执行融合排序，客户端拿到的已是单一排序列表
- **数据库不做向量化**：集合创建时显式 `embeddingFunction: null`，向量只能由客户端写入

## 目录结构

```
├── server/
│   ├── index.mjs            # HTTP 服务：REST API + 静态托管 web/dist
│   └── seekdb.mjs           # seekdb 运行时封装（进程 + TCP 桥 + 集合管理）
├── web/
│   ├── index.html           # 单页应用
│   ├── src/
│   │   ├── main.js          # 应用主逻辑（集合/导入/检索/状态）
│   │   ├── embedder.js      # Transformers.js 封装：后端探测/降级、精度切换、批处理
│   │   ├── benchmark.js     # 内置评测：Recall@K / nDCG@K / 分阶段延迟
│   │   ├── api.js           # 后端 API 客户端
│   │   └── style.css
│   ├── public/
│   │   ├── models/bge-small-zh-v1.5/   # 本地模型（FP16 + INT8，随仓库分发）
│   │   └── eval-corpus.json            # 评测集（60 语料 + 36 查询）
│   └── vite.config.js
├── scripts/
│   ├── download-models.mjs  # 模型获取（hf-mirror/HF，--fp32 附带基线）
│   ├── gen-corpus.mjs       # 评测集生成（确定性输出，可复现/可替换）
│   ├── size-report.mjs      # 模型体积与压缩比统计
│   ├── benchmark-chrome.mjs # 系统 Chrome 自动化评测（多后端×多精度配置）
│   └── batch-ab.mjs         # embedding 批处理优化对比实验
├── test-results/            # 实测数据（Chrome 156 / Apple Silicon）
├── 技术说明.md               # 量化方案对比、性能实测与优化细节
└── README.md
```

## REST API（服务端）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | seekdb 运行状态与版本 |
| GET | `/api/collections` | 集合列表（含文档数） |
| POST | `/api/collections` | 创建集合 `{name, dimension}` |
| DELETE | `/api/collections/:name` | 删除集合 |
| POST | `/api/collections/:name/add` | 写入 `{documents:[{id,text,metadata,embedding}]}`，分批写入 + 自动索引刷新，返回写入/刷新耗时 |
| POST | `/api/collections/:name/search` | 检索 `{text, embedding, nResults, mode: hybrid\|knn\|fulltext}`，返回结果与 seekdb 耗时 |
| POST | `/api/collections/:name/refresh` | 手动索引刷新 |
| GET | `/api/collections/:name/count` | 文档计数 |

环境变量：`PORT`（默认 8787）、`SEEKDB_DATA_DIR`（默认 `./data/seekdb`）、`SEEKDB_MEMORY_BUDGET`（默认 2G）、`SEEKDB_ADD_BATCH`（默认 200）。

## 复现实验

```bash
# 模型体积与压缩比
npm run size-report

# 重新生成评测集（确定性输出）
node scripts/gen-corpus.mjs

# 系统 Chrome 多配置自动化评测（WebGPU/WASM × fp16/fp32/q8 + 自动降级验证）
# 需先 npm run build && npm start
node scripts/benchmark-chrome.mjs
# 结果输出 test-results/benchmark-chrome.json

# embedding 批处理优化对比（batch 1/4/16/32）
node scripts/batch-ab.mjs
```

应用内"评测"面板与上述脚本走同一条链路；切换页面顶部"精度/后端"下拉框可复测任意配置。主要实测结论（详见[技术说明](技术说明.md)）：

| 配置 | Recall@10 | nDCG@10 | 端到端中位 | 导入吞吐 |
| --- | --- | --- | --- | --- |
| FP32 + WebGPU | 1.000 | 1.000 | 8.4–10.8 ms | ~300–380 条/s |
| **FP16 + WebGPU（默认）** | 1.000 | 1.000 | 9.0 ms | 370–417 条/s |
| INT8 + WebGPU | 0.917 | 0.517 | 29.9 ms | 20 条/s |
| FP16 + WASM（降级） | 1.000 | 1.000 | 22.2 ms | 21 条/s |

值得注意的实测发现：ONNX Runtime Web 的 **WebGPU 后端对 INT8 算子走反量化模拟路径，检索质量明显下降且吞吐反而更低**（本项目的评测集中 nDCG 1.00 → 0.52），因此默认使用 FP16；INT8 仅在 WASM 路径下接近无损。批处理（batch 1 → 16）实测带来 **5.9×** 向量化吞吐提升。

## FAQ

- **`npm install` 报 `Unsupported URL Type "workspace:"`**：上游 `@seekdb/default-embed@1.4.0` 的 peerDependencies 误带 workspace 协议，请使用 `npm install --legacy-peer-deps`。
- **模型下载慢/失败**：`download-models.mjs` 默认走 `https://hf-mirror.com`，可用 `HF_ENDPOINT=https://huggingface.co` 切换官方源。仓库已内置 FP16+INT8 权重，通常无需下载。
- **界面显示 WASM 而非 WebGPU**：当前浏览器/设备不支持 WebGPU（或被 `chrome://flags` 关闭）。这是预期降级路径，核心功能不变。macOS Chrome/Edge 最新版默认支持。
- **平台支持**：`@oceanbase/seekdb-node` 原生运行时覆盖 linux x64/arm64、darwin arm64、win32 x64；seekdb-js 亦支持连接独立部署的 seekdb server（改 `server/seekdb.mjs` 中连接参数即可）。
- **首次导入慢**：首轮包含 GPU 管线编译与模型预热，属预期；预热后吞吐显著提升（见技术说明中的冷/热对比数据）。

## 致谢与许可

- [MIT](LICENSE)
- 嵌入模型：[BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)（MIT，本项目使用其 [Xenova ONNX 导出](https://huggingface.co/Xenova/bge-small-zh-v1.5)）
- 推理运行时：[Transformers.js](https://github.com/huggingface/transformers.js) / ONNX Runtime Web（Apache-2.0）
- 数据库：[seekdb](https://github.com/oceanbase/seekdb)（Apache-2.0）
