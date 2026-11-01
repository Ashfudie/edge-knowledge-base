/**
 * 主应用逻辑：模型加载（后端探测/降级）→ 文档导入（分块+浏览器向量化+写入 seekdb）
 * → 混合检索（查询向量化在本地，RRF 融合在 seekdb 内）→ 评测面板。
 */
import { api } from "./api.js";
import {
  MODEL_DIMENSION, MODEL_NAME,
  detectWebGPU, resolveBackend, loadModel, modelReady, embedOne, embedBatch,
} from "./embedder.js";
import { runBenchmark, exportResults } from "./benchmark.js";

const $ = (id) => document.getElementById(id);
const state = {
  collections: [],
  current: null,          // 当前集合名
  backend: null,          // 'webgpu' | 'wasm'
  dtype: "fp16",          // 默认 FP16：WebGPU 下质量与吞吐实测最优（见技术说明）
  importing: false,
};

/* ---------------- 工具 ---------------- */

const CJK_RE = /[\u4e00-\u9fff\u3400-\u4dbf]/g;
function detectLang(text) {
  const cjk = (text.match(CJK_RE) || []).length;
  return cjk > text.length * 0.15 ? "zh" : "en";
}

/** 分块：先按空行分段；超长段落按 size 切并带 overlap */
function chunkText(text, size, overlap) {
  const paras = text.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  const chunks = [];
  for (const para of paras) {
    if (para.length <= size) {
      chunks.push(para);
      continue;
    }
    for (let i = 0; i < para.length; i += size - overlap) {
      const piece = para.slice(i, i + size);
      if (piece.trim()) chunks.push(piece.trim());
      if (i + size >= para.length) break;
    }
  }
  return chunks.length ? chunks : [text.trim()].filter(Boolean);
}

function fmtMs(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms.toFixed(1)}ms`;
}

/* ---------------- 模型与后端 ---------------- */

function setBackendBadge(backend, reason) {
  const el = $("backend-badge");
  el.className = `badge ${backend}`;
  el.textContent = backend === "webgpu"
    ? "推理后端：WebGPU（GPU 加速）"
    : `推理后端：WASM（CPU）${reason ? ` · ${reason}` : ""}`;
  el.title = reason || "";
}

async function initModel({ silent } = { silent: false }) {
  const preference = $("backend-select").value;
  const dtype = $("precision-select").value;
  const status = $("model-status");
  try {
    const backend = await resolveBackend(preference);
    const gpuOK = await detectWebGPU();
    state.backend = backend;
    setBackendBadge(backend, preference === "auto" && !gpuOK ? "WebGPU 不可用，已自动降级" : "");
    status.textContent = `加载模型中（${backend} · ${dtype}）…`;
    const { loadMs } = await loadModel(backend, dtype, (p) => {
      if (p.status === "progress" && p.file?.endsWith(".onnx")) {
        status.textContent = `加载模型 ${p.file} ${(p.loaded / 1048576).toFixed(1)}/${(p.total / 1048576).toFixed(1)}MB`;
      }
    });
    state.dtype = dtype;
    status.textContent = `${MODEL_NAME} · ${dtype} · 加载 ${fmtMs(loadMs)} · ${MODEL_DIMENSION} 维`;
    $("btn-import").disabled = false;
    return true;
  } catch (e) {
    state.backend = null;
    setBackendBadge("error", e.message);
    status.textContent = `模型加载失败：${e.message}`;
    if (!silent) alert(e.message);
    return false;
  }
}

/* ---------------- 集合 ---------------- */

async function refreshCollections(selectName) {
  const { collections } = await api.listCollections();
  state.collections = collections;
  const list = $("collection-list");
  list.innerHTML = "";
  for (const c of collections) {
    const li = document.createElement("li");
    li.className = c.name === (selectName ?? state.current) ? "active" : "";
    li.innerHTML = `<span class="name">${c.name}</span><span class="count">${c.count ?? 0} 条</span>`;
    li.onclick = () => {
      state.current = c.name;
      $("search-collection-tag").textContent = `当前集合：${c.name}`;
      refreshCollections();
    };
    list.appendChild(li);
  }
  if (selectName) {
    state.current = selectName;
    $("search-collection-tag").textContent = `当前集合：${selectName}`;
  }
  if (!collections.find((c) => c.name === state.current)) state.current = null;
  if (!state.current && collections.length) {
    state.current = collections[0].name;
    $("search-collection-tag").textContent = `当前集合：${state.current}`;
    refreshCollections();
  }
}

async function createCollection() {
  const name = $("new-collection-name").value.trim();
  if (!name) return alert("请输入集合名");
  await api.createCollection(name, MODEL_DIMENSION);
  $("new-collection-name").value = "";
  await refreshCollections(name);
}

async function ensureCollectionForImport() {
  if (state.current) return state.current;
  const name = `kb_${new Date().toISOString().slice(0, 10).replace(/-/g, "")}`;
  await api.createCollection(name, MODEL_DIMENSION);
  await refreshCollections(name);
  return name;
}

/* ---------------- 导入 ---------------- */

async function collectInputs() {
  const docs = []; // {source, text}
  const pasted = $("import-text").value.trim();
  if (pasted) docs.push({ source: "粘贴文本", text: pasted });
  const files = $("import-files").files;
  for (const f of files) docs.push({ source: f.name, text: await f.text() });
  return docs;
}

async function startImport() {
  if (state.importing) return;
  if (!modelReady()) return alert("模型尚未就绪");
  const inputs = await collectInputs();
  if (!inputs.length) return alert("请先粘贴文本或选择文件");
  state.importing = true;
  $("btn-import").disabled = true;
  $("import-progress").classList.remove("hidden");
  $("import-summary").classList.add("hidden");

  try {
    const collection = await ensureCollectionForImport();
    const chunkSize = Number($("chunk-size").value) || 500;
    const overlap = 50;
    const batchSize = Number($("batch-size").value) || 16;

    // 1) 分块
    const tChunk0 = performance.now();
    const chunks = [];
    for (const d of inputs) {
      const cs = chunkText(d.text, chunkSize, overlap);
      for (let i = 0; i < cs.length; i++) {
        chunks.push({ source: d.source, chunk: i, text: cs[i], lang: detectLang(cs[i]) });
      }
    }
    const chunkMs = performance.now() - tChunk0;

    // 2) 浏览器本地向量化（批处理）
    const { embeddings, totalMs: embedMs, batches } = await embedBatch(
      chunks.map((c) => c.text), batchSize,
      (done, total, batchMs) => {
        const pct = (done / total) * 100;
        $("import-progress-fill").style.width = `${pct}%`;
        $("import-progress-text").textContent =
          `向量化 ${done}/${total}（批 ${batchSize} 条，最近一批 ${fmtMs(batchMs)}）`;
      }
    );

    // 3) 写入 seekdb（向量+原文+元数据），服务端返回写入/索引刷新耗时
    $("import-progress-text").textContent = "写入 seekdb 并刷新索引…";
    const stamp = Date.now().toString(36);
    const documents = chunks.map((c, i) => ({
      id: `${collection}_${stamp}_${i}`,
      text: c.text,
      metadata: { source: c.source, chunk: c.chunk, lang: c.lang, chars: c.text.length },
      embedding: embeddings[i],
    }));
    const addResult = await api.addDocuments(collection, documents);

    // 4) 写入后立刻验证可检索性（增量一致性）
    const verifyStart = performance.now();
    const verify = await api.search(collection, {
      text: chunks[0].text.slice(0, 24), embedding: embeddings[0], nResults: 3, mode: "hybrid",
    });
    const verifyMs = performance.now() - verifyStart;

    $("import-progress-fill").style.width = "100%";
    const throughput = chunks.length / (embedMs / 1000);
    $("import-summary").classList.remove("hidden");
    $("import-summary").innerHTML = `
      <div>集合 <b>${collection}</b>：新导入 <b>${chunks.length}</b> 块（${inputs.length} 个来源，块大小 ${chunkSize}/重叠 ${overlap}）</div>
      <div class="mono">
        分块 ${fmtMs(chunkMs)} · 向量化 ${fmtMs(embedMs)}（${batches} 批，${throughput.toFixed(1)} 块/秒，${state.backend}/${state.dtype}）
        · 写入 ${fmtMs(addResult.writeMs)} · 索引刷新 ${fmtMs(addResult.refreshMs)}
      </div>
      <div class="mono">写入后可检索性验证：${verify.items.length ? "✅ 首条已可检索" : "⚠️ 未检出"}（${fmtMs(verifyMs)}）</div>`;

    $("import-text").value = "";
    $("import-files").value = "";
    await refreshCollections(collection);
  } catch (e) {
    alert(`导入失败：${e.message}`);
    $("import-progress-text").textContent = `失败：${e.message}`;
  } finally {
    state.importing = false;
    $("btn-import").disabled = false;
  }
}

/* ---------------- 检索 ---------------- */

async function doSearch() {
  const text = $("query-input").value.trim();
  if (!text) return;
  if (!state.current) return alert("请先创建或选择一个集合");
  if (!modelReady()) return alert("模型尚未就绪");
  const mode = $("search-mode").value;
  const nResults = Number($("topk").value) || 8;

  $("btn-search").disabled = true;
  const resultsEl = $("search-results");
  resultsEl.innerHTML = `<div class="empty">检索中…</div>`;

  try {
    const t0 = performance.now();
    // 查询向量：浏览器本地计算（仅向量/混合模式需要）
    let embedding = null;
    let embedMs = 0;
    if (mode !== "fulltext") {
      const e0 = performance.now();
      embedding = await embedOne(text);
      embedMs = performance.now() - e0;
    }
    const res = await api.search(state.current, { text, embedding, nResults, mode });
    const e2eMs = performance.now() - t0;

    $("search-timings").classList.remove("hidden");
    $("search-timings").innerHTML = [
      `查询向量化 <b>${fmtMs(embedMs)}</b>（浏览器，${state.backend}/${state.dtype}）`,
      `seekdb 检索 <b>${fmtMs(res.timings.seekdbMs)}</b>（${mode}，库内执行）`,
      `端到端 <b>${fmtMs(e2eMs)}</b>`,
    ].join(" · ");

    resultsEl.innerHTML = "";
    if (!res.items.length) {
      resultsEl.innerHTML = `<div class="empty">无结果</div>`;
      return;
    }
    for (const item of res.items) {
      const div = document.createElement("div");
      div.className = "result-card";
      const meta = item.metadata ?? {};
      div.innerHTML = `
        <div class="result-head">
          <span class="rank">#</span>
          <span class="source">${meta.source ?? item.id}${meta.chunk != null ? ` · 块 ${meta.chunk}` : ""}</span>
          <span class="score">${item.score != null ? `score ${item.score.toFixed(4)}` : ""}</span>
          <span class="lang-tag ${meta.lang ?? ""}">${meta.lang ?? ""}</span>
        </div>
        <div class="result-text">${(item.text ?? "").replace(/</g, "&lt;")}</div>`;
      div.querySelector(".rank").textContent = `#${res.items.indexOf(item) + 1}`;
      resultsEl.appendChild(div);
    }
  } catch (e) {
    resultsEl.innerHTML = `<div class="empty error">${e.message}</div>`;
  } finally {
    $("btn-search").disabled = false;
  }
}

/* ---------------- 事件绑定与启动 ---------------- */

$("btn-create-collection").onclick = createCollection;
$("btn-refresh-collections").onclick = () => refreshCollections();
$("btn-refresh-index").onclick = async () => {
  if (!state.current) return;
  const r = await api.refreshIndex(state.current);
  alert(`索引刷新完成 ${fmtMs(r.refreshMs)}`);
};
$("btn-delete-collection").onclick = async () => {
  if (!state.current) return;
  if (!confirm(`确认删除集合 ${state.current}？该操作不可恢复。`)) return;
  await api.deleteCollection(state.current);
  state.current = null;
  await refreshCollections();
};
$("btn-import").onclick = startImport;
$("btn-search").onclick = doSearch;
$("query-input").addEventListener("keydown", (e) => e.key === "Enter" && doSearch());
$("backend-select").onchange = () => initModel();
$("precision-select").onchange = () => initModel();
$("btn-benchmark").onclick = () => runBenchmark(state);
$("btn-export-results").onclick = () => exportResults();

(async function boot() {
  await initModel({ silent: true });
  try {
    await refreshCollections();
  } catch (e) {
    console.error("集合列表加载失败（后端未启动？）", e);
  }
})();
