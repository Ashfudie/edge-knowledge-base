/**
 * 迷你知识库 HTTP 服务：
 *  - 启动/关闭本地 seekdb 运行时（@oceanbase/seekdb-node，数据存 SEEKDB_DATA_DIR）
 *  - 提供 REST API：集合管理、向量+原文写入（含索引刷新计时）、向量+全文混合检索（seekdb 内完成）
 *  - 托管前端构建产物（web/dist），生产模式一条命令启动
 *
 * 注意：本服务不做任何 embedding 计算。文档/查询向量均由浏览器端计算后提交，
 * 混合检索由 seekdb 的 DBMS_HYBRID_SEARCH 在数据库内执行（RRF 融合）。
 */
import http from "node:http";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SeekdbRuntime } from "./seekdb.mjs";

const ROOT = path.resolve(fileURLToPath(new URL("../", import.meta.url)));
const DIST_DIR = path.join(ROOT, "web", "dist");
const PORT = Number(process.env.PORT || 8787);
const DATA_DIR = process.env.SEEKDB_DATA_DIR || path.join(ROOT, "data", "seekdb");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".onnx": "application/octet-stream",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".txt": "text/plain; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
};

const runtime = new SeekdbRuntime(DATA_DIR);

/* ---------------- helpers ---------------- */

function json(res, status, body) {
  const buf = Buffer.from(JSON.stringify(body));
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": buf.length,
    "access-control-allow-origin": "*",
  });
  res.end(buf);
}

async function readBody(req, limit = 256 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const c of req) {
    size += c.length;
    if (size > limit) throw new Error("body too large");
    chunks.push(c);
  }
  if (!chunks.length) return {};
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function validName(name) {
  return typeof name === "string" && /^[A-Za-z0-9_][A-Za-z0-9_-]{0,63}$/.test(name);
}

/* ---------------- API handlers ---------------- */

async function handleApi(req, res, url) {
  const route = `${req.method} ${url.pathname}`;

  // ---- 健康检查 ----
  if (route === "GET /api/health") {
    const info = await runtime.client.execute("SELECT VERSION() AS v");
    return json(res, 200, {
      ok: true,
      seekdb: { mode: "local-server (@oceanbase/seekdb-node)", version: info?.[0]?.v ?? "unknown", dataDir: DATA_DIR },
    });
  }

  // ---- 集合列表 ----
  if (route === "GET /api/collections") {
    const cols = await runtime.listCollections();
    const out = [];
    for (const c of cols) {
      let count = null;
      try { count = await c.count(); } catch {}
      out.push({ name: c.name, count });
    }
    out.sort((a, b) => a.name.localeCompare(b.name));
    return json(res, 200, { collections: out });
  }

  // ---- 创建集合 ----
  if (route === "POST /api/collections") {
    const body = await readBody(req);
    const { name, dimension } = body;
    if (!validName(name)) return json(res, 400, { error: "集合名需匹配 ^[A-Za-z0-9_][A-Za-z0-9_-]{0,63}$" });
    if (!Number.isInteger(dimension) || dimension < 2 || dimension > 4096) {
      return json(res, 400, { error: "dimension 必须是 2-4096 的整数（等于浏览器端 embedding 模型维度）" });
    }
    const t0 = performance.now();
    await runtime.ensureCollection(name, dimension);
    return json(res, 201, { name, dimension, createMs: +(performance.now() - t0).toFixed(1) });
  }

  // /api/collections/:name/xxx
  const m = url.pathname.match(/^\/api\/collections\/([^/]+)(?:\/(.*))?$/);
  if (m) {
    const name = decodeURIComponent(m[1]);
    const action = m[2] || "";

    // ---- 删除集合 ----
    if (req.method === "DELETE" && !action) {
      await runtime.dropCollection(name);
      return json(res, 200, { deleted: name });
    }

    // ---- 写入文档（向量已在浏览器算好）----
    if (req.method === "POST" && action === "add") {
      const body = await readBody(req);
      const documents = body.documents;
      if (!Array.isArray(documents) || documents.length === 0) return json(res, 400, { error: "documents 为空" });
      for (const d of documents) {
        if (typeof d.id !== "string" || typeof d.text !== "string" || !Array.isArray(d.embedding)) {
          return json(res, 400, { error: "每条文档需含 id:string, text:string, embedding:number[]" });
        }
      }
      const collection = await runtime.getCollection(name);
      const t0 = performance.now();
      // 分批写入，避免单条 SQL 过大
      const BATCH = Number(process.env.SEEKDB_ADD_BATCH || 200);
      for (let i = 0; i < documents.length; i += BATCH) {
        const part = documents.slice(i, i + BATCH);
        await collection.add({
          ids: part.map((d) => d.id),
          documents: part.map((d) => d.text),
          metadatas: part.map((d) => d.metadata ?? {}),
          embeddings: part.map((d) => d.embedding),
        });
      }
      const writeMs = performance.now() - t0;
      // 索引刷新：保证新写入内容可检索（两层 HNSW：增量段 + 快照段）
      const t1 = performance.now();
      await collection.refresh_index();
      const refreshMs = performance.now() - t1;
      return json(res, 200, {
        added: documents.length,
        writeMs: +writeMs.toFixed(1),
        refreshMs: +(performance.now() - t1).toFixed(1),
        totalMs: +(performance.now() - t0).toFixed(1),
      });
    }

    // ---- 检索（hybrid / knn / fulltext 三种模式，均由 seekdb 执行）----
    if (req.method === "POST" && action === "search") {
      const body = await readBody(req);
      const { text, embedding, nResults = 10, mode = "hybrid" } = body;
      if (!text && !embedding) return json(res, 400, { error: "需要 text（全文检索用）或 embedding（向量检索用）" });
      const collection = await runtime.getCollection(name);

      const t0 = performance.now();
      let result;
      if (mode === "fulltext") {
        // 仅全文：仍走 hybridSearch（query=全文，无 knn），排序在库内完成
        result = await collection.hybridSearch({
          query: { whereDocument: { $contains: text }, nResults },
          rank: { rrf: {} },
          nResults,
          include: ["documents", "metadatas", "distances"],
        });
      } else if (mode === "knn") {
        // 仅向量：向量由浏览器计算，kNN 在库内执行
        result = await collection.hybridSearch({
          knn: { queryEmbeddings: embedding, nResults },
          rank: { rrf: {} },
          nResults,
          include: ["documents", "metadatas", "distances"],
        });
      } else {
        // 混合检索：全文 + 向量两路在 seekdb 内用 RRF 融合成单一排序列表。
        // rank_constant=20（默认 60）：头部排名权重更陡，利于把"双路都靠前"的文档顶到最前
        result = await collection.hybridSearch({
          query: text ? { whereDocument: { $contains: text }, nResults } : undefined,
          knn: embedding ? { queryEmbeddings: embedding, nResults } : undefined,
          rank: { rrf: { rankConstant: 20 } },
          nResults,
          include: ["documents", "metadatas", "distances"],
        });
      }
      const seekdbMs = performance.now() - t0;

      const items = [];
      const ids = result.ids[0] ?? [];
      for (let i = 0; i < ids.length; i++) {
        items.push({
          id: ids[i],
          score: result.distances?.[0]?.[i] ?? null,
          text: result.documents?.[0]?.[i] ?? null,
          metadata: result.metadatas?.[0]?.[i] ?? null,
        });
      }
      return json(res, 200, { mode, items, timings: { seekdbMs: +seekdbMs.toFixed(1) } });
    }

    // ---- 手动刷新索引 ----
    if (req.method === "POST" && action === "refresh") {
      const collection = await runtime.getCollection(name);
      const t0 = performance.now();
      await collection.refresh_index();
      return json(res, 200, { refreshMs: +(performance.now() - t0).toFixed(1) });
    }

    // ---- 文档计数 ----
    if (req.method === "GET" && action === "count") {
      const collection = await runtime.getCollection(name);
      return json(res, 200, { count: await collection.count() });
    }
  }

  json(res, 404, { error: `no route: ${route}` });
}

/* ---------------- static files ---------------- */

async function serveStatic(res, pathname) {
  let rel = decodeURIComponent(pathname);
  if (rel === "/") rel = "/index.html";
  const file = path.join(DIST_DIR, rel);
  if (!file.startsWith(DIST_DIR)) { res.writeHead(403); return res.end(); }
  try {
    const data = await fs.readFile(file);
    res.writeHead(200, { "content-type": MIME[path.extname(file)] ?? "application/octet-stream" });
    res.end(data);
  } catch {
    // SPA fallback
    try {
      const data = await fs.readFile(path.join(DIST_DIR, "index.html"));
      res.writeHead(200, { "content-type": MIME[".html"] });
      res.end(data);
    } catch {
      res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
      res.end("前端未构建：请先运行 npm run build（开发模式用 npm run dev）");
    }
  }
}

/* ---------------- main ---------------- */

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  try {
    if (req.method === "OPTIONS") {
      res.writeHead(204, {
        "access-control-allow-origin": "*",
        "access-control-allow-methods": "GET,POST,DELETE,OPTIONS",
        "access-control-allow-headers": "content-type",
      });
      return res.end();
    }
    if (url.pathname.startsWith("/api/")) return await handleApi(req, res, url);
    return await serveStatic(res, url.pathname);
  } catch (err) {
    console.error(`[api] ${req.method} ${url.pathname} 失败:`, err.message);
    json(res, 500, { error: err.message });
  }
});

const boot = await runtime.start();
server.listen(PORT, "127.0.0.1", () => {
  console.log(`[seekdb] 本地运行时已启动 (${boot.transport}${boot.transport === "unix_socket" ? " + TCP桥" : ""}) 127.0.0.1:${boot.port} | open ${boot.openMs}ms`);
  console.log(`[app]   http://127.0.0.1:${PORT}  (静态目录 ${DIST_DIR})`);
});

async function shutdown() {
  console.log("\n[shutdown] 关闭服务与 seekdb 运行时…");
  server.close();
  await runtime.close();
  process.exit(0);
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
