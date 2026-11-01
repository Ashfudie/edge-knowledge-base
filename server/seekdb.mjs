/**
 * seekdb 运行时封装：
 *  1. @oceanbase/seekdb-node 启动本地 seekdb server 进程（数据落本地目录，Unix socket 通信）
 *  2. 本地起一个 TCP↔UnixSocket 桥（seekdb-js 的 ConnectionConfig 仅支持 host/port）
 *  3. seekdb-js SeekdbClient 以 server 模式接入，Collection API 全部走数据库执行
 */
import { seekdb } from "@oceanbase/seekdb-node";
import { SeekdbClient, Schema, VectorIndexConfig, FulltextIndexConfig } from "seekdb";
import net from "node:net";
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const ROOT = path.resolve(fileURLToPath(new URL("../", import.meta.url)));

export class SeekdbRuntime {
  constructor(dataDir) {
    this.dataDir = dataDir;
    this.handle = null;
    this.bridgeServer = null;
    this.port = null;
    this.client = null;
    this.collectionCache = new Map(); // name -> { collection, dimension }
  }

  async start() {
    mkdirSync(this.dataDir, { recursive: true });
    const t0 = Date.now();
    this.handle = await seekdb.open(this.dataDir, {
      parameters: { memory_budget: process.env.SEEKDB_MEMORY_BUDGET || "2G" },
    });
    const openMs = Date.now() - t0;

    const ep = this.handle.connectionOptions();
    if (ep.transport === "tcp") {
      // 某些平台默认 TCP，直接用
      this.port = ep.port;
      this.host = ep.host;
    } else {
      // macOS/Linux 默认 unix socket：架桥转成 127.0.0.1 TCP 供 seekdb-js 使用
      const { host, port } = await this.#bridge(ep.unixSocket);
      this.host = host;
      this.port = port;
    }

    this.client = new SeekdbClient({
      host: this.host,
      port: this.port,
      user: ep.user,
      password: "",
      database: process.env.SEEKDB_DATABASE || "test",
    });
    return { openMs, host: this.host, port: this.port, transport: ep.transport };
  }

  #bridge(unixSocket) {
    return new Promise((resolve, reject) => {
      const server = net.createServer();
      server.on("error", reject);
      server.on("connection", (client) => {
        const upstream = net.connect(unixSocket);
        client.pipe(upstream).pipe(client);
        const cleanup = () => { client.destroy(); upstream.destroy(); };
        client.on("error", cleanup);
        upstream.on("error", cleanup);
        client.on("close", () => upstream.destroy());
        upstream.on("close", () => client.destroy());
      });
      server.listen(0, "127.0.0.1", () => {
        this.bridgeServer = server;
        resolve({ host: "127.0.0.1", port: server.address().port });
      });
    });
  }

  /**
   * 创建（或复用）知识库集合。
   * 向量维度由调用方传入（= 浏览器端 embedding 模型维度）；
   * embeddingFunction 显式为 null：向量一律由客户端计算后写入，数据库不做向量化。
   */
  async ensureCollection(name, dimension) {
    const cached = this.collectionCache.get(name);
    if (cached && (dimension == null || cached.dimension === dimension)) return cached.collection;

    const schema = new Schema()
      .createIndex(new FulltextIndexConfig("ik"))
      .createIndex(new VectorIndexConfig({
        // m/ef_construction 为 HNSW 建图默认质量参数；ef_search=256 提高查询时候选队列，
        // 改善近似检索的排序质量（60-数百条规模下延迟增幅可忽略）
        hnsw: { dimension, distance: "cosine", m: 16, ef_construction: 200, ef_search: 256 },
        embeddingFunction: null,
      }));

    const collection = await this.client.getOrCreateCollection({ name, schema });
    this.collectionCache.set(name, { collection, dimension });
    return collection;
  }

  async getCollection(name) {
    const cached = this.collectionCache.get(name);
    if (cached) return cached.collection;
    const collection = await this.client.getCollection({ name });
    this.collectionCache.set(name, { collection, dimension: null });
    return collection;
  }

  async listCollections() {
    // withEmbeddingFunction:false —— 本应用所有集合均为 embeddingFunction:null
    const cols = await this.client.listCollections({ withEmbeddingFunction: false });
    return cols;
  }

  async dropCollection(name) {
    this.collectionCache.delete(name);
    await this.client.deleteCollection(name);
  }

  async close() {
    try { await this.client?.close(); } catch {}
    this.bridgeServer?.close();
    this.handle?.close();
  }
}
