/**
 * 浏览器端 embedding 封装（Transformers.js / ONNX Runtime Web）：
 *  - 后端选择：优先 WebGPU，显式配置 device:'webgpu'；不可用时自动降级 WASM 并回调通知 UI
 *  - 量化：dtype q8 (INT8) / fp16 / fp32，模型文件从本站 /models/ 本地加载（离线可用）
 *  - 批处理与计时：embedBatch 提供吞吐统计；loadModel 区分加载耗时
 *
 * 模型：Xenova/bge-small-zh-v1.5（BAAI/bge-small-zh-v1.5 的 ONNX 导出，MIT），
 * CLS pooling + L2 归一化，512 维 —— 与服务端集合的 HNSW dimension 一致。
 */
import { pipeline, env } from "@huggingface/transformers";

// 模型与 ORT WASM 运行时全部从本站加载，不依赖外网 CDN。
// ORT 运行时资源（*.wasm/.mjs）不覆盖 wasmPaths：使用 transformers.js 默认路径，
// dev 模式由 vite 直接服务 node_modules 内文件，build 模式由 vite 打进 dist/assets。
env.allowLocalModels = true;
env.allowRemoteModels = false;

const MODEL_ID = "/models/bge-small-zh-v1.5/";
export const MODEL_DIMENSION = 512;
export const MODEL_NAME = "bge-small-zh-v1.5 (ONNX, Xenova 导出)";

let pipe = null;
let loadedKey = null; // `${device}:${dtype}` 防重复加载

/** 探测 WebGPU 是否真正可用（有 adapter 才算） */
export async function detectWebGPU() {
  try {
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

/**
 * 解析最终后端：auto → webgpu 可用则 webgpu，否则 wasm；显式指定则直接用（不可用时抛错）
 */
export async function resolveBackend(preference) {
  const webgpuOK = await detectWebGPU();
  if (preference === "webgpu") {
    if (!webgpuOK) throw new Error("当前环境不支持 WebGPU（无 navigator.gpu 或无可用适配器）");
    return "webgpu";
  }
  if (preference === "wasm") return "wasm";
  return webgpuOK ? "webgpu" : "wasm";
}

/**
 * 加载模型（指定后端+精度）。重复调用同配置直接复用；配置变化时重新加载。
 * 返回 { device, dtype, loadMs }；进度回调用于 UI 展示。
 */
export async function loadModel(device, dtype, onProgress) {
  const key = `${device}:${dtype}`;
  if (pipe && loadedKey === key) return { device, dtype, loadMs: 0 };
  if (pipe) {
    try { await pipe.dispose(); } catch {}
    pipe = null;
    loadedKey = null;
  }
  const t0 = performance.now();
  pipe = await pipeline("feature-extraction", MODEL_ID, {
    device,
    dtype,
    progress_callback: onProgress,
  });
  loadedKey = key;
  return { device, dtype, loadMs: +(performance.now() - t0).toFixed(1) };
}

export function modelReady() {
  return pipe !== null;
}

/** 单条文本 embedding（查询用） */
export async function embedOne(text) {
  const out = await pipe(text, { pooling: "cls", normalize: true });
  return Array.from(out[0].data);
}

/**
 * 批量 embedding（文档导入用）。
 * @param {string[]} texts
 * @param {number} batchSize 每次送入模型的条数
 * @param {(done:number,total:number,batchMs:number)=>void} onProgress
 * @returns {Promise<{embeddings:number[][], totalMs:number, batches:number}>}
 */
export async function embedBatch(texts, batchSize, onProgress) {
  if (!pipe) throw new Error("模型未加载");
  const embeddings = [];
  const t0 = performance.now();
  let batches = 0;
  for (let i = 0; i < texts.length; i += batchSize) {
    const part = texts.slice(i, i + batchSize);
    const b0 = performance.now();
    const out = await pipe(part, { pooling: "cls", normalize: true });
    for (let j = 0; j < part.length; j++) embeddings.push(Array.from(out[j].data));
    batches++;
    onProgress?.(Math.min(i + batchSize, texts.length), texts.length, performance.now() - b0);
    // 让出主线程，保证进度条可渲染
    await new Promise((r) => setTimeout(r, 0));
  }
  return { embeddings, totalMs: performance.now() - t0, batches };
}
