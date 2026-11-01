#!/usr/bin/env node
/**
 * 模型体积与压缩比统计（口径：ONNX 权重 + tokenizer/config 等全部随模型分发文件）。
 * 口径：ONNX 权重 + tokenizer/config 等全部随模型分发文件（不含 ORT 运行时本身）。
 * 运行：node scripts/size-report.mjs
 */
import { statSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const MODEL_DIR = decodeURIComponent(
  path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "web", "public", "models", "bge-small-zh-v1.5")
);

const PRECISIONS = {
  fp32: ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "onnx/model.onnx"],
  fp16: ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "onnx/model_fp16.onnx"],
  q8:   ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "onnx/model_quantized.onnx"],
};

const mb = (b) => b / 1048576;
const report = {};
for (const [name, files] of Object.entries(PRECISIONS)) {
  let total = 0;
  const perFile = {};
  for (const f of files) {
    const p = path.join(MODEL_DIR, f);
    if (!existsSync(p)) { perFile[f] = null; continue; }
    const size = statSync(p).size;
    perFile[f] = size;
    total += size;
  }
  report[name] = { files: perFile, totalBytes: total, totalMB: +mb(total).toFixed(2) };
}
if (report.fp32.totalBytes && report.q8.totalBytes) {
  report.compression = {
    fp32_to_q8: +(report.fp32.totalBytes / report.q8.totalBytes).toFixed(2),
    fp32_to_fp16: +(report.fp32.totalBytes / report.fp16.totalBytes).toFixed(2),
  };
}
console.log(JSON.stringify(report, (k, v) => (typeof v === "number" && v > 1024 ? +mb(v).toFixed(2) + "MB" : v), 2));
if (report.compression) {
  console.log(`\n压缩比: FP32→INT8 = ${report.compression.fp32_to_q8}x, FP32→FP16 = ${report.compression.fp32_to_fp16}x`);
} else {
  console.log("\n提示：缺少 FP32 基线文件，运行 npm run download-models-fp32 后再统计");
}
