#!/usr/bin/env node
/**
 * 模型获取脚本：从 HuggingFace（或镜像）下载 bge-small-zh-v1.5 的 ONNX 权重与 tokenizer。
 *
 * 用法:
 *   node scripts/download-models.mjs            # 下载 INT8(q8) + FP16（默认，vendor 进仓库的精度）
 *   node scripts/download-models.mjs --fp32     # 额外下载 FP32 基线（约 90MB，用于量化前后质量对比）
 *
 * 环境变量:
 *   HF_ENDPOINT  HF 端点，默认 https://hf-mirror.com（国内可达），海外可设为 https://huggingface.co
 *
 * 模型来源: https://huggingface.co/Xenova/bge-small-zh-v1.5 （BAAI/bge-small-zh-v1.5 的 ONNX 导出）
 * License: MIT (bge-small-zh-v1.5)
 */
import { mkdirSync, createWriteStream, existsSync, statSync, rmSync, renameSync } from "node:fs";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import process from "node:process";

const MODEL_REPO = "Xenova/bge-small-zh-v1.5";
const ENDPOINT = process.env.HF_ENDPOINT || "https://hf-mirror.com";
// 注意：URL.pathname 会把中文等非 ASCII 字符转义，必须 decodeURIComponent 还原
const OUT_DIR = decodeURIComponent(new URL("../web/public/models/bge-small-zh-v1.5/", import.meta.url).pathname);

const BASE_FILES = [
  "config.json",
  "tokenizer.json",
  "tokenizer_config.json",
  "special_tokens_map.json",
  "vocab.txt",
];
const PRECISION_FILES = {
  q8: "onnx/model_quantized.onnx",   // INT8 量化，~23MB
  fp16: "onnx/model_fp16.onnx",      // FP16，~45MB
  fp32: "onnx/model.onnx",           // FP32 原始权重，~90MB（仅 --fp32 时下载）
};

const wantFp32 = process.argv.includes("--fp32");
const precisions = wantFp32 ? ["q8", "fp16", "fp32"] : ["q8", "fp16"];
const files = [...BASE_FILES, ...precisions.map((p) => PRECISION_FILES[p])];

async function download(rel) {
  const url = `${ENDPOINT}/${MODEL_REPO}/resolve/main/${rel}`;
  const out = OUT_DIR + rel;
  mkdirSync(out.slice(0, out.lastIndexOf("/")), { recursive: true });
  if (existsSync(out) && statSync(out).size > 0) {
    console.log(`[skip] ${rel} 已存在`);
    return;
  }
  const tmp = out + ".tmp";
  process.stdout.write(`[down] ${rel} ... `);
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  await pipeline(Readable.fromWeb(res.body), createWriteStream(tmp));
  rmSync(out, { force: true });
  renameSync(tmp, out);
  console.log(`${(statSync(out).size / 1048576).toFixed(1)}MB`);
}

console.log(`模型仓库: ${MODEL_REPO}\n端点: ${ENDPOINT}\n输出: ${OUT_DIR}\n`);
for (const f of files) await download(f);
console.log("\n完成。模型目录结构:");
for (const p of precisions) console.log(`  ${PRECISION_FILES[p]}`);
