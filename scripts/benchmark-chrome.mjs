#!/usr/bin/env node
/**
 * Chrome 自动化评测：驱动系统 Chrome（playwright-core + channel:chrome）打开本应用，
 * 依次在多种「后端 × 精度」配置下运行应用内置评测，收集质量与延迟数据。
 *
 * 覆盖配置：
 *   1. WebGPU + q8     （主配置）
 *   2. WebGPU + fp16   （量化对比）
 *   3. WebGPU + fp32   （量化前基线）
 *   4. WASM  + q8      （降级路径；等价于无 WebGPU 环境的实际推理路径）
 *   5. 自动后端 + q8，且以 --disable-unsafe-webgpu 启动浏览器（验证自动降级与界面提示）
 *
 * 每个配置新建浏览器上下文（冷启动，记录模型首次加载时间），连续运行两轮评测
 * （第一轮冷、第二轮预热后），并记录 Chrome 版本。
 *
 * 用法：先确保后端已启动（PORT=8787 node server/index.mjs，且已 npm run build），
 *       然后 node scripts/benchmark-chrome.mjs
 * 输出：test-results/benchmark-chrome.json
 */
import { chromium } from "playwright-core";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = decodeURIComponent(path.join(path.dirname(fileURLToPath(import.meta.url)), ".."));
const APP_URL = process.env.APP_URL || "http://127.0.0.1:8787/";
const OUT_DIR = path.join(ROOT, "test-results");

const CONFIGS = [
  { label: "webgpu-fp16", backend: "webgpu", dtype: "fp16", args: [] },
  { label: "webgpu-fp32", backend: "webgpu", dtype: "fp32", args: [] },
  { label: "webgpu-q8", backend: "webgpu", dtype: "q8", args: [] },
  { label: "wasm-fp16", backend: "wasm", dtype: "fp16", args: [] },
  { label: "wasm-q8", backend: "wasm", dtype: "q8", args: [] },
  // 模拟"浏览器不支持 WebGPU"的环境：页面加载前删除 navigator.gpu，验证 auto 模式的自动降级
  { label: "nogpu-auto-fp16", backend: "auto", dtype: "fp16", args: [], noWebGPU: true },
];

const browserInfo = { userAgent: null, chromeVersion: null };

async function waitModelReady(page, timeout = 120000, expectSnippet = "维") {
  const start = Date.now();
  await page.waitForFunction(
    (snippet) => {
      const el = document.getElementById("model-status");
      return el && el.textContent.includes(snippet);
    },
    expectSnippet,
    { timeout, polling: 500 }
  );
  return Date.now() - start;
}

async function runOnce(page) {
  await page.evaluate(() => document.getElementById("btn-benchmark").click());
  await page.waitForFunction(
    () => {
      const el = document.getElementById("benchmark-status");
      return el && (el.textContent.includes("完成") || el.textContent.includes("失败"));
    },
    { timeout: 300000, polling: 1000 }
  );
  const status = await page.evaluate(() => document.getElementById("benchmark-status").textContent);
  if (status.includes("失败")) throw new Error(`评测失败: ${status}`);
  const round = await page.evaluate(() => JSON.parse(JSON.stringify(window.__benchRounds.at(-1))));
  return round;
}

async function runConfig(cfg) {
  console.log(`\n===== ${cfg.label} (${cfg.backend}/${cfg.dtype}) =====`);
  const browser = await chromium.launch({
    channel: "chrome",
    headless: false,
    args: ["--no-first-run", "--disable-search-engine-choice-screen", ...cfg.args],
  });
  try {
    const context = await browser.newContext();
    if (cfg.noWebGPU) {
      // 在任何页面脚本运行前移除 WebGPU 入口，等价于"浏览器不支持 WebGPU"的环境
      await context.addInitScript(() => {
        try { Object.defineProperty(navigator, "gpu", { get: () => undefined, configurable: true }); } catch {}
      });
    }
    const page = await context.newPage();
    if (!browserInfo.userAgent) {
      browserInfo.userAgent = await page.evaluate(() => navigator.userAgent);
      browserInfo.chromeVersion = (await browser.version()) + " | " + browserInfo.userAgent.match(/Chrome\/[\d.]+/)?.[0];
    }

    const t0 = Date.now();
    await page.goto(APP_URL, { waitUntil: "domcontentloaded" });
    const modelWaitMs = await waitModelReady(page);

    const badge = await page.textContent("#backend-badge");
    const modelStatus = await page.textContent("#model-status");

    // 选择后端与精度（切换会触发模型重载：等待状态文本出现新精度标记）
    await page.selectOption("#backend-select", cfg.backend);
    await page.selectOption("#precision-select", cfg.dtype);
    const switchWaitMs = await waitModelReady(page, 180000, `· ${cfg.dtype} ·`);
    await page.waitForTimeout(1500); // 状态渲染稳定后再读数
    const badgeAfter = await page.textContent("#backend-badge");
    const modelStatusAfter = await page.textContent("#model-status");

    const round1 = await runOnce(page); // 冷（建集合 + 导入 + 查询）
    const round2 = await runOnce(page); // 预热后
    console.log(`  徽章: ${badgeAfter.trim()}`);
    console.log(`  模型: ${modelStatusAfter.trim()}`);
    console.log(`  冷轮  R@10=${round1.metrics["recall@10"]} nDCG=${round1.metrics["ndcg@10"]} e2e中位=${round1.latencies.e2eMs.median}ms 吞吐=${round1.importTimings.throughputPerSec}/s`);
    console.log(`  热轮  R@10=${round2.metrics["recall@10"]} nDCG=${round2.metrics["ndcg@10"]} e2e中位=${round2.latencies.e2eMs.median}ms 吞吐=${round2.importTimings.throughputPerSec}/s`);

    return {
      label: cfg.label,
      requestedBackend: cfg.backend,
      dtype: cfg.dtype,
      browserArgs: cfg.args,
      initialBadge: badge.trim(),
      badge: badgeAfter.trim(),
      modelStatus: modelStatusAfter.trim(),
      pageLoadToModelReadyMs: modelWaitMs,
      backendSwitchReloadMs: switchWaitMs,
      coldRound: round1,
      warmRound: round2,
    };
  } finally {
    await browser.close();
  }
}

mkdirSync(OUT_DIR, { recursive: true });
const results = [];
for (const cfg of CONFIGS) {
  try {
    results.push(await runConfig(cfg));
  } catch (e) {
    results.push({ label: cfg.label, error: String(e.message || e) });
    console.error(`  ✗ ${cfg.label} 失败: ${e.message}`);
  }
}
const out = {
  meta: {
    appUrl: APP_URL,
    chrome: browserInfo.chromeVersion,
    device: "macOS (Apple Silicon)",
    timestamp: new Date().toISOString(),
    note: "数据来自应用的实际运行：应用内置评测面板（向量化在浏览器本地，检索由 seekdb 执行）",
  },
  results,
};
const outPath = path.join(OUT_DIR, "benchmark-chrome.json");
writeFileSync(outPath, JSON.stringify(out, null, 2));
console.log(`\n结果已写入 ${outPath}`);
