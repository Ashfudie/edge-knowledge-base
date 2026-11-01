#!/usr/bin/env node
/**
 * 批处理优化对比实验：同一语料，批大小 1（无批处理基线） vs 16（批处理），
 * 在当前应用 UI 中实测向量化吞吐（embedding 批处理优化前后对比）。
 *
 * 前置：npm run build && npm start（默认配置 WebGPU+FP16）
 * 用法：node scripts/batch-ab.mjs
 */
import { chromium } from "playwright-core";

const APP_URL = process.env.APP_URL || "http://127.0.0.1:8787/";

const browser = await chromium.launch({ channel: "chrome", headless: false, args: ["--no-first-run"] });
const results = [];
try {
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.goto(APP_URL, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => document.getElementById("model-status")?.textContent.includes("维"),
    { timeout: 120000, polling: 500 }
  );
  console.log("模型就绪:", (await page.textContent("#model-status")).trim());

  // 用应用自身的导入流程跑：粘贴较长文本（分块数 = 块数），对比不同批大小的总耗时
  // 生成 ~48 块的文本（每块 ~120 字，空行分隔）
  const paragraph =
    "向量数据库把嵌入与倒排结合，混合检索在库内融合两路排序，端侧推理降低了数据出域风险，" +
    "浏览器里的 WebGPU 让通用计算无需安装运行时，模型的 INT8 与 FP16 量化在体积和精度间权衡。";
  const text = Array.from({ length: 48 }, (_, i) => `${paragraph}（样本 ${i + 1}）`).join("\n\n");

  for (const batchSize of [1, 4, 16, 32]) {
    await page.fill("#chunk-size", "130"); // 让 48 段各自成块
    await page.fill("#batch-size", String(batchSize));
    await page.fill("#import-text", text);
    await page.evaluate(() => document.getElementById("btn-import").click());
    await page.waitForFunction(
      () => !document.getElementById("import-summary").classList.contains("hidden"),
      { timeout: 300000, polling: 500 }
    );
    const summary = await page.textContent("#import-summary");
    const m = summary.match(/向量化 ([\d.]+)s?（/) || summary.match(/向量化 ([\d.]+)(ms|s)/);
    const throughput = parseFloat(summary.match(/([\d.]+) 块\/秒/)?.[1] ?? "0");
    results.push({ batchSize, throughputPerSec: throughput, summary: summary.trim().replace(/\s+/g, " ") });
    console.log(`batch=${batchSize}: ${throughput} 块/秒`);
    await page.waitForTimeout(1500);
  }
} finally {
  await browser.close();
}
console.log("\n结论数据:");
for (const r of results) console.log(`  batch=${String(r.batchSize).padEnd(3)} → ${r.throughputPerSec} 块/秒`);
