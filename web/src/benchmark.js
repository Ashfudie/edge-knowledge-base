/**
 * 内置评测：走真实应用链路（浏览器向量化 → 写入 seekdb → 索引刷新 → 混合检索），
 * 输出检索质量（Recall@K / nDCG@K）与分阶段延迟（中位数/P90），支持多轮对比与 JSON 导出。
 *
 * 评测集：web/public/eval-corpus.json（中英混合语料 + 查询 + 相关性标注，
 * 由 scripts/gen-corpus.mjs 生成，亦可替换为自定义同结构文件）。
 */
import { api } from "./api.js";
import { MODEL_NAME, MODEL_DIMENSION, modelReady, embedOne, embedBatch } from "./embedder.js";

const rounds = []; // 每轮一个结果对象
// 调试/分析用暴露口（也可用于外部自动化读取结果）
if (typeof window !== "undefined") window.__benchRounds = rounds;

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}
function p90(arr) {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(s.length * 0.9))];
}
const f1 = (x) => +x.toFixed(1);
const f3 = (x) => +x.toFixed(3);

function recallAtK(rankedIds, relevantSet, k) {
  const hit = rankedIds.slice(0, k).filter((id) => relevantSet.has(id)).length;
  return hit / relevantSet.size;
}
function ndcgAtK(rankedIds, relevantSet, k) {
  let dcg = 0;
  for (let i = 0; i < Math.min(k, rankedIds.length); i++) {
    if (relevantSet.has(rankedIds[i])) dcg += 1 / Math.log2(i + 2);
  }
  let idcg = 0;
  for (let i = 0; i < Math.min(k, relevantSet.size); i++) idcg += 1 / Math.log2(i + 2);
  return idcg ? dcg / idcg : 0;
}

/**
 * 运行一轮评测。state 由 main.js 传入（读取当前后端/精度）。
 */
export async function runBenchmark(state) {
  const status = $("benchmark-status");
  const out = $("benchmark-output");
  try {
    if (!modelReady()) throw new Error("模型未就绪");
    status.textContent = "载入评测集…";
    const evalSet = await (await fetch("/eval-corpus.json")).json();
    const corpus = evalSet.corpus;
    const queries = evalSet.queries;

    // 独立评测集合，避免污染用户数据；跑完删除
    const evalCollection = `kb_eval_${Date.now().toString(36)}`;
    await api.createCollection(evalCollection, MODEL_DIMENSION);

    // ---- 1) 导入：分块(0，语料已按块给出) → 向量化 → 写入 → 刷新 ----
    status.textContent = `向量化 ${corpus.length} 条语料…`;
    const { embeddings, totalMs: embedAllMs } = await embedBatch(
      corpus.map((d) => d.text), 16,
      (done, total, batchMs) => {
        status.textContent = `向量化 ${done}/${total}（批 ${f1(batchMs)}ms）`;
      }
    );
    status.textContent = "写入 seekdb…";
    const addResult = await api.addDocuments(
      evalCollection,
      corpus.map((d, i) => ({
        id: d.id, text: d.text, metadata: d.meta ?? { source: "eval" }, embedding: embeddings[i],
      }))
    );

    // ---- 2) 查询：先预热 3 条（分离首次与稳态），再全量计量 ----
    status.textContent = "预热查询…";
    for (const q of queries.slice(0, 3)) {
      const e = await embedOne(q.text);
      await api.search(evalCollection, { text: q.text, embedding: e, nResults: 10, mode: "hybrid" });
    }

    status.textContent = `评测 ${queries.length} 条查询…`;
    const embedLat = [], seekLat = [], e2eLat = [];
    const perQuery = [];
    for (let qi = 0; qi < queries.length; qi++) {
      const q = queries[qi];
      const t0 = performance.now();
      const emb = await embedOne(q.text);
      const t1 = performance.now();
      const res = await api.search(evalCollection, { text: q.text, embedding: emb, nResults: 10, mode: "hybrid" });
      const t2 = performance.now();
      embedLat.push(t1 - t0);
      seekLat.push(res.timings.seekdbMs);
      e2eLat.push(t2 - t0);
      perQuery.push({ q: q.id, ranked: res.items.map((it) => it.id) });
      status.textContent = `评测查询 ${qi + 1}/${queries.length}`;
    }

    // ---- 3) 指标（relevant=同语言目标文档；跨语言命中作为观察项单独统计）----
    const recalls5 = [], recalls10 = [], ndcgs10 = [];
    let crossLangHits = 0, crossLangTotal = 0;
    for (const pq of perQuery) {
      const q = queries.find((x) => x.id === pq.q);
      const rel = new Set(q.relevant);
      recalls5.push(recallAtK(pq.ranked, rel, 5));
      recalls10.push(recallAtK(pq.ranked, rel, 10));
      ndcgs10.push(ndcgAtK(pq.ranked, rel, 10));
      if (q.crossLangRelevant) {
        crossLangTotal++;
        const hit = pq.ranked.slice(0, 10).some((id) => q.crossLangRelevant.includes(id));
        if (hit) crossLangHits++;
      }
    }
    const round = {
      timestamp: new Date().toISOString(),
      backend: state.backend,
      dtype: state.dtype,
      model: MODEL_NAME,
      corpusSize: corpus.length,
      queryCount: queries.length,
      importTimings: {
        vectorizeMs: f1(embedAllMs),
        throughputPerSec: +(corpus.length / (embedAllMs / 1000)).toFixed(1),
        writeMs: addResult.writeMs,
        refreshMs: addResult.refreshMs,
      },
      metrics: {
        // 质量指标取全查询平均（Recall/nDCG 为比例值，中位数无意义）；延迟另行用中位数
        "recall@5": f3(mean(recalls5)),
        "recall@10": f3(mean(recalls10)),
        "ndcg@10": f3(mean(ndcgs10)),
        ...(crossLangTotal
          ? { crossLangHitRate10: f3(crossLangHits / crossLangTotal) }
          : {}),
      },
      latencies: {
        embedMs: { median: f1(median(embedLat)), p90: f1(p90(embedLat)) },
        seekdbMs: { median: f1(median(seekLat)), p90: f1(p90(seekLat)) },
        e2eMs: { median: f1(median(e2eLat)), p90: f1(p90(e2eLat)) },
      },
      perQuery: perQuery.map((pq) => {
        const q = queries.find((x) => x.id === pq.q);
        return {
          q: pq.q,
          targetRank: pq.ranked.indexOf(q.relevant[0]) + 1 || null,
          top3: pq.ranked.slice(0, 3),
        };
      }),
    };
    rounds.push(round);

    // ---- 4) 清理评测集合，保持知识库状态一致 ----
    await api.deleteCollection(evalCollection);

    status.textContent = `完成 ✓（${round.backend}/${round.dtype} · 语料 ${round.corpusSize} · 查询 ${round.queryCount}）`;
    renderRounds();
    $("btn-export-results").classList.remove("hidden");
    return round;
  } catch (e) {
    status.textContent = `评测失败：${e.message}`;
    throw e;
  }
}

function renderRounds() {
  const out = $("benchmark-output");
  const crossCol = rounds.some((r) => r.metrics.crossLangHitRate10 != null);
  const crossHead = crossCol ? `<th>跨语言命中@10<br>（观察项）</th>` : "";
  const rows = rounds.map((r, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${r.backend}</td>
      <td>${r.dtype}</td>
      <td>${r.metrics["recall@5"]}</td>
      <td>${r.metrics["recall@10"]}</td>
      <td>${r.metrics["ndcg@10"]}</td>
      ${crossCol ? `<td>${r.metrics.crossLangHitRate10 ?? "-"}</td>` : ""}
      <td class="mono">${r.latencies.embedMs.median}</td>
      <td class="mono">${r.latencies.seekdbMs.median}</td>
      <td class="mono">${r.latencies.e2eMs.median}</td>
      <td class="mono">${r.importTimings.throughputPerSec}</td>
    </tr>`).join("");
  out.innerHTML = `
    <table class="bench-table">
      <thead><tr>
        <th>#</th><th>后端</th><th>精度</th>
        <th>Recall@5</th><th>Recall@10</th><th>nDCG@10</th>${crossHead}
        <th>查询向量化<br>中位数 ms</th><th>seekdb 检索<br>中位数 ms</th><th>端到端<br>中位数 ms</th><th>导入吞吐<br>条/秒</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="hint">同配置多跑几轮可观察稳定性；切换上方「精度/后端」后再运行即可对比量化与降级路径。P90 与首末次数据见导出 JSON。</div>`;
}

export function exportResults() {
  const blob = new Blob([JSON.stringify(rounds, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `edge-kb-benchmark-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

const $ = (id) => document.getElementById(id);
