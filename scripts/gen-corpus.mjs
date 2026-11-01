#!/usr/bin/env node
/**
 * 评测集生成脚本：生成 web/public/eval-corpus.json
 *
 * 结构：
 *   corpus: [{id, text, meta:{source,lang,topic}}]
 *   queries: [{id, text, relevant:[corpusId...]}]
 *
 * 设计：
 *   - 18 个独立技术知识点 × 中英两种语言 = 36 条锚点文档（每条含一个可验证的独立事实）
 *   - 24 条干扰文档（其他主题，不应被查询命中）
 *   - 每条锚点文档配 1 条自然语言查询（措辞与原文刻意不同，考察语义而非关键词重叠），
 *     relevant 标注同一知识点的中英两条文档（混合检索理想情况两条都召回）
 *   - 查询语言与目标语言一致（另含少量跨语言查询，标注同一知识点两条）
 *
 * 运行：node scripts/gen-corpus.mjs
 */
import { writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

// ---- 18 个知识点：{topic, zh, en, qZh, qEn, qCross?} ----
const KNOWLEDGE = [
  {
    topic: "ob-paxos",
    zh: "OceanBase 通过 Paxos 多数派协议在多个副本之间复制事务日志，任一副本故障时只要多数派存活，数据就不会丢失，服务可以继续。Paxos 是 OceanBase 高可用能力的基石。",
    en: "OceanBase replicates transaction logs across replicas using the Paxos majority protocol. As long as a majority of replicas survive any failure, no data is lost and the service keeps running. Paxos is the cornerstone of OceanBase high availability.",
    qZh: "OceanBase 用哪种一致性协议在副本间复制日志来保证不丢数据？",
    qEn: "Which consensus protocol does OceanBase use to replicate logs across replicas?",
  },
  {
    topic: "seekdb-hnsw",
    zh: "seekdb 的向量检索默认使用 HNSW 图索引，写入时分为增量段与快照段两层：新数据先进入可检索的增量段，后台合并成快照段，兼顾写入吞吐与查询性能。",
    en: "seekdb uses HNSW graph indexes for vector search. Writes go through a two-level structure: an incremental segment that is immediately searchable, and background snapshot segments, balancing write throughput with query performance.",
    qZh: "seekdb 的向量索引是什么类型？新增数据如何变成可检索的？",
    qEn: "What type of vector index does seekdb use and how do new writes become searchable?",
  },
  {
    topic: "seekdb-hybrid",
    zh: "seekdb 的混合检索可以在一条 SQL 里同时执行全文检索与向量近邻检索，并用 RRF（倒数排序融合）把两路结果合并成单一排序，无需应用层分别取回再拼接。",
    en: "seekdb hybrid search executes full-text and vector nearest-neighbor retrieval in a single SQL statement and fuses the two ranked lists with Reciprocal Rank Fusion (RRF), so applications never merge results themselves.",
    qZh: "seekdb 如何在数据库内把全文和向量两路检索结果合并排序？",
    qEn: "How does seekdb combine full-text and vector results into one ranking inside the database?",
  },
  {
    topic: "webgpu-standard",
    zh: "WebGPU 是 W3C 制定的浏览器图形与通用计算标准，是 WebGL 的继任者。它把现代 GPU API（Vulkan、Metal、Direct3D 12）的能力带进浏览器，并支持通用计算着色器，因此可以用来跑神经网络推理。",
    en: "WebGPU is a W3C standard for browser graphics and general-purpose compute, succeeding WebGL. It exposes modern GPU APIs (Vulkan, Metal, Direct3D 12) to the web and supports compute shaders, which makes neural network inference in the browser practical.",
    qZh: "浏览器里继任 WebGL、可用于通用 GPU 计算的标准叫什么？",
    qEn: "What is the browser standard that succeeds WebGL and enables general-purpose GPU compute?",
  },
  {
    topic: "ort-web-backends",
    zh: "ONNX Runtime Web 提供 WebGPU 与 WASM 两种执行后端：优先尝试 WebGPU 获得 GPU 加速；环境不支持时自动回退到 WASM 在 CPU 上执行。应用应当显式指定设备并在界面提示当前后端。",
    en: "ONNX Runtime Web ships two execution backends: WebGPU for GPU acceleration and WebAssembly for CPU. When the environment lacks WebGPU support it falls back to WASM. Applications should pin the device explicitly and surface the active backend in the UI.",
    qZh: "ONNX Runtime Web 在浏览器里有哪些执行后端？降级策略是什么？",
    qEn: "Which execution backends does ONNX Runtime Web provide in the browser and what is the fallback strategy?",
  },
  {
    topic: "int8-quantization",
    zh: "INT8 量化把模型权重与激活从 32 位浮点压缩到 8 位整数，模型体积缩至约四分之一，内存带宽需求同步下降；代价是数值精度损失，通常需要评测量化前后检索质量（如 Recall@K）的变化。",
    en: "INT8 quantization compresses model weights and activations from 32-bit floats to 8-bit integers, shrinking model size to roughly a quarter and cutting memory bandwidth. The cost is numeric precision, so retrieval quality before and after quantization should be evaluated with metrics such as Recall@K.",
    qZh: "把模型权重压到 8 位整数有什么收益和代价？",
    qEn: "What are the benefits and costs of compressing model weights to 8-bit integers?",
  },
  {
    topic: "transformers-js",
    zh: "Transformers.js 是 Hugging Face 推出的 JavaScript 端侧推理库，直接在浏览器或 Node.js 中运行 ONNX 格式模型，覆盖文本嵌入、分类、生成等任务，无需服务端 API。",
    en: "Transformers.js is Hugging Face's JavaScript inference library that runs ONNX models directly in the browser or Node.js, covering embeddings, classification and generation without any server-side API.",
    qZh: "Hugging Face 那个能在浏览器直接跑 ONNX 模型的 JS 库叫什么？",
    qEn: "What is the Hugging Face JavaScript library that runs ONNX models directly in the browser?",
  },
  {
    topic: "bge-embedding",
    zh: "BGE 系列（BAAI 通用文本向量）是国内常用的开源文本嵌入模型家族，其中 bge-small-zh-v1.5 约 2400 万参数、输出 512 维向量，取 CLS 位置的隐状态并做 L2 归一化后用于语义相似度检索。",
    en: "The BGE family from BAAI is a widely used open-source text embedding model series. bge-small-zh-v1.5 has about 24 million parameters and outputs 512-dimensional vectors, taking the CLS hidden state followed by L2 normalization for semantic similarity search.",
    qZh: "bge-small-zh-v1.5 输出多少维向量？取哪个位置的表示？",
    qEn: "How many dimensions does bge-small-zh-v1.5 output and which token representation does it use?",
  },
  {
    topic: "rrf-fusion",
    zh: "RRF（Reciprocal Rank Fusion）是一种仅依赖排名的融合算法：对每路检索结果，按名次贡献 1/(k+rank) 的分数后求和。它不需要分数归一化，对量纲不同的多路检索（如 BM25 与余弦相似度）尤其稳健。",
    en: "Reciprocal Rank Fusion (RRF) is a rank-only fusion algorithm: each retrieval result contributes 1/(k+rank) and the scores are summed. It needs no score normalization, which makes it robust for merging retrievers with incompatible score scales such as BM25 and cosine similarity.",
    qZh: "RRF 融合算法为什么不需要对两路检索的分数做归一化？",
    qEn: "Why does RRF fusion not require score normalization across retrieval channels?",
  },
  {
    topic: "ik-analyzer",
    zh: "IK 分词器是面向中文的倒排索引分析器，内置词典与二元歧义处理，能把中文连续文本切分为可索引的词元。seekdb 的全文索引默认使用 IK 分析器处理中文。",
    en: "The IK analyzer is a Chinese-oriented analyzer for inverted indexes. With a built-in dictionary and ambiguity handling, it segments continuous Chinese text into indexable tokens. seekdb uses IK as the default analyzer for its full-text index.",
    qZh: "seekdb 全文索引默认用什么分析器切分中文？",
    qEn: "Which analyzer does seekdb use by default to segment Chinese text for full-text search?",
  },
  {
    topic: "hnsw-params",
    zh: "HNSW 图索引的关键构建参数是 M（每节点连边数）与 ef_construction（建图时候选队列长度）：更大的 M 和 ef_construction 提升召回率但增加内存与构建时间；查询时的 ef_search 则在延迟与召回之间权衡。",
    en: "The key build parameters of an HNSW graph index are M (edges per node) and ef_construction (candidate queue length during graph construction). Larger values improve recall at the cost of memory and build time; ef_search at query time trades latency for recall.",
    qZh: "HNSW 索引里 M 和 ef_construction 分别控制什么？",
    qEn: "What do M and ef_construction control in an HNSW index?",
  },
  {
    topic: "mysql-protocol",
    zh: "seekdb 兼容 MySQL 协议与 SQL 语法，现有 MySQL 生态的驱动、ORM 与工具可以直接连接使用。这大幅降低了向量数据库接入现有业务系统的迁移成本。",
    en: "seekdb is compatible with the MySQL wire protocol and SQL dialect, so existing MySQL drivers, ORMs and tooling can connect directly. This dramatically lowers the migration cost of introducing a search database into existing systems.",
    qZh: "seekdb 兼容哪种数据库通信协议？对存量系统有什么好处？",
    qEn: "Which database wire protocol is seekdb compatible with, and why does that matter for existing systems?",
  },
  {
    topic: "embed-batching",
    zh: "端侧文档向量化的重要优化是批处理：把多条文本一次性送入模型推理，摊薄每次调用的固定开销，吞吐通常可提升数倍。批大小受限于设备内存与序列长度，需要实测选择。",
    en: "A key optimization for on-device document embedding is batching: feeding multiple texts into the model per inference call amortizes fixed per-call overhead, often multiplying throughput several-fold. The batch size is bounded by device memory and sequence length and should be chosen empirically.",
    qZh: "端侧做文档向量化时批处理为什么能明显提升吞吐？",
    qEn: "Why does batching significantly improve throughput for on-device document embedding?",
  },
  {
    topic: "model-cache",
    zh: "浏览器会把 HTTP 缓存控制头允许的模型文件缓存在磁盘上，首次加载模型需要网络下载，之后的会话命中缓存后加载耗时主要来自反序列化与 GPU 着色器编译，因此性能报告应区分首次与预热后两种情况。",
    en: "Browsers cache model files on disk according to HTTP cache headers. The first load downloads over the network; later sessions hit the cache, where load time is dominated by deserialization and GPU shader compilation. Performance reports should therefore separate cold and warm numbers.",
    qZh: "为什么模型加载性能要区分首次加载和预热后两种情况？",
    qEn: "Why should model-loading performance be reported separately for cold and warm starts?",
  },
  {
    topic: "cosine-distance",
    zh: "文本嵌入通常经 L2 归一化后用余弦相似度比较。归一化后余弦相似度与内积等价，很多向量索引因此直接用内积实现以减少计算量。距离越小表示语义越接近。",
    en: "Text embeddings are usually L2-normalized and compared with cosine similarity. After normalization, cosine similarity becomes equivalent to the inner product, so many vector indexes implement inner product directly to save computation. Smaller distance means closer semantics.",
    qZh: "嵌入向量做 L2 归一化之后，余弦相似度和内积是什么关系？",
    qEn: "After L2 normalization, what is the relationship between cosine similarity and inner product?",
  },
  {
    topic: "chunking-strategy",
    zh: "知识库导入长文档时需要分块：块太大稀释语义、超出模型最大长度，块太小丢失上下文。常见做法是按段落边界切分，超长段落再按固定长度加重叠切分，并把来源与块序号写入元数据以便溯源。",
    en: "Long documents must be chunked before ingestion into a knowledge base: oversized chunks dilute semantics and exceed model context limits, while undersized chunks lose context. A common approach splits on paragraph boundaries, then applies fixed-size overlapping cuts for long paragraphs, storing source and chunk index as metadata for provenance.",
    qZh: "知识库导入时长文档分块要权衡什么？常见的切分方法是什么？",
    qEn: "What trade-offs matter when chunking long documents for a knowledge base, and what is the common approach?",
  },
  {
    topic: "index-refresh",
    zh: "向量数据库的两层索引（增量段+快照段）意味着写入成功不等于立即可检索：增量段需要显式或定时的索引刷新才会参与查询。应用应在新文档写入并刷新后验证可检索性，避免“写入成功却搜不到”的一致性问题。",
    en: "With two-level vector indexes (incremental plus snapshot segments), a successful write is not immediately searchable: the incremental segment must be refreshed—explicitly or on a timer—before queries can see it. Applications should verify searchability after writes and refreshes to avoid consistency surprises.",
    qZh: "向量写入成功后为什么可能暂时搜不到？应用应该怎么处理？",
    qEn: "Why may freshly written vectors be temporarily unsearchable, and what should applications do about it?",
  },
  {
    topic: "edge-privacy",
    zh: "在浏览器本地完成文档向量化意味着语料不必上传服务器，只有向量与原文写入用户自建的数据库。对隐私敏感场景（个人笔记、内部文档），端侧推理从架构上消除了数据出域的环节。",
    en: "Computing document embeddings locally in the browser means the corpus never leaves the device; only vectors and text go into the user's own database. For privacy-sensitive scenarios such as personal notes or internal documents, on-device inference architecturally eliminates the data-egress step.",
    qZh: "端侧向量化对隐私保护有什么架构上的意义？",
    qEn: "What is the architectural significance of on-device embedding for privacy?",
  },
];

// ---- 干扰文档（不应命中任何查询）----
const DISTRACTORS = [
  ["zh", "红烧肉的经典做法是先焯水去腥，再炒糖色上色，小火慢炖四十分钟收汁。肥而不腻的关键在于煸炒出多余油脂。"],
  ["en", "The classic beef bourguignon recipe sears the meat first, then braises it slowly in red wine with pearl onions and mushrooms for about three hours."],
  ["zh", "西湖位于浙江省杭州市，苏堤春晓、断桥残雪是著名的西湖十景之一。2011 年西湖文化景观被列入世界遗产名录。"],
  ["en", "Lake Geneva sits between Switzerland and France, famous for its vineyard terraces of Lavaux, a UNESCO World Heritage site since 2007."],
  ["zh", "马拉松全程 42.195 公里，起源于公元前希腊士兵从马拉松平原跑回雅典报信的传说，1896 年首届现代奥运会设立该项目。"],
  ["en", "The Tour de France covers more than 3,000 kilometers over three weeks each July, making it the most prestigious stage race in professional road cycling."],
  ["zh", "蜂鸟是唯一能真正倒飞的鸟类，翅膀每秒可扇动数十次，新陈代谢率极高，必须全天持续进食花蜜维持能量。"],
  ["en", "The Arctic tern completes the longest known migration of any animal, flying from the Arctic to the Antarctic and back, over 70,000 km annually."],
  ["zh", "宋代画家张择端的风俗画长卷《清明上河图》描绘了北宋汴京的城市面貌，全卷五米多长，藏于北京故宫博物院。"],
  ["en", "Vincent van Gogh painted The Starry Night in 1889 while staying at the asylum in Saint-Remy; it now hangs in the Museum of Modern Art in New York."],
  ["zh", "光合作用中植物利用叶绿素把二氧化碳和水转化为葡萄糖与氧气，光反应在类囊体膜上进行，暗反应在基质中完成。"],
  ["en", "Plate tectonics describes the movement of lithospheric plates over the asthenosphere, explaining continental drift, earthquakes and volcanic arcs."],
  ["zh", "广义相对论指出引力是质量弯曲时空的几何效应，1919 年日全食观测到的星光偏折是这一理论最早的实验验证之一。"],
  ["en", "Quantum entanglement correlates the states of separated particles such that measuring one instantly constrains the other, a phenomenon Einstein called spooky action at a distance."],
  ["zh", "二十四节气是依据太阳在黄道上的位置划分的补充历法，2016 年被列入联合国教科文组织人类非物质文化遗产名录。"],
  ["en", "The Fibonacci sequence appears in sunflower seed arrangements and pinecone spirals because it provides the optimal packing of florets at the golden angle."],
  ["zh", "人民币的防伪特征包括光变镂空开窗安全线、光彩光变面额数字和水印人像，可通过倾斜观察颜色变化来辨别。"],
  ["en", "The Federal Reserve raised interest rates several times during 2023 to curb inflation, which cooled the housing market and strengthened the dollar."],
  ["zh", "咖啡的苦味主要来自烘焙过程中绿原酸降解产生的咖啡酸与奎宁酸，烘焙度越深，苦味物质越多，酸味越少。"],
  ["en", "Espresso is brewed by forcing hot water at about 9 bars through finely ground coffee, extracting oils and dissolved solids into a concentrated shot with crema."],
  ["zh", "故宫太和殿俗称金銮殿，是明清两代皇帝举行大典的场所，殿前广场可容纳数万人，是现存中国最大的木结构大殿之一。"],
  ["en", "The Colosseum in Rome could hold some 50,000 spectators and featured a retractable awning operated by sailors from the imperial fleet."],
  ["zh", "睡眠周期由浅睡、深睡与快速眼动期构成，约 90 分钟一轮。深度睡眠帮助身体修复，快速眼动期与记忆巩固密切相关。"],
  ["en", "Honey never spoils: its low water content and acidic pH create an environment hostile to bacteria, which is why edible honey has been found in ancient tombs."],
];

function build() {
  const corpus = [];
  const queries = [];
  for (const k of KNOWLEDGE) {
    corpus.push({ id: `doc_${k.topic}_zh`, text: k.zh, meta: { source: "eval/knowledge", lang: "zh", topic: k.topic } });
    corpus.push({ id: `doc_${k.topic}_en`, text: k.en, meta: { source: "eval/knowledge", lang: "en", topic: k.topic } });
    // 主指标按同语言标注（bge-small-zh 为中文模型，跨语言语义对齐不是其设计目标）；
    // 另一条语言的对应文档记入 crossLangRelevant，作为观察项单独报告命中率。
    queries.push({ id: `q_${k.topic}_zh`, text: k.qZh, relevant: [`doc_${k.topic}_zh`], crossLangRelevant: [`doc_${k.topic}_en`] });
    queries.push({ id: `q_${k.topic}_en`, text: k.qEn, relevant: [`doc_${k.topic}_en`], crossLangRelevant: [`doc_${k.topic}_zh`] });
  }
  DISTRACTORS.forEach(([lang, text], i) => {
    corpus.push({ id: `dis_${i}`, text, meta: { source: "eval/distractor", lang, topic: `distractor_${i}` } });
  });
  return {
    meta: {
      name: "edge-knowledge-base 评测集",
      generatedAt: new Date().toISOString(),
      description: "18 个技术知识点 × 中英双语锚点文档 + 24 条干扰文档；36 条查询（每知识点中英各一）标注双语相关文档",
      counts: { corpus: corpus.length, queries: queries.length, knowledgeTopics: KNOWLEDGE.length, distractors: DISTRACTORS.length },
    },
    corpus,
    queries,
  };
}

const out = build();
const outPath = decodeURIComponent(
  path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "web", "public", "eval-corpus.json")
);
mkdirSync(path.dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify(out, null, 2));
console.log(`评测集已生成: ${outPath}`);
console.log(`  corpus: ${out.corpus.length} 条（锚点 ${KNOWLEDGE.length * 2} + 干扰 ${DISTRACTORS.length}）`);
console.log(`  queries: ${out.queries.length} 条`);
