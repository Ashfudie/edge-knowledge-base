/** 后端 API 客户端（全部围绕 seekdb 集合） */

async function req(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${method} ${path} → HTTP ${res.status}`);
  return data;
}

export const api = {
  health: () => req("GET", "/api/health"),
  listCollections: () => req("GET", "/api/collections"),
  createCollection: (name, dimension) => req("POST", "/api/collections", { name, dimension }),
  deleteCollection: (name) => req("DELETE", `/api/collections/${encodeURIComponent(name)}`),
  count: (name) => req("GET", `/api/collections/${encodeURIComponent(name)}/count`),
  addDocuments: (name, documents) =>
    req("POST", `/api/collections/${encodeURIComponent(name)}/add`, { documents }),
  search: (name, payload) =>
    req("POST", `/api/collections/${encodeURIComponent(name)}/search`, payload),
  refreshIndex: (name) => req("POST", `/api/collections/${encodeURIComponent(name)}/refresh`),
};
