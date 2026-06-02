/**
 * arXiv API Client for OpenResearchOS v2.
 *
 * Searches for papers by category + keywords, fetches metadata via Atom XML,
 * and provides PDF/source download URLs.
 *
 * API docs: https://info.arxiv.org/help/api/user-manual.html
 * Rate limit: 3 seconds between requests (enforced here).
 */

const ARXIV_API = "https://export.arxiv.org/api/query";
const ARXIV_PDF_BASE = "https://arxiv.org/pdf";
const ARXIV_ABS_BASE = "https://arxiv.org/abs";

let lastRequestTime = 0;

async function rateLimitedFetch(url, timeoutMs = 20000) {
  const now = Date.now();
  const elapsed = now - lastRequestTime;
  if (elapsed < 3500) {
    await new Promise((r) => setTimeout(r, 3500 - elapsed));
  }
  lastRequestTime = Date.now();

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      signal: controller.signal,
      headers: { "User-Agent": "OpenResearchOS/2.0 (research agent)" },
    });
    if (!res.ok) throw new Error(`arXiv API error: ${res.status} ${res.statusText}`);
    return await res.text();
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Parse arXiv Atom XML response into paper objects.
 * Uses regex parsing (no XML lib needed in Node).
 */
function parseAtomEntries(xml) {
  const entries = [];
  const entryRegex = /<entry>([\s\S]*?)<\/entry>/g;
  let match;
  while ((match = entryRegex.exec(xml)) !== null) {
    const block = match[1];
    const get = (tag) => {
      const m = block.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`));
      return m ? m[1].trim() : "";
    };
    const getAttr = (tag, attr) => {
      const m = block.match(new RegExp(`<${tag}[^>]*?${attr}="([^"]*?)"`));
      return m ? m[1].trim() : "";
    };

    const id = get("id"); // e.g. http://arxiv.org/abs/2408.06292v1
    const arxivId = id.replace("http://arxiv.org/abs/", "").replace(/v\d+$/, "");
    const title = get("title").replace(/\s+/g, " ");
    const summary = get("summary").replace(/\s+/g, " ");
    const published = get("published"); // ISO date
    const year = published ? parseInt(published.slice(0, 4)) : null;

    // Authors
    const authorRegex = /<author>\s*<name>([^<]+)<\/name>/g;
    const authors = [];
    let am;
    while ((am = authorRegex.exec(block)) !== null) {
      authors.push(am[1].trim());
    }

    // Categories
    const catRegex = /category[^>]*term="([^"]+)"/g;
    const categories = [];
    let cm;
    while ((cm = catRegex.exec(block)) !== null) {
      categories.push(cm[1]);
    }

    // PDF link
    const pdfLink = getAttr('link[^>]*title="pdf"', "href") ||
      `${ARXIV_PDF_BASE}/${arxivId}.pdf`;

    entries.push({
      arxiv_id: arxivId,
      title,
      abstract: summary,
      authors,
      year,
      published,
      categories,
      url: `${ARXIV_ABS_BASE}/${arxivId}`,
      pdf_url: pdfLink.startsWith("http") ? pdfLink : `${ARXIV_PDF_BASE}/${arxivId}.pdf`,
      source: "arxiv_api",
    });
  }
  return entries;
}

/**
 * Search arXiv for papers matching a query.
 *
 * @param {string} query - Search terms
 * @param {object} opts
 * @param {string[]} opts.categories - arXiv categories like ["cs.AI", "cs.LG"]
 * @param {number} opts.maxResults - Max papers to return (default 25)
 * @param {string} opts.sortBy - "relevance" | "submittedDate" | "lastUpdatedDate"
 * @param {string} opts.sortOrder - "descending" | "ascending"
 * @returns {Promise<object[]>}
 */
export async function search(query, opts = {}) {
  const {
    categories = [],
    maxResults = 25,
    sortBy = "relevance",
    sortOrder = "descending",
  } = opts;

  // Build query string
  let searchQuery = `all:${encodeURIComponent(query)}`;
  if (categories.length > 0) {
    const catFilter = categories.map((c) => `cat:${c}`).join("+OR+");
    searchQuery = `(${searchQuery})+AND+(${catFilter})`;
  }

  const url = `${ARXIV_API}?search_query=${searchQuery}&sortBy=${sortBy}&sortOrder=${sortOrder}&max_results=${maxResults}`;
  console.error(`  [arXiv] Searching: ${query} (categories: ${categories.join(", ") || "all"}, max: ${maxResults})`);

  const xml = await rateLimitedFetch(url);
  const entries = parseAtomEntries(xml);
  console.error(`  [arXiv] Found ${entries.length} papers`);
  return entries;
}

/**
 * Search for the latest papers in specific categories.
 */
export async function searchLatest(categories, maxResults = 20) {
  const catFilter = categories.map((c) => `cat:${c}`).join("+OR+");
  const url = `${ARXIV_API}?search_query=${catFilter}&sortBy=submittedDate&sortOrder=descending&max_results=${maxResults}`;
  console.error(`  [arXiv] Fetching latest from: ${categories.join(", ")}`);

  const xml = await rateLimitedFetch(url);
  return parseAtomEntries(xml);
}

/**
 * Fetch metadata for a specific arXiv paper by ID.
 *
 * @param {string} arxivId - e.g. "2408.06292"
 */
export async function fetchPaper(arxivId) {
  const cleanId = arxivId.replace(/^arxiv:/i, "").replace(/v\d+$/, "");
  const url = `${ARXIV_API}?id_list=${cleanId}`;
  const xml = await rateLimitedFetch(url);
  const entries = parseAtomEntries(xml);
  return entries[0] || null;
}

/**
 * Multi-query search: runs multiple queries and deduplicates.
 */
export async function multiSearch(queries, opts = {}) {
  const { perQuery = 15, categories = ["cs.AI", "cs.LG", "cs.CL"] } = opts;
  const seen = new Set();
  const results = [];

  for (const query of queries) {
    try {
      const papers = await search(query, { categories, maxResults: perQuery });
      for (const paper of papers) {
        if (!seen.has(paper.arxiv_id)) {
          seen.add(paper.arxiv_id);
          results.push(paper);
        }
      }
    } catch (err) {
      console.error(`  [arXiv] Error searching "${query}":`, err.message);
    }
  }

  return results;
}

/**
 * Get PDF download URL for a paper.
 */
export function getPdfUrl(arxivId) {
  const cleanId = arxivId.replace(/^arxiv:/i, "").replace(/v\d+$/, "");
  return `${ARXIV_PDF_BASE}/${cleanId}.pdf`;
}

/**
 * Get the LaTeX source download URL.
 */
export function getSourceUrl(arxivId) {
  const cleanId = arxivId.replace(/^arxiv:/i, "").replace(/v\d+$/, "");
  return `https://arxiv.org/e-print/${cleanId}`;
}
