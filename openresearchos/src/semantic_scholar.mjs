/**
 * Semantic Scholar API client for OpenResearchOS.
 *
 * Provides deeper paper discovery than OpenAlex alone:
 *   - Full-text search with relevance ranking
 *   - Citation graph traversal (snowball citations)
 *   - Paper detail fetching (abstract, venue, citations, references)
 *   - Author and field-specific search
 *
 * Rate limits: 100 requests/5 minutes for unauthenticated.
 * We use conservative timing, exponential backoff, and small page sizes.
 */

const S2_BASE = "https://api.semanticscholar.org/graph/v1";
const S2_FIELDS = "title,abstract,year,venue,citationCount,url,externalIds,authors,publicationDate";

// Global flag to skip S2 if we've hit consecutive 429s
let s2Disabled = false;
let s2ConsecutiveFailures = 0;
const S2_MAX_CONSECUTIVE_FAILURES = 3;

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Fetch with timeout, exponential backoff on 429, and graceful failure.
 */
async function fetchWithTimeout(url, timeoutMs = 15000, retries = 2) {
  if (s2Disabled) {
    throw new Error("Semantic Scholar temporarily disabled (rate limit)");
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        "User-Agent": "OpenResearchOS/0.1 (research demo; contact: opensource)",
      },
    });

    if (response.status === 429) {
      s2ConsecutiveFailures += 1;
      if (s2ConsecutiveFailures >= S2_MAX_CONSECUTIVE_FAILURES) {
        s2Disabled = true;
        console.error(`  [S2] Rate limit hit ${s2ConsecutiveFailures} times. Disabling Semantic Scholar for this run.`);
        throw new Error("429 Too Many Requests (S2 disabled for run)");
      }
      const delay = 2000 * Math.pow(2, retries - 1); // 2s, 4s, 8s
      console.error(`[Semantic Scholar] Rate limited (429). Retrying in ${delay}ms...`);
      await sleep(delay);
      return fetchWithTimeout(url, timeoutMs, retries - 1);
    }

    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }

    // Reset on success
    s2ConsecutiveFailures = 0;
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Reset the disabled flag (useful before a fresh run).
 */
export function resetRateLimit() {
  s2Disabled = false;
  s2ConsecutiveFailures = 0;
}

/**
 * Search for papers by query string.
 * Returns normalized paper objects.
 */
export async function searchPapers(query, { limit = 10, yearFrom = 2018, fieldsOfStudy = "Computer Science" } = {}) {
  if (s2Disabled) return [];

  let url = `${S2_BASE}/paper/search?query=${encodeURIComponent(query)}&limit=${limit}&fields=${S2_FIELDS}`;
  if (yearFrom) url += `&year=${yearFrom}-`;
  if (fieldsOfStudy) url += `&fieldsOfStudy=${encodeURIComponent(fieldsOfStudy)}`;

  try {
    const json = await fetchWithTimeout(url);
    return (json.data || []).map(normalizePaper);
  } catch (error) {
    console.error(`Semantic Scholar search error for "${query}": ${error.message}`);
    return [];
  }
}


/**
 * Get detailed info for a specific paper by ID.
 */
export async function getPaper(paperId) {
  if (s2Disabled) return null;
  try {
    const json = await fetchWithTimeout(
      `${S2_BASE}/paper/${encodeURIComponent(paperId)}?fields=${S2_FIELDS},references.title,references.year,references.url,citations.title,citations.year,citations.url`
    );
    return normalizePaper(json);
  } catch (error) {
    console.error(`Semantic Scholar paper fetch error for "${paperId}": ${error.message}`);
    return null;
  }
}

/**
 * Snowball: get references and citations of a paper.
 * Returns { references: [...], citations: [...] }
 */
export async function snowballCitations(paperId, { maxRefs = 10, maxCites = 10 } = {}) {
  const paper = await getPaper(paperId);
  if (!paper) return { references: [], citations: [] };

  return {
    references: (paper._raw?.references || []).slice(0, maxRefs).map(normalizePaper),
    citations: (paper._raw?.citations || []).slice(0, maxCites).map(normalizePaper),
  };
}

/**
 * Search for papers by multiple queries in sequence.
 * De-duplicates by title (lowercased).
 * Pauses between queries to respect rate limits.
 */
export async function multiSearch(queries, { perQuery = 6, yearFrom = null, delayMs = 2000 } = {}) {
  const seen = new Set();
  const results = [];

  for (const query of queries) {
    if (s2Disabled) break;
    const papers = await searchPapers(query, { limit: perQuery, yearFrom });
    for (const paper of papers) {
      const key = paper.title.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
      if (key && !seen.has(key)) {
        seen.add(key);
        results.push(paper);
      }
    }
    // Rate limit: pause between queries (default 2s)
    if (!s2Disabled) await sleep(delayMs);
  }

  return results;
}

/**
 * Search specifically for competitors of an idea.
 * Generates targeted queries from the idea title only to minimize API calls.
 */
export async function searchCompetitors(ideaTitle, _ideaPitch, { limit = 5 } = {}) {
  if (s2Disabled) {
    console.error("  [S2] Skipping competitor search (S2 rate-limited for this run).");
    return [];
  }

  // Only use the title — the pitch generates redundant 429s
  const queries = [ideaTitle];
  const results = await multiSearch(queries, { perQuery: limit, delayMs: 2500 });
  return results.map(paper => ({
    ...paper,
    overlap_source: "semantic_scholar_competitor_search",
  }));
}

function normalizePaper(raw) {
  if (!raw) return null;
  const doi = raw.externalIds?.DOI || null;
  const arxivId = raw.externalIds?.ArXiv || null;

  let bestUrl = raw.url;
  if (arxivId && !bestUrl) bestUrl = `https://arxiv.org/abs/${arxivId}`;
  if (doi && !bestUrl) bestUrl = `https://doi.org/${doi}`;

  return {
    title: raw.title || "Untitled",
    year: raw.year || null,
    url: bestUrl || null,
    source: "semantic_scholar",
    semantic_scholar_id: raw.paperId || null,
    doi,
    arxiv_id: arxivId,
    cited_by_count: raw.citationCount || 0,
    abstract: raw.abstract || "",
    venue: raw.venue || null,
    authors: (raw.authors || []).map(a => a.name).slice(0, 5),
    publication_date: raw.publicationDate || null,
    _raw: raw,
  };
}

/**
 * ArXiv XML API search — returns basic metadata from Atom feed.
 */
export async function searchArxiv(query, { maxResults = 5 } = {}) {
  const url = `https://export.arxiv.org/api/query?search_query=all:${encodeURIComponent(query)}&max_results=${maxResults}&sortBy=relevance&sortOrder=descending`;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 12000);
    const response = await fetch(url, {
      signal: controller.signal,
      headers: { "User-Agent": "OpenResearchOS/2.0 (research agent)" },
    });
    clearTimeout(timeout);

    if (!response.ok) return [];
    const text = await response.text();

    const entries = [];
    const entryRegex = /<entry>([\s\S]*?)<\/entry>/g;
    let match;
    while ((match = entryRegex.exec(text)) !== null) {
      const entry = match[1];
      const titleMatch = entry.match(/<title>([\s\S]*?)<\/title>/);
      const idMatch = entry.match(/<id>https?:\/\/arxiv\.org\/abs\/([^<]+)<\/id>/);
      const abstractMatch = entry.match(/<summary>([\s\S]*?)<\/summary>/);
      const publishedMatch = entry.match(/<published>(\d{4})/);

      if (titleMatch && idMatch) {
        entries.push({
          title: titleMatch[1].trim().replace(/\s+/g, " "),
          year: publishedMatch ? parseInt(publishedMatch[1]) : null,
          url: `https://arxiv.org/abs/${idMatch[1].trim()}`,
          source: "arxiv",
          arxiv_id: idMatch[1].trim(),
          doi: null,
          cited_by_count: 0,
          abstract: abstractMatch ? abstractMatch[1].trim().replace(/\s+/g, " ").slice(0, 500) : "",
          venue: "arXiv",
          authors: [],
          semantic_scholar_id: null,
        });
      }
    }
    return entries;
  } catch {
    return [];
  }
}

// ─── v2 Additions: Bulk Search, Batch Fetch, Snowball ─────────────────────────

/**
 * Bulk search: uses GET /paper/search/bulk for large-scale retrieval.
 * Returns up to `maxPapers` papers with continuation token pagination.
 */
export async function bulkSearch(query, { maxPapers = 100, yearFrom = 2018, fieldsOfStudy = "Computer Science" } = {}) {
  if (s2Disabled) return [];
  const results = [];
  let continuationToken = null;
  let pages = 0;
  const MAX_PAGES = 3;

  while (pages < MAX_PAGES && results.length < maxPapers) {
    let url = `${S2_BASE}/paper/search/bulk?query=${encodeURIComponent(query)}&fields=${S2_FIELDS}`;
    if (yearFrom) url += `&year=${yearFrom}-`;
    if (fieldsOfStudy) url += `&fieldsOfStudy=${encodeURIComponent(fieldsOfStudy)}`;
    if (continuationToken) url += `&token=${continuationToken}`;

    try {
      console.error(`  [S2 Bulk] Page ${pages + 1}: searching "${query}"...`);
      const json = await fetchWithTimeout(url, 25000);
      const papers = (json.data || []).map(normalizePaper).filter(Boolean);
      results.push(...papers);
      continuationToken = json.token || null;
      if (!continuationToken) break;
      pages++;
      await sleep(2500);
    } catch (error) {
      console.error(`  [S2 Bulk] Error on page ${pages + 1}:`, error.message);
      break;
    }
  }

  console.error(`  [S2 Bulk] Total: ${results.length} papers for "${query}"`);
  return results.slice(0, maxPapers);
}


/**
 * Batch fetch: POST /paper/batch to get metadata for up to 500 papers at once.
 * @param {string[]} paperIds - Semantic Scholar paper IDs, DOIs, or arXiv IDs
 */
export async function batchFetchPapers(paperIds) {
  if (s2Disabled || !paperIds?.length) return [];
  const BATCH_SIZE = 500;
  const results = [];

  for (let i = 0; i < paperIds.length; i += BATCH_SIZE) {
    const batch = paperIds.slice(i, i + BATCH_SIZE);
    try {
      console.error(`  [S2 Batch] Fetching ${batch.length} papers (offset ${i})...`);
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 30000);
      const response = await fetch(`${S2_BASE}/paper/batch?fields=${S2_FIELDS}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "User-Agent": "OpenResearchOS/2.0 (research agent)",
        },
        body: JSON.stringify({ ids: batch }),
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (response.status === 429) {
        s2ConsecutiveFailures++;
        if (s2ConsecutiveFailures >= S2_MAX_CONSECUTIVE_FAILURES) {
          s2Disabled = true;
        }
        console.error(`  [S2 Batch] Rate limited (429). Skipping batch.`);
        await sleep(5000);
        continue;
      }
      if (!response.ok) {
        console.error(`  [S2 Batch] Error: ${response.status}`);
        continue;
      }

      s2ConsecutiveFailures = 0;
      const json = await response.json();
      for (const paper of json) {
        if (paper) results.push(normalizePaper(paper));
      }
      await sleep(2000);
    } catch (error) {
      console.error(`  [S2 Batch] Error:`, error.message);
    }
  }

  return results;
}

/**
 * Fetch references of a paper (papers it cites).
 */
export async function fetchReferences(paperId, { limit = 20 } = {}) {
  if (s2Disabled) return [];
  try {
    const url = `${S2_BASE}/paper/${encodeURIComponent(paperId)}/references?fields=title,year,url,abstract,venue,citationCount,externalIds&limit=${limit}`;
    const json = await fetchWithTimeout(url);
    return (json.data || [])
      .map((item) => item.citedPaper)
      .filter(Boolean)
      .map(normalizePaper);
  } catch (error) {
    console.error(`  [S2] fetchReferences error:`, error.message);
    return [];
  }
}

/**
 * Fetch citations of a paper (papers that cite it).
 */
export async function fetchCitations(paperId, { limit = 20 } = {}) {
  if (s2Disabled) return [];
  try {
    const url = `${S2_BASE}/paper/${encodeURIComponent(paperId)}/citations?fields=title,year,url,abstract,venue,citationCount,externalIds&limit=${limit}`;
    const json = await fetchWithTimeout(url);
    return (json.data || [])
      .map((item) => item.citingPaper)
      .filter(Boolean)
      .map(normalizePaper);
  } catch (error) {
    console.error(`  [S2] fetchCitations error:`, error.message);
    return [];
  }
}

/**
 * Get the open-access PDF URL for a paper (if available).
 */
export async function getOpenAccessPdf(paperId) {
  if (s2Disabled) return null;
  try {
    const url = `${S2_BASE}/paper/${encodeURIComponent(paperId)}?fields=openAccessPdf`;
    const json = await fetchWithTimeout(url);
    return json.openAccessPdf?.url || null;
  } catch {
    return null;
  }
}

