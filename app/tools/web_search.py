import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import httpx
from ddgs import DDGS

SEARCH_TIMEOUT_SECONDS = 18
DEFAULT_MAX_RESULTS = 5

RANKING_QUERY_PATTERN = re.compile(
    r"\b(rankings?|ranked|rank\b|best\b.{0,30}\b(universit|college|school)|"
    r"top\b.{0,30}\b(universit|college|school)|compare\b.{0,20}\b(universit|college)|"
    r"curriculum\b.{0,40}\bresearch\b)\b",
    re.IGNORECASE,
)

TRUSTED_WEB_DOMAINS = (
    ".edu",
    "usnews.com",
    "niche.com",
    "timeshighereducation.com",
    "forbes.com",
    "washingtonmonthly.com",
    "collegefactual.com",
    "princetonreview.com",
)

CAMPUS_DOMAINS = {
    "csulb": "csulb.edu",
    "cal state long beach": "csulb.edu",
    "long beach state": "csulb.edu",
    "csuf": "fullerton.edu",
    "cal state fullerton": "fullerton.edu",
    "csun": "csun.edu",
    "cal state northridge": "csun.edu",
    "sdsu": "sdsu.edu",
    "san diego state": "sdsu.edu",
    "sfsu": "sfsu.edu",
    "san francisco state": "sfsu.edu",
    "sjsu": "sjsu.edu",
    "san jose state": "sjsu.edu",
    "csusb": "csusb.edu",
    "cal poly pomona": "cpp.edu",
    "cpp": "cpp.edu",
    "cal poly slo": "calpoly.edu",
    "uci": "uci.edu",
    "uc irvine": "uci.edu",
    "ucla": "ucla.edu",
    "ucsd": "ucsd.edu",
    "uc berkeley": "berkeley.edu",
    "ucb": "berkeley.edu",
    "uc davis": "ucdavis.edu",
    "ucsb": "ucsb.edu",
    "ucsc": "ucsc.edu",
}

CAMPUS_FALLBACK_PAGES: dict[str, list[tuple[str, str]]] = {
    "csulb.edu": [
        ("CSULB Student Internships", "https://www.csulb.edu/career-development-center/students/internships"),
        ("CSULB Academic Internships", "https://www.csulb.edu/center-for-community-engagement/academic-internships"),
        ("CSULB Internship Policy", "https://www.csulb.edu/academic-senate/policy-academic-internships-non-clinicalnon-licensure"),
    ],
}


class WebSearchTool:
    def __init__(self):
        self.name = "web_search"
        self.description = "Search .edu sites for the latest academic information"

    def _is_ranking_query(self, query: str) -> bool:
        return bool(RANKING_QUERY_PATTERN.search(query))

    def _detect_campus_domain(self, query: str) -> str | None:
        query_lower = query.lower()
        for campus, domain in sorted(CAMPUS_DOMAINS.items(), key=lambda item: -len(item[0])):
            if campus in query_lower:
                return domain
        return None

    def _extract_keywords(self, query: str) -> str:
        cleaned = re.sub(r"\s+", " ", query.strip())
        cleaned = re.sub(r"[?!.]+$", "", cleaned)
        cleaned = re.sub(
            r"^(what|how|when|where|why|who|can|do|does|is|are)\s+"
            r"(is|are|do|does|can|could|would|the|a|an)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip() or query.strip()

    def _build_queries(self, query: str) -> list[str]:
        keywords = self._extract_keywords(query)
        campus_domain = self._detect_campus_domain(keywords)
        queries = []

        if self._is_ranking_query(query):
            queries.extend([
                f"best California universities research curriculum rankings 2026",
                f"US News university rankings research {keywords}",
                f"{keywords} university rankings",
            ])
            return queries

        if campus_domain:
            short_keywords = keywords
            for campus in CAMPUS_DOMAINS:
                short_keywords = re.sub(re.escape(campus), "", short_keywords, flags=re.IGNORECASE)
            short_keywords = re.sub(r"\s+", " ", short_keywords).strip()

            if short_keywords:
                queries.append(f"{short_keywords} site:{campus_domain}")
            queries.append(f"{keywords} site:{campus_domain}")

        queries.append(f"{keywords} site:edu")

        seen = set()
        unique_queries = []
        for item in queries:
            if item not in seen:
                seen.add(item)
                unique_queries.append(item)
        return unique_queries

    def _is_allowed_result(self, result: dict, campus_domain: str | None, broad: bool) -> bool:
        url = (result.get("href") or result.get("url") or "").lower()
        if campus_domain and campus_domain in url:
            return True
        if broad:
            return any(domain in url for domain in TRUSTED_WEB_DOMAINS)
        return ".edu" in url

    def _run_search(self, search_query: str, max_results: int) -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(search_query, max_results=max_results, backend="auto"))

    def _search_once(self, search_query: str, max_results: int, timeout: float) -> list[dict]:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_search, search_query, max_results)
            return future.result(timeout=timeout)

    def _format_results(self, results: list[dict]) -> str:
        formatted_results = []
        for result in results:
            title = result.get("title", "No title")
            snippet = result.get("body", result.get("snippet", ""))
            url = result.get("href", result.get("url", ""))
            formatted_results.append(
                f"Title: {title}\nSnippet: {snippet}\nURL: {url}\n"
            )
        return "\n".join(formatted_results)

    def _strip_html(self, html: str) -> str:
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
        text = re.sub(r"(?s)<.*?>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _fetch_page_snippet(self, title: str, url: str) -> str:
        try:
            response = httpx.get(
                url,
                timeout=8.0,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; AcademicAssistant/1.0)"},
            )
            response.raise_for_status()
            text = self._strip_html(response.text)
            return f"Title: {title}\nSnippet: {text[:1200]}\nURL: {url}\n"
        except Exception:
            return f"Title: {title}\nSnippet: Official campus page.\nURL: {url}\n"

    def _campus_fallback(self, campus_domain: str, query: str) -> str:
        pages = CAMPUS_FALLBACK_PAGES.get(campus_domain, [])
        if not pages:
            return ""

        keywords = self._extract_keywords(query).lower()
        selected = pages
        if "internship" in keywords:
            selected = [page for page in pages if "internship" in page[0].lower()] or pages

        snippets = []
        for title, url in selected[:3]:
            snippets.append(self._fetch_page_snippet(title, url))
        return "\n".join(snippets)

    def search(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout: float = SEARCH_TIMEOUT_SECONDS,
        broad: bool | None = None,
    ) -> str:
        broad = self._is_ranking_query(query) if broad is None else broad
        campus_domain = self._detect_campus_domain(query)
        search_queries = self._build_queries(query)

        collected: list[dict] = []
        seen_urls: set[str] = set()

        for search_query in search_queries:
            if len(collected) >= max_results:
                break
            try:
                results = self._search_once(search_query, max_results, timeout)
            except FuturesTimeoutError:
                continue
            except Exception:
                continue

            for result in results:
                if not self._is_allowed_result(result, campus_domain, broad):
                    continue
                url = result.get("href") or result.get("url") or ""
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                collected.append(result)
                if len(collected) >= max_results:
                    break

        if collected:
            return self._format_results(collected[:max_results])

        if campus_domain:
            fallback = self._campus_fallback(campus_domain, query)
            if fallback:
                return fallback

        return "No web results found.\n"
