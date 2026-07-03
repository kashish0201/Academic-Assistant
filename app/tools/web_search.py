from duckduckgo_search import DDGS


class WebSearchTool:
    def __init__(self):
        self.name = "web_search"
        self.description = "Search .edu sites for the latest academic information"

    def search(self, query: str, max_results: int = 5) -> str:
        search_query = f"site:.edu {query.strip()}"

        try:
            with DDGS() as ddgs:
                results = ddgs.text(search_query, max_results=max_results)
        except Exception:
            return "No web results found.\n"

        if not results:
            return "No web results found.\n"

        formatted_results = []
        for result in results:
            title = result.get("title", "No title")
            snippet = result.get("body", result.get("snippet", ""))
            url = result.get("href", result.get("url", ""))
            formatted_results.append(
                f"Title: {title}\nSnippet: {snippet}\nURL: {url}\n"
            )

        return "\n".join(formatted_results)
