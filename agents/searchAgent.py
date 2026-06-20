"""
SearchAgent — Local knowledge base search + LLM-based ranking.

Searches documents stored in the local knowledge base.
No external APIs or web scraping required.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from services.ollamaService import ollama_service


KNOWLEDGE_DIR = Path("data/knowledge")


def _ensure_knowledge_dir() -> None:
    """Create knowledge base directory if needed."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)


def _search_knowledge_base_local(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Simple keyword-based search in local knowledge base."""
    _ensure_knowledge_dir()
    results: List[Dict[str, Any]] = []
    query_terms = query.lower().split()
    
    for doc_path in KNOWLEDGE_DIR.glob("*.json"):
        try:
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            text = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
            
            # Simple relevance scoring: term frequency
            score = sum(text.count(term) for term in query_terms)
            
            if score > 0:
                results.append({
                    "title": doc.get("title", "Untitled"),
                    "content": doc.get("content", ""),
                    "url": f"local://kb/{doc['_id']}",
                    "relevance_score": score / max(len(query_terms), 1),
                    "credibility": 0.85,
                })
        except (json.JSONDecodeError, OSError):
            continue
    
    if not results:
        return [{
            "title": "No local knowledge base available",
            "url": "local://fallback",
            "content": f"No documents found matching: {query}. Upload documents to the knowledge base.",
            "credibility": 0.2,
            "relevance_score": 0.1,
        }]
    
    return sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:top_k]


class SearchAgent:
    def __init__(self) -> None:
        self.llm = ollama_service

    async def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search local knowledge base and rank by LLM relevance scoring."""
        results = await asyncio.to_thread(_search_knowledge_base_local, query, max_results)
        return await self._rank_sources(list(results), query)

    async def _rank_sources(
        self, sources: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Use local LLM to score each source for relevance to the query."""
        if not sources or all(s.get("url") == "local://fallback" for s in sources):
            return sources
            
        trimmed = [
            {k: (v[:800] if k == "content" else v) for k, v in s.items()}
            for s in sources
        ]
        
        prompt = f"""You are a research relevance expert. Score these sources for relevance to this query.

Query: {query}

Sources to rank:
{json.dumps(trimmed, indent=2)}

For each source, provide a relevance_score between 0.0 and 1.0."""
        
        try:
            response = await self.llm.generate_content(prompt, max_tokens=1024)
            data = self.llm.extract_json(response)
            
            if isinstance(data, list) and len(data) == len(sources):
                score_map = {r.get("url"): r.get("relevance_score", 0.5) for r in data}
                for s in sources:
                    s["relevance_score"] = score_map.get(s["url"], s.get("relevance_score", 0.5))
                return sorted(sources, key=lambda x: x["relevance_score"], reverse=True)
        except Exception as exc:
            print(f"[SearchAgent] LLM ranking failed: {exc}")

        # Fallback: use keyword-based scores
        for s in sources:
            s.setdefault("relevance_score", 0.5)
        return sorted(sources, key=lambda x: x["relevance_score"], reverse=True)


search_agent = SearchAgent()