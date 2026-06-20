import json
from typing import Any, Dict, List

from services.ollamaService import ollama_service

_REQUIRED_KEYS = ["keyInsights", "statistics", "arguments", "risks", "opportunities"]

_SUMMARY_SCHEMA = {
    "keyInsights": ["string (3-5 main insights)"],
    "statistics": ["string (quantified facts, numbers, percentages)"],
    "arguments": ["string (main arguments or claims)"],
    "risks": ["string (identified risks or concerns)"],
    "opportunities": ["string (opportunities or positive possibilities)"],
}


class SummarizationAgent:
    def __init__(self) -> None:
        self.llm = ollama_service

    async def summarize_content(
        self, sources: List[Dict[str, Any]], query: str
    ) -> Dict[str, Any]:
        combined_content = "\n\n".join(
            f"Source: {s.get('title', 'Unknown')}\nURL: {s.get('url', '')}\n{s.get('content', '')[:2500]}"
            for s in sources
        )

        prompt = f"""You are a professional research summarizer. Analyze these sources for the query: "{query}"

Sources to analyze:
{combined_content[:15000]}

Extract and organize the key information into structured categories."""
        
        try:
            data = await self.llm.generate_json(prompt, _SUMMARY_SCHEMA, max_tokens=2048)
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object, got {type(data)}")
            return {key: data.get(key, []) for key in _REQUIRED_KEYS}
        except Exception as exc:
            print(f"[SummarizationAgent] Failed: {exc}")
            return {
                "keyInsights": ["Summarization failed — check Ollama is running."],
                "statistics": [],
                "arguments": [],
                "risks": [str(exc)],
                "opportunities": ["Verify OLLAMA_MODEL in .env is available locally."],
            }


summarize_agent = SummarizationAgent()