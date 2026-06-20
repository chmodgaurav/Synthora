import json
from typing import Any, Dict, List

from services.ollamaService import ollama_service


class ReportAgent:
    def __init__(self) -> None:
        self.llm = ollama_service

    async def generate_report(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        summary: Dict[str, Any],
        fact_checks: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a comprehensive markdown research report.
        Uses local Ollama to create professional, well-structured output.
        """
        source_list = [
            {
                "title": s.get("title"),
                "url": s.get("url"),
                "credibility": s.get("credibility"),
                "relevance_score": s.get("relevance_score"),
            }
            for s in sources
        ]
        
        prompt = f"""You are a professional research report writer. Create a polished Markdown research report.
Be objective. Do not invent facts. Use only the provided data.

Research query: {query}

Summary findings: {json.dumps(summary, indent=2)}

Fact-check results: {json.dumps(fact_checks, indent=2)}

Sources used: {json.dumps(source_list, indent=2)}

Create a structured markdown report with these sections:
# Research Report: {query}
## Executive Summary
## Key Findings
## Detailed Analysis
## Risks and Opportunities
## Fact-Checking Results
## Sources and References

Be concise and professional. Cite sources inline by title and URL."""
        
        try:
            return await self.llm.generate_content(prompt, max_tokens=4000)
        except Exception as exc:
            print(f"[ReportAgent] Report generation failed: {exc}")
            source_lines = "\n".join(
                f"- [{s.get('title', 'Unknown')}]({s.get('url', '')})" for s in sources
            )
            return f"""# Research Report: {query}

## Status
Report generation failed — verify Ollama is running.

## Error
```
{exc}
```

## Sources Collected
{source_lines}
"""


report_agent = ReportAgent()