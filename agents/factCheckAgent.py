import json
from typing import Any, Dict, List

from services.ollamaService import ollama_service

_MAX_CLAIMS = 8
_MAX_SOURCE_CONTENT = 1200

_FACT_CHECK_SCHEMA = {
    "claim": "string (the original claim being verified)",
    "status": "one of: verified|partially_verified|contradicted|unverified",
    "confidence": "float from 0.0 to 1.0",
    "supporting_sources": ["string (URLs of supporting sources)"],
    "explanation": "string (brief explanation of verification)",
}


class FactCheckAgent:
    def __init__(self) -> None:
        self.llm = ollama_service

    async def fact_check_claims(
        self, sources: List[Dict[str, Any]], summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        claims: List[str] = []
        for key in ["keyInsights", "statistics", "arguments", "risks", "opportunities"]:
            claims.extend(summary.get(key, []))

        # Limit to avoid excessive LLM calls
        claims = [c for c in claims if c and len(c) > 10][:_MAX_CLAIMS]

        results = []
        for claim in claims:
            results.append(await self._verify_claim(claim, sources))
        return results

    async def _verify_claim(
        self, claim: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        evidence = [
            {
                "title": s.get("title"),
                "url": s.get("url"),
                "content": s.get("content", "")[:_MAX_SOURCE_CONTENT],
            }
            for s in sources
        ]
        prompt = f"""Verify this claim using ONLY the provided source excerpts. Be objective and evidence-based.

Claim to verify: {claim}

Available sources:
{json.dumps(evidence, indent=2)}

Determine if the claim is verified, partially verified, contradicted, or unverified based on the evidence."""
        
        try:
            data = await self.llm.generate_json(prompt, _FACT_CHECK_SCHEMA, max_tokens=1024)
            if isinstance(data, dict):
                data.setdefault("claim", claim)
                data.setdefault("supporting_sources", [])
                # Coerce status to valid enum value
                valid_statuses = {"verified", "partially_verified", "contradicted", "unverified"}
                if data.get("status") not in valid_statuses:
                    data["status"] = "unverified"
                return data
        except Exception as exc:
            print(f"[FactCheckAgent] Verification failed for claim '{claim[:60]}': {exc}")

        return {
            "claim": claim,
            "status": "unverified",
            "confidence": 0.1,
            "supporting_sources": [],
            "explanation": "Model verification failed or source evidence was insufficient.",
        }


fact_check_agent = FactCheckAgent()