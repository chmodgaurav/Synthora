import logging

from agents.factCheckAgent import fact_check_agent
from agents.reportAgent import report_agent
from agents.searchAgent import search_agent
from agents.summarizeAgent import summarize_agent
from utils.jsonDB import json_db

logger = logging.getLogger(__name__)


class ResearchController:
    def __init__(self) -> None:
        self.search_agent = search_agent
        self.summarize_agent = summarize_agent
        self.fact_check_agent = fact_check_agent
        self.report_agent = report_agent
        self.db = json_db

    async def process_research_request(self, user_id: str, query: str) -> str:
        project_id = await self.db.create_project(user_id, query)
        logger.info("Created project %s for user %s", project_id, user_id)

        try:
            await self.db.update_project(project_id, {"status": "searching"})
            sources = await self.search_agent.search_web(query)
            logger.info("Search complete: %d sources", len(sources))

            await self.db.update_project(project_id, {"sources": sources, "status": "summarizing"})
            summary = await self.summarize_agent.summarize_content(sources, query)
            logger.info("Summarization complete")

            await self.db.update_project(project_id, {"summary": summary, "status": "fact_checking"})
            fact_checks = await self.fact_check_agent.fact_check_claims(sources, summary)
            logger.info("Fact-checking complete: %d claims", len(fact_checks))

            await self.db.update_project(
                project_id, {"factChecks": fact_checks, "status": "generating_report"}
            )
            final_report = await self.report_agent.generate_report(
                query, sources, summary, fact_checks
            )
            logger.info("Report generation complete")

            await self.db.update_project(
                project_id, {"finalReport": final_report, "status": "completed"}
            )

            report_id = await self.db.save_report(
                {
                    "projectId": project_id,
                    "query": query,
                    "report": final_report,
                    "sources": sources,
                    "summary": summary,
                    "factChecks": fact_checks,
                }
            )
            await self.db.update_project(project_id, {"reportId": report_id})
            return project_id

        except Exception as exc:
            logger.exception("Research pipeline failed for project %s", project_id)
            await self.db.update_project(project_id, {"status": "failed", "error": str(exc)})
            raise


controller = ResearchController()