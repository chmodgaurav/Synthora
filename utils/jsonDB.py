import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class JsonDB:
    def __init__(self, base_dir: str = "data") -> None:
        self.base = Path(base_dir)
        for folder in ["users", "projects", "reports", "cache"]:
            (self.base / folder).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    async def create_project(self, user_id: str, query: str) -> str:
        project_id = f"project_{uuid.uuid4().hex[:12]}"
        project = {
            "_id": project_id,
            "userId": user_id,
            "query": query,
            "status": "created",
            "_created_at": self._now(),
            "_updated_at": self._now(),
        }
        await self._write_project(project_id, project)
        return project_id

    async def update_project(self, project_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        path = self._project_path(project_id)
        data: Dict[str, Any] = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        data.update(updates)
        data.setdefault("_id", project_id)
        data["_updated_at"] = self._now()
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return data

    async def get_project(self, project_id: str) -> Dict[str, Any]:
        path = self._project_path(project_id)
        if not path.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    async def list_user_projects(self, user_id: str) -> List[Dict[str, Any]]:
        projects = []
        for path in (self.base / "projects").glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("userId") == user_id:
                    projects.append(data)
            except (json.JSONDecodeError, OSError):
                continue
        return sorted(projects, key=lambda x: x.get("_updated_at", ""), reverse=True)

    async def list_all_projects(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recently updated projects across all users."""
        projects = []
        for path in (self.base / "projects").glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                projects.append(data)
            except (json.JSONDecodeError, OSError):
                continue
        return sorted(projects, key=lambda x: x.get("_updated_at", ""), reverse=True)[:limit]

    async def save_report(self, report_data: Dict[str, Any]) -> str:
        report_id = f"report_{uuid.uuid4().hex[:12]}"
        report_data.update(
            {"_id": report_id, "_created_at": self._now(), "_updated_at": self._now()}
        )
        path = self.base / "reports" / f"{report_id}.json"
        path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
        return report_id

    async def get_report(self, report_id: str) -> Dict[str, Any]:
        path = self.base / "reports" / f"{report_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Report not found: {report_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _project_path(self, project_id: str) -> Path:
        return self.base / "projects" / f"{project_id}.json"

    async def _write_project(self, project_id: str, data: Dict[str, Any]) -> None:
        self._project_path(project_id).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


json_db = JsonDB()