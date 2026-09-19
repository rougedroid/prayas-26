from dataclasses import dataclass
import os

@dataclass(slots=True)
class VCConfig:
    base_url: str
    project_id: str
    username: str | None = None
    password: str | None = None
    token: str | None = None
    data_bucket_id: str | None = None
    verify_tls: bool = True
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls):
        base_url = os.getenv("VC_PUBLISHER_URL", "").rstrip("/")
        project_id = os.getenv("VC_PROJECT_ID", "")
        if not base_url:
            raise ValueError("VC_PUBLISHER_URL is required")
        if not project_id:
            raise ValueError("VC_PROJECT_ID is required")
        return cls(
            base_url=base_url,
            project_id=project_id,
            username=os.getenv("VC_USERNAME") or None,
            password=os.getenv("VC_PASSWORD") or None,
            token=os.getenv("VC_TOKEN") or None,
            data_bucket_id=os.getenv("VC_DATA_BUCKET_ID") or None,
            verify_tls=os.getenv("VC_VERIFY_TLS", "true").lower() not in {"0","false","no"},
            timeout_seconds=float(os.getenv("VC_TIMEOUT_SECONDS", "30")),
        )
