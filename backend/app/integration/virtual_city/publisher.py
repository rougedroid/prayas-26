from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .config import VCConfig

class VCPublisherError(RuntimeError):
    pass

class VCPublisherClient:
    def __init__(self, config: VCConfig):
        self.config = config
        self.session = requests.Session()
        retry = Retry(
            total=3, connect=3, read=3, backoff_factor=0.5,
            status_forcelist=(429,500,502,503,504),
            allowed_methods=frozenset({"GET","POST","PUT","PATCH","DELETE"})
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        if config.token:
            self.session.headers["Authorization"] = f"Bearer {config.token}"

    def _url(self, path): return f"{self.config.base_url}{path}"

    def _check(self, response, expected):
        if response.status_code not in expected:
            raise VCPublisherError(
                f"VC Publisher HTTP {response.status_code}: {response.text[:1000]}"
            )

    def login(self):
        if self.config.token:
            return self.config.token
        if not self.config.username or not self.config.password:
            raise VCPublisherError("Set VC_TOKEN or VC_USERNAME + VC_PASSWORD")
        r = self.session.post(
            self._url("/api/v1/login"),
            json={"username":self.config.username,"password":self.config.password},
            timeout=self.config.timeout_seconds,
            verify=self.config.verify_tls,
        )
        self._check(r,{200})
        token = r.json().get("token")
        if not token:
            raise VCPublisherError("Login response contained no token")
        self.session.headers["Authorization"] = f"Bearer {token}"
        return token

    def list_data_buckets(self):
        self.login()
        r = self.session.get(
            self._url(f"/api/v1/project/{self.config.project_id}/data-buckets"),
            timeout=self.config.timeout_seconds,
            verify=self.config.verify_tls,
        )
        self._check(r,{200})
        body = r.json()
        return body.get("items", body if isinstance(body,list) else [])

    def resolve_data_bucket_id(self):
        if self.config.data_bucket_id:
            return self.config.data_bucket_id
        buckets = self.list_data_buckets()
        if not buckets:
            raise VCPublisherError("No project data bucket found")
        return buckets[0]["_id"]

    def upload_file(self, file_path, overwrite=True):
        self.login()
        path = Path(file_path)
        bucket_id = self.resolve_data_bucket_id()
        endpoint = (
            f"/api/v1/project/{self.config.project_id}/data-bucket/"
            f"{bucket_id}/upload"
        )
        with path.open("rb") as fh:
            r = self.session.post(
                self._url(endpoint),
                params={"overwrite":"true"} if overwrite else {},
                files={"file":(path.name,fh,"application/geo+json")},
                timeout=self.config.timeout_seconds,
                verify=self.config.verify_tls,
            )
        self._check(r,{200,201,204,207})
        return path.name

    def create_internal_datasource(self, name, data_bucket_key, datasource_type="geojson"):
        self.login()
        bucket_id = self.resolve_data_bucket_id()
        payload = {
            "name":name,
            "type":datasource_type,
            "typeProperties":{},
            "sourceProperties":{
                "type":"internal",
                "dataBucketId":bucket_id,
                "dataBucketKey":data_bucket_key,
            },
        }
        r = self.session.post(
            self._url(f"/api/v1/project/{self.config.project_id}/datasource"),
            json=payload,
            timeout=self.config.timeout_seconds,
            verify=self.config.verify_tls,
        )
        self._check(r,{201})
        return r.json()

    def publish_geojson(self, file_path, datasource_name="Prayas Optimized City Plan"):
        key = self.upload_file(file_path)
        return self.create_internal_datasource(datasource_name, key, "geojson")
