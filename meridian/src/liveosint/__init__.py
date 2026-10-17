"""Live OSINT: responsible collection, evidence extraction, two separate assessments and analyst-gated Twin updates."""
from .extract import CLASSES, Candidate, extract, dedupe, corroborate, vendor_domain
from .fetcher import FetchResult, FixtureFetcher, SafeFetcher
from .pipeline import DEFAULT_CATEGORIES, RunReport, run_collection
from .review import ReviewError, review_evidence, sync_vendor
from .scoring import ADOPTION_WEIGHTS, adoption_score, exposure_assessment, score_pair
from .search import SearchUnavailable, provider_from_env, configured_provider_name
from .store import Store
from .urlguard import UrlBlocked, check_url
