"""
It automatically handles:
  - pacing between requests (to stay under the rate limit ~3000 req/5min);
  - retry with exponential backoff on 429 / 5xx / network errors.
"""

import time  #for pauses 
import requests #HTTP library

#PUBLIC_BASE is the base URL of public endpoints. xrpc is the AT protocol calling system.
PUBLIC_BASE = "https://public.api.bsky.app/xrpc"


class RateLimitedSession:
    def __init__(self, min_interval=0.2, max_retries=6):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "sna-unipi-final-project-crawler/1.0 (academic use)"}
        )
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last = 0.0

    def get(self, endpoint, params=None):
        url = f"{PUBLIC_BASE}/{endpoint}"
        for attempt in range(self.max_retries):
            # pacing: respect the minimum interval between requests
            dt = time.time() - self._last
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)
            self._last = time.time()

            try:
                r = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                wait = 2 ** attempt
                print(f"[warn] network error: {e}; retrying in {wait}s")
                time.sleep(wait)
                continue

            if r.status_code == 429:  # Too Many Requests
                retry_after = int(r.headers.get("Retry-After", 2 ** attempt))
                print(f"[ratelimit] 429; waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            if r.status_code == 400:
                # typically a deactivated / not-found account: skip it
                return None

            if r.status_code >= 500:
                wait = 2 ** attempt
                print(f"[warn] server {r.status_code}; retrying in {wait}s")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json()

        print(f"[error] request failed after {self.max_retries} attempts: {endpoint}")
        return None


#helpers for the endpoints used

def resolve_handle(session, handle):
    """handle (e.g. 'sara_alice_maria_paula.bsky.social') -> persistent DID"""
    data = session.get("com.atproto.identity.resolveHandle", {"handle": handle})
    return data["did"] if data and "did" in data else None


def get_follows(session, actor, cursor=None, limit=100):
    """Accounts followed by `actor` (one page, max 100)"""
    params = {"actor": actor, "limit": limit}
    if cursor:
        params["cursor"] = cursor
    return session.get("app.bsky.graph.getFollows", params)


def get_profiles(session, actors):
    """Detailed profiles for a list of at most 25 actors (DID or handle)"""
    return session.get("app.bsky.actor.getProfiles", {"actors": actors})


def get_author_feed(session, actor, cursor=None, limit=100):
    """Post feed of an account (for the textual/temporal layer)"""
    params = {"actor": actor, "limit": limit, "filter": "posts_with_replies"}
    if cursor:
        params["cursor"] = cursor
    return session.get("app.bsky.feed.getAuthorFeed", params)


def search_actors(session, q, limit=25):
    """Search accounts by keyword (useful to pick the seeds)"""
    return session.get("app.bsky.actor.searchActors", {"q": q, "limit": limit})
