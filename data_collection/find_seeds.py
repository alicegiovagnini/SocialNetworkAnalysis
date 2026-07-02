"""
Utility to find the "seed" accounts the crawl starts from.

Usage:
    python find_seeds.py science

Prints handle, name and DID of the first 25 results. Pick 1-5 central
accounts coherent with the topic and copy them into config.SEED_HANDLES.
"""

import sys
from api import RateLimitedSession, search_actors


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_seeds.py <keywords>")
        return
    q = " ".join(sys.argv[1:])
    session = RateLimitedSession()
    data = search_actors(session, q, limit=25)
    if not data or not data.get("actors"):
        print("No results.")
        return
    print(f"{'handle':38s} {'displayName':28s} followers")
    for a in data["actors"]:
        name = (a.get("displayName") or "")[:26]
        fol = a.get("followersCount", "")
        print(f"{a.get('handle',''):38s} {name:28s} {fol}")


if __name__ == "__main__":
    main()
