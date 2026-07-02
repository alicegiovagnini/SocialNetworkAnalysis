"""
Metadata collection (node attributes)

  - Profiles (always): handle, displayName, bio, followers/follows/posts
    counts, account-creation date. Needed for community detection
    (semantic interpretation) and for the feature-rich analysis
  - Posts (optional, --posts flag): a sample of posts per node with text
    and timestamp. Enables the textual layer (NLP, open question) and the
    temporal one (stream graph, dated reply/repost interactions)

Usage:
    python fetch_metadata.py                 # profiles only (fast)
    python fetch_metadata.py --posts         # profiles + 20 posts/node
    python fetch_metadata.py --posts --max-posts 10
"""

import os
import json
import argparse

from api import RateLimitedSession, get_profiles, get_author_feed
import config


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def fetch_profiles(session, dids):
    out = {}
    total = len(dids)
    for k, batch in enumerate(chunked(dids, 25)):  # getProfiles: max 25 per call
        data = get_profiles(session, batch)
        if data:
            for p in data.get("profiles", []):
                out[p["did"]] = {
                    "handle": p.get("handle", ""),
                    "displayName": p.get("displayName", ""),
                    "description": p.get("description", ""),
                    "followersCount": p.get("followersCount", 0),
                    "followsCount": p.get("followsCount", 0),
                    "postsCount": p.get("postsCount", 0),
                    "createdAt": p.get("createdAt", ""),
                    "indexedAt": p.get("indexedAt", ""),
                }
        if (k + 1) % 20 == 0:
            print(f"[profiles] {min((k + 1) * 25, total)}/{total}")
    return out


def fetch_posts(session, dids, max_posts=20):
    records = []
    total = len(dids)
    for i, did in enumerate(dids):
        data = get_author_feed(session, did, limit=min(max_posts, 100))
        if data:
            for item in data.get("feed", [])[:max_posts]:
                post = item.get("post", {}) or {}
                rec = post.get("record", {}) or {}
                reply = (rec.get("reply") or {}).get("parent") or {}
                records.append({
                    "author": did,
                    "uri": post.get("uri", ""),
                    "createdAt": rec.get("createdAt", ""),
                    "text": rec.get("text", ""),
                    "replyParentUri": reply.get("uri", ""),
                    "likeCount": post.get("likeCount", 0),
                    "repostCount": post.get("repostCount", 0),
                })
        if (i + 1) % 500 == 0:
            print(f"[posts] {i + 1}/{total} nodes")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", action="store_true",
                    help="also download a sample of posts per node")
    ap.add_argument("--max-posts", type=int, default=20)
    args = ap.parse_args()

    session = RateLimitedSession(min_interval=config.REQUEST_MIN_INTERVAL)
    with open(os.path.join(config.DATA_DIR, "nodes.json")) as f:
        nodes = json.load(f)
    dids = list(nodes.keys())

    print(f"[meta] profiles for {len(dids)} nodes...")
    profiles = fetch_profiles(session, dids)
    with open(os.path.join(config.DATA_DIR, "node_attributes.json"), "w") as f:
        json.dump(profiles, f, ensure_ascii=False)
    print(f"[meta] profiles saved: {len(profiles)}")

    if args.posts:
        print(f"[meta] post sample (max {args.max_posts}/node): takes longer...")
        recs = fetch_posts(session, dids, max_posts=args.max_posts)
        path = os.path.join(config.DATA_DIR, "posts_sample.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[meta] post records saved: {len(recs)} -> {path}")


if __name__ == "__main__":
    main()
