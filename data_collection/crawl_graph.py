"""
Building the follow graph via snowball / BFS.

Strategy:
  - start from the SEEDS (config.SEED_HANDLES), resolved to DIDs;
  - breadth-first search (BFS): for each node we download its "follows";
  - newly discovered accounts are added until MAX_NODES is reached; after
    the cap no new nodes are added, but the queue keeps being processed to
    record ALL edges internal to the set;
  - every observed edge (u -> v) is written to disk (raw_edges.jsonl);
    the filtering to the induced subgraph is done in build_network.py.
    This way no edge internal to the set is lost (exhaustive approach).

It is resumable: the state (nodes, queue, processed) is saved at every
checkpoint; if interrupted, just re-run the same command.
"""

import os
import json
import pickle
import collections

from api import RateLimitedSession, resolve_handle, get_follows
import config


def load_state(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_state(path, nodes, queue, processed):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(
            {"nodes": nodes, "queue": list(queue), "processed": processed}, f
        )
    os.replace(tmp, path)  # atomic write


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    state_path = os.path.join(config.DATA_DIR, "crawl_state.pkl")
    edges_path = os.path.join(config.DATA_DIR, "raw_edges.jsonl")
    session = RateLimitedSession(min_interval=config.REQUEST_MIN_INTERVAL)

    saved = load_state(state_path)
    if saved is None:
        nodes = {}                      # did -> handle
        queue = collections.deque()
        processed = set()
        for h in config.SEED_HANDLES:
            did = resolve_handle(session, h)
            if did and did not in nodes:
                nodes[did] = h
                queue.append(did)
                print(f"[seed] {h} -> {did}")
        if not nodes:
            print("[error] no seed resolved. Check config.SEED_HANDLES.")
            return
    else:
        nodes = saved["nodes"]
        queue = collections.deque(saved["queue"])
        processed = saved["processed"]
        print(f"[resume] nodes={len(nodes)} processed={len(processed)} "
              f"queue={len(queue)}")

    # open in append mode: on resume, already-written edges are harmless
    # (build_network.py de-duplicates with a set).
    edges_file = open(edges_path, "a")
    count = 0

    while queue:
        u = queue.popleft()
        if u in processed:
            continue

        cursor = None
        pages = 0
        while True:
            data = get_follows(session, u, cursor=cursor, limit=100)
            if not data:
                break
            for f in data.get("follows", []):
                v = f["did"]
                if v not in nodes and len(nodes) < config.MAX_NODES:
                    nodes[v] = f.get("handle", "")
                    queue.append(v)
                # always record the observed edge (filtering to the induced
                # subgraph happens at build time): guarantees completeness.
                edges_file.write(json.dumps([u, v]) + "\n")
            cursor = data.get("cursor")
            pages += 1
            if not cursor or pages >= config.MAX_FOLLOWS_PAGES_PER_NODE:
                break

        processed.add(u)
        count += 1
        if count % config.CHECKPOINT_EVERY == 0:
            edges_file.flush()
            save_state(state_path, nodes, queue, processed)
            print(f"[ckpt] nodes={len(nodes)} processed={len(processed)} "
                  f"queue={len(queue)}")

    edges_file.flush()
    edges_file.close()
    save_state(state_path, nodes, queue, processed)

    with open(os.path.join(config.DATA_DIR, "nodes.json"), "w") as f:
        json.dump(nodes, f)

    print(f"[done] nodes discovered={len(nodes)} processed={len(processed)}")
    print(f"       raw (directed) edges -> {edges_path}")
    print("       next step: python fetch_metadata.py  then  python build_network.py")


if __name__ == "__main__":
    main()
