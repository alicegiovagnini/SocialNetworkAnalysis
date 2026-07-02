# Configuration parameters for the Bluesky crawler

import os

# Starting accounts (Bluesky handles, WITHOUT the leading @)
# Chosen topic: science outreach / scientific community on Bluesky
# A mix of outlets/organisations (which follow many science accounts) and very active communicators: excellent "hubs" for the snowball.
SEED_HANDLES = [
    "science.org",            # Science Magazine
    "scifri.bsky.social",     # Science Friday
    "standupforscience.net",  # Stand Up for Science
    "whysharksmatter.bsky.social",  # David Shiffman (very active sci-communicator)
    "carlbergstrom.com",      # Carl T. Bergstrom (prolific academic)
]

# Maximum number of nodes to collect
MAX_NODES = 15000

# Cap on the number of follow pages downloaded per node (100 follows/page)
# Keeps the cost bounded on mega-hubs (e.g. accounts with 50k+ follows)
# 10 pages = at most 1000 follows per node
MAX_FOLLOWS_PAGES_PER_NODE = 10

# Minimum pause between two requests (seconds)
# 0.1 -> about 5 requests/second, well below the limit (~10/s)
REQUEST_MIN_INTERVAL = 0.1

# How often (in processed nodes) to save the state (to resume after a crash)
CHECKPOINT_EVERY = 200

# Output data directory
# Anchored to this file's folder (data_collection/) so every script finds the
# data regardless of the working directory it is launched from.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
