"""Seed site/data from the currently deployed site.

Frames are deliberately not stored in git — see the note in
.github/workflows/main.yml — so every CI run starts with an empty site/data.
The 10-frame rolling animation buffer would restart from a single frame each
run unless the previous frames are brought back first.

The Actions cache normally supplies them.  This script is the fallback for a
cold start or an evicted cache: it reads data/frames.json from the live site
and downloads whatever it lists.  Everything here is best-effort — a failure
just means the buffer refills over the next few runs, so nothing is allowed to
fail the job.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OUTPUT_DIR = 'site/data'
TIMEOUT = 20


def _get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'goes19-restore'})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read()


def _fetch_one(base_url, name):
    """Download one frame, skipping anything already present locally."""
    dest = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(dest):
        return False  # cache already provided a newer copy; never overwrite it
    try:
        data = _get(f'{base_url}/data/{name}')
    except Exception:
        return False
    # Write via a temp name so an interrupted download can't leave a truncated
    # PNG that the next run would happily shift into the buffer.
    tmp = dest + '.part'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, dest)
    return True


def main():
    base_url = os.environ.get('SITE_URL', '').rstrip('/')
    if not base_url:
        print("SITE_URL not set; skipping frame restore.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        manifest = json.loads(_get(f'{base_url}/data/frames.json'))
    except urllib.error.HTTPError as e:
        print(f"No manifest at {base_url}/data/frames.json (HTTP {e.code}) — "
              f"cold start, the buffer will fill over the next few runs.")
        return
    except Exception as e:
        print(f"Could not read manifest from {base_url}: {e}")
        return

    names = [n for n in manifest.get('files', [])
             if isinstance(n, str) and '/' not in n and not n.startswith('.')]
    if not names:
        print("Manifest is empty; nothing to restore.")
        return

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda n: _fetch_one(base_url, n), names))

    print(f"Restored {sum(results)} of {len(names)} file(s) from {base_url}.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # never fail the job over a best-effort restore
        print(f"Frame restore failed ({e}); continuing with an empty buffer.")
        sys.exit(0)
