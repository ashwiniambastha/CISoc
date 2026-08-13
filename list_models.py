"""
Ask the catalog what model IDs actually exist.

A 404 from chat.completions means the model string is wrong. The authoritative
answer is the endpoint's own list, not the docs and not memory.

    python list_models.py            # nemotron-ish IDs
    python list_models.py --all      # everything the key can see
    python list_models.py gpt        # substring filter
"""

import sys

from openai import OpenAI

import config

client = OpenAI(base_url=config.BASE_URL, api_key=config.get_api_key())

args = [a for a in sys.argv[1:]]
show_all = "--all" in args
needle = next((a.lower() for a in args if not a.startswith("--")), "nemotron")

ids = sorted(m.id for m in client.models.list().data)
print(f"{len(ids)} models visible to this key\n")

hits = ids if show_all else [i for i in ids if needle in i.lower()]
if not hits:
    print(f"nothing matching {needle!r}. Re-run with --all to see the full list.")
else:
    for i in hits:
        print(" ", i)

print("\nconfigured in config.py:")
for name, cfg in config.MODELS.items():
    mark = "OK " if cfg["model_id"] in ids else "404"
    print(f"  [{mark}] {name:<10} {cfg['model_id']}")
