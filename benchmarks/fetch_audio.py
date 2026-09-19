"""Fetch small reproducible regression inputs; not a comprehensive quality corpus."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    manifest=Path("data/manifest.json")
    for record in json.loads(manifest.read_text()):
        path=Path(record["path"])
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            with urlopen(record["source"],timeout=60) as response:
                data=response.read()
            if hashlib.sha256(data).hexdigest()!=record["sha256"]:
                raise ValueError(f"Downloaded bytes changed: {path}")
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record["sha256"]:
            raise ValueError(f"Asset checksum mismatch: {path}")
        print(path)


if __name__=="__main__":
    main()
