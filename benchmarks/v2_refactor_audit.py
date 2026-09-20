"""Verify that module separation and formatting preserve all measured computation."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

REPORT = Path("results/v2_refactor_equivalence.json")
MOVED = {
    "validate": "fast_moss/v2_runtime.py",
    "V2Runtime": "fast_moss/v2_runtime.py",
    "optimized": "fast_moss/v2_runtime.py",
    "encode_fixed": "fast_moss/v2_codec.py",
    "codec": "fast_moss/v2_codec.py",
}


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def verify(report):
    original = report["original_runtime_sha256"]
    snapshots = report["original_changed_sources"]
    current = {
        str(p): sha(p.read_text())
        for p in Path("fast_moss").glob("*")
        if p.suffix in (".py", ".json")
    }
    assert set(current) - set(original) == {
        "fast_moss/v2_runtime.py",
        "fast_moss/v2_codec.py",
    }
    assert not set(original) - set(current)
    checks = []
    for path, digest in original.items():
        if path in snapshots:
            assert sha(snapshots[path]) == digest
        else:
            assert current[path] == digest, path
            checks.append(path + " unchanged bytes")
    for path in ["fast_moss/v2_loading.py", "fast_moss/v2_pointwise.py"]:
        assert ast.dump(ast.parse(snapshots[path])) == ast.dump(
            ast.parse(Path(path).read_text())
        ), path
        checks.append(path + " identical AST")
    old_nodes = {
        node.name: node
        for node in ast.parse(snapshots["fast_moss/v2.py"]).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert set(old_nodes) == set(MOVED)
    for name, path in MOVED.items():
        node = next(
            node
            for node in ast.parse(Path(path).read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
        )
        assert ast.dump(old_nodes[name]) == ast.dump(node), name
        checks.append(name + " identical moved AST")
    facade = ast.parse(Path("fast_moss/v2.py").read_text()).body
    assert all(
        isinstance(node, (ast.Expr, ast.ImportFrom, ast.Assign)) for node in facade
    )
    imports = {
        node.module: [item.name for item in node.names]
        for node in facade
        if isinstance(node, ast.ImportFrom)
    }
    assert imports == {
        "v2_codec": ["codec", "encode_fixed"],
        "v2_loading": ["MODEL_ID", "REVISION", "load_model"],
        "v2_runtime": ["V2Runtime", "optimized", "validate"],
    }
    checks.append("public facade re-exports only")
    return current, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Original runtime directory; omit to verify committed evidence",
    )
    args = parser.parse_args()
    if args.snapshot:
        original = {
            "fast_moss/" + p.name: p.read_text()
            for p in args.snapshot.glob("*")
            if p.suffix in (".py", ".json")
        }
        report = {
            "scope": "Pure module separation and formatting after timing; executable definitions unchanged",
            "original_runtime_sha256": {p: sha(s) for p, s in original.items()},
            "original_changed_sources": {
                p: original[p]
                for p in [
                    "fast_moss/v2.py",
                    "fast_moss/v2_loading.py",
                    "fast_moss/v2_pointwise.py",
                ]
            },
        }
        current, checks = verify(report)
        report.update(current_runtime_sha256=current, checks=checks, all_passed=True)
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
    else:
        report = json.loads(REPORT.read_text())
        current, checks = verify(report)
        assert (
            current == report["current_runtime_sha256"]
            and checks == report["checks"]
            and report["all_passed"]
        )
    print("PASS", len(checks), "module/format equivalence checks")


if __name__ == "__main__":
    main()
