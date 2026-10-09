"""Verify preserved artifacts, runtime text, API schemas, and test discovery."""
import ast
import hashlib
import json
import os
import unittest
from pathlib import Path
from backend.core.paths import PROJECT_ROOT, project_path

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def main():
    manifest = json.loads(project_path("docs/REORGANIZATION_MANIFEST.json").read_text(encoding="utf-8"))
    for old, expected in manifest["checksums"].items():
        path = project_path(manifest["moves"].get(old, old))
        assert digest(path) == expected, f"Protected artifact changed: {old}"
    for old, expected in manifest["long_runtime_strings"].items():
        tree = ast.parse(project_path(manifest["moves"][old]).read_text(encoding="utf-8"))
        strings = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) > 100]
        assert hashlib.sha256(json.dumps(strings, ensure_ascii=False).encode()).hexdigest() == expected, f"Runtime text changed: {old}"
    # Deliberately import/load with a different working directory.
    previous = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT / "logs")
        from backend.api.app import app, engine
        schema = hashlib.sha256(json.dumps(app.openapi(), sort_keys=True).encode()).hexdigest()
        assert schema == manifest["openapi_sha256"], "API contract changed"
        assert len(engine.content.movies) == len(engine.content.overview_embeddings) == 62398
        assert engine.content.overview_embeddings.shape[1] == 384
        assert len(engine.svd.movie_factors) == len(engine.svd.movie_to_index) == 16034
        suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT), top_level_dir=str(PROJECT_ROOT))
        assert suite.countTestCases() == manifest["test_count"], "Test discovery count changed"
    finally:
        os.chdir(previous)
    report = {"protected_files": len(manifest["checksums"]), "runtime_text_modules": len(manifest["long_runtime_strings"]),
        "api_contract_unchanged": True, "routes": len(app.routes), "backend_tests_discovered": suite.countTestCases(),
        "catalog_rows": len(engine.content.movies), "embedding_shape": list(engine.content.overview_embeddings.shape),
        "svd_shape": list(engine.svd.movie_factors.shape), "working_directory_independent": True}
    project_path("logs/structure-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
