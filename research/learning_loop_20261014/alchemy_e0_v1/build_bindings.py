"""Generate exact official bindings in a pinned archive copy, without setup.py."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import grpc_tools
from grpc_tools import protoc


def build(folder, runtime_path):
    runtime = json.loads(runtime_path.read_text())
    for name, digest in runtime["upstream_source_file_hashes"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
    tree = ast.parse((folder / "setup.py").read_text())
    paths = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_DM_ALCHEMY_PROTOS" for t in node.targets)
    )
    assert len(paths) == 10
    for name in paths:
        assert ".." not in Path(name).parts and not Path(name).is_absolute()
        args = [
            "grpc_tools.protoc",
            "--proto_path=" + str(Path(grpc_tools.__file__).parent / "_proto"),
            "--proto_path=" + str(folder),
            "--python_out=" + str(folder),
            "--grpc_python_out=" + str(folder),
            str(folder / name),
        ]
        assert protoc.main(args) == 0, name
    for name, digest in runtime["generated_binding_hashes"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
    print("All 20 generated bindings match the registered runtime")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path(__file__).with_name("RUNTIME.json"))
    args = parser.parse_args()
    build(args.build, args.runtime)
