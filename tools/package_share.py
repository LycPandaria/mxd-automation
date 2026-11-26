"""打包可分享的源码与当前运行模型；不改动个人配置或原始数据。"""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "mxd-automation"


def collect() -> dict[str, bytes]:
    files: dict[str, bytes] = {}

    def add(path: Path, destination: str | None = None) -> None:
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"拒绝打包项目外文件或符号链接: {path}")
        data = path.read_bytes()
        if path.suffix == ".py":
            ast.parse(data, filename=str(path))
        files[destination or path.relative_to(ROOT).as_posix()] = data

    for directory, suffix in (
        ("src", ".py"), ("ui", ".py"), ("utils", ".py"),
        ("tools", ".py"), ("tests", ".py"), ("docs", ".md"),
    ):
        for path in sorted((ROOT / directory).rglob(f"*{suffix}")):
            if not any(part.startswith(".") or part == "__pycache__"
                       for part in path.relative_to(ROOT).parts):
                add(path)

    for name in (
        "main.py", "README.md", "requirements.txt", "requirements-run.txt",
        "replay_label_check.bat", "train/README.md",
        "train/scripts/01_auto_annotate/main.py",
        "train/scripts/02_train_yolo/main.py",
        "dist/best.onnx", "models/lie_detector/best.onnx",
        "models/lie_detector/labels.txt", "models/lie_detector/model_info.json",
        "models/lie_target/best.onnx", "models/lie_target/model_info.json",
        "config/share.example.yaml",
    ):
        add(ROOT / name)
    for path in sorted((ROOT / "train/windows_lie_target").iterdir()):
        if path.is_file() and path.suffix in {".py", ".bat", ".txt"}:
            add(path)

    add(ROOT / "config/share.example.yaml", "config/user.yaml")
    add(ROOT / "docs/SHARING.md", "先看这里.md")
    # 不携带原开发机的绝对路径；收件人按自己的数据位置修改。
    data_config = (ROOT / "train/data/data.yaml").read_text(encoding="utf-8")
    lines = ["path: train/data" if line.startswith("path:") else line
             for line in data_config.splitlines()]
    files["train/data/data.yaml"] = ("\n".join(lines) + "\n").encode("utf-8")
    manifest = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n"
                       for name, data in sorted(files.items()))
    files["MANIFEST.sha256"] = manifest.encode("utf-8")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "share/mxd-automation-share.zip")
    args = parser.parse_args()
    files = collect()
    output = args.output.resolve()
    if output.suffix.lower() != ".zip":
        parser.error("输出路径必须以 .zip 结尾")
    if output.exists():
        parser.error(f"输出已存在，请另选 --output 路径：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(f"{PREFIX}/{name}", data)
    with ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"压缩包校验失败: {bad}")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8")
    print(f"{output}\n{len(files)} files, {output.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
