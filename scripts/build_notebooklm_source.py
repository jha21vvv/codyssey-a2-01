"""Google NotebookLM 업로드용 순수 소스 코드 단일 텍스트 빌더 (Step 6).

테스트 코드(`tests/`), 프롬프트 가이드, 기획 문서를 제외하고
오직 실제 핵심 구현 소스 코드(`src/*.py`)만을 모아 `NOTEBOOKLM_FULL_SOURCE.txt`로 통합합니다.
"""

from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
TARGET_FILES = [
    "src/__init__.py",
    "src/linear_algebra.py",
    "src/calculus.py",
    "src/optimizer.py",
]
OUTPUT_FILE = ROOT_DIR / "NOTEBOOKLM_FULL_SOURCE.txt"


def build_source_bundle() -> Path:
    """핵심 소스 파일들을 하나의 텍스트 파일로 병합합니다."""
    sections = [
        "================================================================================",
        "CODYSSEY A2-01 — AI 수학 기반 핵심 구현 소스 코드 통합본 (NotebookLM용)",
        "포함 파일: " + ", ".join(TARGET_FILES),
        "================================================================================\n",
    ]

    for rel_path in TARGET_FILES:
        file_path = ROOT_DIR / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"대상 소스 파일을 찾을 수 없습니다: {file_path}")
        content = file_path.read_text(encoding="utf-8")
        sections.append(f"\n{'#' * 80}")
        sections.append(f"# FILE: {rel_path}")
        sections.append(f"{'#' * 80}\n")
        sections.append(content.rstrip() + "\n")

    OUTPUT_FILE.write_text("\n".join(sections), encoding="utf-8")
    return OUTPUT_FILE


if __name__ == "__main__":
    out = build_source_bundle()
    print(f"Built NotebookLM source bundle: {out} ({out.stat().st_size} bytes)")
