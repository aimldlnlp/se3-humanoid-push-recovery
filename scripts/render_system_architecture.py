"""Build the paper-style system architecture from its TikZ source.

The controller implementation and experiment data are intentionally outside
this renderer. The final figure is sourced from
``docs/figures/system_architecture.tex``; the optional candidate sheet keeps
the three layouts used during visual selection reproducible.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
FINAL_SOURCE = ROOT / "docs" / "figures" / "system_architecture.tex"
CANDIDATE_SOURCE = ROOT / "docs" / "figures" / "system_architecture_candidates.tex"
FINAL_PDF = ROOT / "results" / "figures" / "pdf" / "system_architecture.pdf"
FINAL_PNG = ROOT / "results" / "figures" / "png" / "system_architecture.png"
CANDIDATE_DIR = ROOT / "docs" / "figures" / "candidates"


def _tool(name: str, environment_name: str) -> str:
    configured = os.environ.get(environment_name)
    if configured:
        path = Path(configured)
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"{environment_name} does not point to a file: {path}")
    discovered = shutil.which(name)
    if discovered:
        return discovered
    raise FileNotFoundError(
        f"Could not find {name!r}. Install it or set {environment_name}."
    )


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def _compile(
    source: Path,
    output_dir: Path,
    compiler: str,
    compiler_kind: str,
    environment: dict[str, str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if compiler_kind == "pdflatex":
        command = [
            compiler,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(output_dir),
            source.name,
        ]
    else:
        command = [compiler, "--keep-logs", "--outdir", str(output_dir)]
        bundle = environment.get("TECTONIC_BUNDLE_URL")
        if bundle:
            command.extend(["-b", bundle])
        command.append(source.name)
    _run(command, cwd=source.parent, environment=environment)
    compiled = output_dir / f"{source.stem}.pdf"
    if not compiled.is_file():
        raise RuntimeError(f"The selected LaTeX compiler did not produce the expected PDF: {compiled}")
    return compiled


def _render_png(
    pdf: Path,
    output: Path,
    pdftoppm: str,
    dpi: int,
    environment: dict[str, str],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.with_suffix("")
    command = [
        pdftoppm,
        "-png",
        "-r",
        str(dpi),
        "-singlefile",
        str(pdf),
        str(prefix),
    ]
    _run(command, cwd=ROOT, environment=environment)
    rendered = prefix.with_suffix(".png")
    if not rendered.is_file():
        raise RuntimeError(f"pdftoppm did not produce the expected PNG: {rendered}")


def _render_candidate_pages(
    pdf: Path,
    output_dir: Path,
    pdftoppm: str,
    dpi: int,
    environment: dict[str, str],
) -> None:
    names = (
        "system_architecture_candidate_a.png",
        "system_architecture_candidate_b.png",
        "system_architecture_candidate_c.png",
    )
    for page, name in enumerate(names, start=1):
        output = output_dir / name
        prefix = output.with_suffix("")
        command = [
            pdftoppm,
            "-png",
            "-r",
            str(dpi),
            "-f",
            str(page),
            "-l",
            str(page),
            "-singlefile",
            str(pdf),
            str(prefix),
        ]
        _run(command, cwd=ROOT, environment=environment)
        rendered = prefix.with_suffix(".png")
        if not rendered.is_file():
            raise RuntimeError(f"pdftoppm did not produce the expected PNG: {rendered}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile and render the TikZ system-architecture figure."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG render resolution (default: 300).",
    )
    parser.add_argument(
        "--no-candidates",
        action="store_true",
        help="Only build the selected final layout.",
    )
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    try:
        try:
            compiler = _tool("tectonic", "TECTONIC_BIN")
            compiler_kind = "tectonic"
        except FileNotFoundError:
            compiler = _tool("pdflatex", "PDFLATEX_BIN")
            compiler_kind = "pdflatex"
        pdftoppm = _tool("pdftoppm", "PDFTOPPM_BIN")
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 2

    environment = os.environ.copy()
    cache_dir = Path(
        environment.get(
            "TECTONIC_CACHE_DIR",
            ROOT / ".tmp" / "tectonic-cache",
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    environment["TECTONIC_CACHE_DIR"] = str(cache_dir)

    FINAL_PDF.parent.mkdir(parents=True, exist_ok=True)
    FINAL_PNG.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    build_root = ROOT / ".tmp"
    build_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="system_architecture_", dir=build_root
    ) as temp:
        build_dir = Path(temp)
        final_pdf = _compile(FINAL_SOURCE, build_dir / "final", compiler, compiler_kind, environment)
        shutil.copy2(final_pdf, FINAL_PDF)
        _render_png(final_pdf, FINAL_PNG, pdftoppm, args.dpi, environment)

        if not args.no_candidates:
            candidates_pdf = _compile(
                CANDIDATE_SOURCE,
                build_dir / "candidates",
                compiler,
                compiler_kind,
                environment,
            )
            stable_candidates_pdf = CANDIDATE_DIR / "system_architecture_candidates.pdf"
            shutil.copy2(candidates_pdf, stable_candidates_pdf)
            _render_candidate_pages(
                candidates_pdf,
                CANDIDATE_DIR,
                pdftoppm,
                args.dpi,
                environment,
            )

    print(f"Wrote {FINAL_PDF}")
    print(f"Wrote {FINAL_PNG}")
    if not args.no_candidates:
        print(f"Wrote candidate sheet and previews under {CANDIDATE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
