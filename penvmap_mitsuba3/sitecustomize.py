from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BUILD_MANIFEST = REPO_ROOT / ".mitsuba-build.json"
DLL_HANDLES: list[object] = []
CANDIDATE_RUNTIME_DIRS = (
	REPO_ROOT / ".mitsuba3-source" / "build" / "Release",
	REPO_ROOT / "external" / "mitsuba3" / "build" / "Release",
	REPO_ROOT / "external" / "mitsuba3-source" / "build" / "Release",
)


def _prepend_env_path(name: str, path: Path) -> None:
	current = os.environ.get(name, "")
	path_str = str(path)
	entries = [entry for entry in current.split(os.pathsep) if entry]
	if path_str in entries:
		return
	os.environ[name] = path_str if not current else path_str + os.pathsep + current


def _locate_llvm_c_api() -> Path | None:
	candidates: list[Path] = []
	direct_env = os.environ.get("DRJIT_LIBLLVM_PATH")
	if direct_env:
		candidates.append(Path(direct_env))
	for variable in ("LLVM_DIR", "LLVM_PATH"):
		value = os.environ.get(variable)
		if value:
			root = Path(value)
			candidates.extend([root / "bin" / "LLVM-C.dll", root / "LLVM-C.dll"])
	for entry in os.environ.get("PATH", "").split(os.pathsep):
		if entry:
			candidates.append(Path(entry) / "LLVM-C.dll")
	home = Path.home()
	candidates.extend(
		[
			home / "Documents" / "LLVM" / "bin" / "LLVM-C.dll",
			Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "LLVM" / "bin" / "LLVM-C.dll",
			Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "LLVM" / "bin" / "LLVM-C.dll",
		]
	)
	for candidate in candidates:
		if candidate.exists():
			return candidate.resolve()
	return None


def _load_manifest() -> tuple[Path, Path, Path, Path | None] | None:
	if not BUILD_MANIFEST.exists():
		return None
	try:
		data = json.loads(BUILD_MANIFEST.read_text(encoding="utf-8"))
	except (json.JSONDecodeError, OSError):
		return None

	runtime_dir = Path(data.get("runtime_dir", ""))
	python_root = Path(data.get("python_root", ""))
	package_dir = Path(data.get("package_dir", ""))
	llvm_value = data.get("llvm_c_api")
	llvm_c_api = Path(llvm_value) if llvm_value else None
	if not runtime_dir.exists() or not python_root.exists() or not package_dir.exists():
		return None
	return runtime_dir, python_root, package_dir, llvm_c_api


def _scan_build_tree() -> tuple[Path, Path, Path, Path | None] | None:
	llvm_c_api = _locate_llvm_c_api()
	for runtime_dir in CANDIDATE_RUNTIME_DIRS:
		python_root = runtime_dir / "python"
		package_dir = python_root / "mitsuba"
		plugin_path = runtime_dir / "plugins" / "penvmap.dll"
		if (package_dir / "__init__.py").exists() and plugin_path.exists():
			return runtime_dir, python_root, package_dir, llvm_c_api
	return None


def _add_dll_directory(path: Path) -> None:
	if not hasattr(os, "add_dll_directory") or not path.exists():
		return
	try:
		DLL_HANDLES.append(os.add_dll_directory(str(path)))
	except OSError:
		return


def _bootstrap_mitsuba() -> None:
	if os.environ.get("PENVMAP_DISABLE_MITSUBA_BOOTSTRAP"):
		return

	bootstrap = _load_manifest() or _scan_build_tree()
	if bootstrap is None:
		return

	runtime_dir, python_root, package_dir, llvm_c_api = bootstrap
	python_root_str = str(python_root)
	if python_root_str not in sys.path:
		sys.path.insert(0, python_root_str)

	mitsuba_dir = os.environ.get("MITSUBA_DIR")
	if not mitsuba_dir or not Path(mitsuba_dir).exists():
		os.environ["MITSUBA_DIR"] = str(runtime_dir)

	_prepend_env_path("PATH", runtime_dir)
	_prepend_env_path("PATH", package_dir)
	_add_dll_directory(runtime_dir)
	_add_dll_directory(package_dir)

	current_llvm = os.environ.get("DRJIT_LIBLLVM_PATH")
	if llvm_c_api is not None:
		if not current_llvm or not Path(current_llvm).exists():
			os.environ["DRJIT_LIBLLVM_PATH"] = str(llvm_c_api)
		_prepend_env_path("PATH", llvm_c_api.parent)
		_add_dll_directory(llvm_c_api.parent)


_bootstrap_mitsuba()