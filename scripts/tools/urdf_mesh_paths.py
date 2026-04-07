"""Helpers for resolving URDF mesh paths before USD conversion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET


def prepare_urdf_for_import(urdf_path: str, usd_dir: str | None = None) -> tuple[str, list[tuple[str, str]]]:
    """Write a resolved URDF copy when mesh filenames need normalization.

    Isaac's URDF importer can silently drop visuals when ``package://`` URIs or
    relative mesh paths are not resolved as expected. This helper rewrites mesh
    filenames to absolute filesystem paths before import.

    Args:
        urdf_path: Source URDF path.
        usd_dir: Optional USD output directory. Used as a stable location for the
            generated resolved URDF copy.

    Returns:
        A tuple of ``(prepared_urdf_path, rewrites)`` where ``rewrites`` records
        each original mesh filename and its resolved absolute path.
    """

    source_path = Path(urdf_path).expanduser().resolve()
    tree = ET.parse(source_path)
    root = tree.getroot()

    rewrites: list[tuple[str, str]] = []
    package_root_cache: dict[str, Path] = {}

    for mesh_elem in root.findall(".//mesh"):
        filename = mesh_elem.get("filename")
        if not filename:
            continue
        resolved_filename = resolve_mesh_filename(
            filename=filename,
            urdf_path=source_path,
            package_root_cache=package_root_cache,
        )
        if resolved_filename != filename:
            mesh_elem.set("filename", resolved_filename)
            rewrites.append((filename, resolved_filename))

    if not rewrites:
        return str(source_path), rewrites

    resolved_dir = Path(usd_dir).expanduser().resolve() if usd_dir else source_path.parent
    resolved_dir = resolved_dir / ".resolved_urdf"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = resolved_dir / f"{source_path.stem}.resolved{source_path.suffix}"
    tree.write(resolved_path, encoding="utf-8", xml_declaration=True)
    return str(resolved_path), rewrites


def resolve_mesh_filename(filename: str, urdf_path: str | Path, package_root_cache: dict[str, Path] | None = None) -> str:
    """Resolve a URDF mesh filename to an absolute path when possible."""

    if package_root_cache is None:
        package_root_cache = {}

    urdf_path = Path(urdf_path).expanduser().resolve()
    filename = filename.strip()

    if filename.startswith("package://"):
        package_name, package_relative_path = _split_package_uri(filename)
        package_root = package_root_cache.get(package_name)
        if package_root is None:
            package_root = _find_ros_package_root(package_name, urdf_path)
            package_root_cache[package_name] = package_root
        candidate = package_root / package_relative_path
    elif filename.startswith("file://"):
        candidate = Path(filename[7:])
    elif "://" in filename:
        return filename
    else:
        candidate = Path(filename)
        if not candidate.is_absolute():
            candidate = urdf_path.parent / candidate

    candidate = candidate.expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Could not resolve mesh file '{filename}' from URDF '{urdf_path}'.")
    return candidate.as_posix()


def _split_package_uri(package_uri: str) -> tuple[str, Path]:
    package_spec = package_uri[len("package://") :]
    if "/" not in package_spec:
        raise ValueError(
            f"Invalid package URI '{package_uri}'. Expected format 'package://<package>/<relative/path>'."
        )
    package_name, relative_path = package_spec.split("/", 1)
    return package_name, Path(relative_path)


def _find_ros_package_root(package_name: str, urdf_path: Path) -> Path:
    """Locate a ROS package root by matching the name declared in package.xml."""

    nearest_package_root = _find_nearest_package_root(urdf_path.parent)
    search_roots = list(_iter_search_roots(urdf_path.parent, nearest_package_root))

    for search_root in search_roots:
        package_root = _match_package_name(search_root / "package.xml", package_name)
        if package_root is not None:
            return package_root

    for search_root in search_roots:
        for package_xml in _iter_package_xml_files(search_root):
            package_root = _match_package_name(package_xml, package_name)
            if package_root is not None:
                return package_root

    searched = ", ".join(str(path) for path in search_roots)
    raise ValueError(f"Unable to resolve ROS package '{package_name}'. Searched under: {searched}")


def _find_nearest_package_root(start_dir: Path) -> Path | None:
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "package.xml").is_file():
            return candidate
    return None


def _iter_search_roots(urdf_dir: Path, nearest_package_root: Path | None) -> Iterable[Path]:
    seen: set[Path] = set()

    def add(path: Path | None):
        if path is None:
            return
        path = path.expanduser().resolve()
        if path not in seen and path.exists():
            seen.add(path)
            yield path

    yield from add(urdf_dir)
    yield from add(nearest_package_root)
    yield from add(Path.cwd())

    ros_package_path = os.environ.get("ROS_PACKAGE_PATH", "")
    for entry in ros_package_path.split(":"):
        if not entry:
            continue
        yield from add(Path(entry))


def _iter_package_xml_files(search_root: Path, max_depth: int = 4) -> Iterable[Path]:
    search_root = search_root.expanduser().resolve()
    base_depth = len(search_root.parts)
    for current_root, dir_names, file_names in os.walk(search_root):
        current_path = Path(current_root)
        depth = len(current_path.parts) - base_depth
        if depth > max_depth:
            dir_names[:] = []
            continue
        if "package.xml" in file_names:
            yield current_path / "package.xml"


def _match_package_name(package_xml: Path, package_name: str) -> Path | None:
    if not package_xml.is_file():
        return None
    try:
        root = ET.parse(package_xml).getroot()
    except ET.ParseError:
        return None

    name_elem = root.find("name")
    if name_elem is None or name_elem.text is None:
        return None
    if name_elem.text.strip() == package_name:
        return package_xml.parent
    return None
