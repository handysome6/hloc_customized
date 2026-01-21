from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import struct

import numpy as np

from .dataset import FrameData
from .geometry import Pose, transform_points


_PLY_TYPE_MAP: dict[str, Tuple[str, int]] = {
    "char": ("b", 1),
    "uchar": ("B", 1),
    "short": ("h", 2),
    "ushort": ("H", 2),
    "int": ("i", 4),
    "uint": ("I", 4),
    "float": ("f", 4),
    "double": ("d", 8),
}


def _parse_ply_header(data: bytes, path: Path) -> Tuple[str, int, List[str], int]:
    header_end = data.find(b"end_header")
    if header_end == -1:
        raise ValueError(f"Invalid PLY header in {path}")
    header_end = data.find(b"\n", header_end)
    if header_end == -1:
        raise ValueError(f"Invalid PLY header in {path}")
    header_end += 1
    header_text = data[:header_end].decode("ascii", errors="replace")
    lines = [line.strip() for line in header_text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("ply"):
        raise ValueError(f"Missing PLY magic in {path}")

    fmt_line = next((line for line in lines if line.startswith("format ")), "")
    if not fmt_line:
        raise ValueError(f"Missing PLY format in {path}")
    fmt = fmt_line.split()[1]

    count_line = next((line for line in lines if line.startswith("element vertex")), "")
    if not count_line:
        raise ValueError(f"Missing vertex count in {path}")
    vertex_count = int(count_line.split()[-1])

    properties: List[str] = []
    in_vertex = False
    for line in lines:
        if line.startswith("element "):
            in_vertex = line.startswith("element vertex")
            continue
        if in_vertex and line.startswith("property "):
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "list":
                raise ValueError(f"List properties not supported in {path}")
            if len(parts) >= 3:
                properties.append(parts[1])
            continue
        if in_vertex and line.startswith("end_header"):
            break
    return fmt, vertex_count, properties, header_end


def load_ply(path: Path) -> np.ndarray:
    data = path.read_bytes()
    fmt, vertex_count, properties, header_end = _parse_ply_header(data, path)
    payload = data[header_end:]

    if fmt == "ascii":
        text = payload.decode("ascii", errors="ignore")
        points = []
        for row in text.splitlines():
            values = row.strip().split()
            if len(values) < 3:
                continue
            points.append([float(values[0]), float(values[1]), float(values[2])])
        return np.array(points, dtype=np.float64)

    if fmt != "binary_little_endian":
        raise ValueError(f"Unsupported PLY format '{fmt}' in {path}")

    if not properties:
        raise ValueError(f"No vertex properties found in {path}")

    fmt_codes = []
    stride = 0
    for prop in properties:
        if prop not in _PLY_TYPE_MAP:
            raise ValueError(f"Unsupported PLY type '{prop}' in {path}")
        code, size = _PLY_TYPE_MAP[prop]
        fmt_codes.append(code)
        stride += size
    struct_fmt = "<" + "".join(fmt_codes)
    unpacker = struct.Struct(struct_fmt)

    points = []
    offset = 0
    for _ in range(vertex_count):
        chunk = payload[offset : offset + stride]
        if len(chunk) < stride:
            break
        values = unpacker.unpack(chunk)
        points.append([float(values[0]), float(values[1]), float(values[2])])
        offset += stride
    return np.array(points, dtype=np.float64)


def save_ply_ascii(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{x} {y} {z}\n")


def fuse_point_clouds(
    frames: List[FrameData],
    poses: Dict[str, Pose],
    output_path: Path,
) -> None:
    all_points = []
    for frame in frames:
        pose = poses.get(frame.frame_id)
        if pose is None:
            continue
        points = load_ply(frame.cloud_path)
        points_world = transform_points(points, pose)
        all_points.append(points_world)
    if not all_points:
        raise ValueError("No point clouds available to fuse.")
    fused = np.vstack(all_points)
    save_ply_ascii(output_path, fused)
