#!/usr/bin/env python3
"""Batch convert GLB files and textures to PSG assets, with a simple UI.

Workflow:

* Uses root-level `Pres` folder (next to this script) as the input AND output
  folder for GLB/PSG.
* Blender exporter writes `glb_bounds.json` into that Pres folder, with
  world-space bounds for each GLB (by mesh name / GLB stem).
* On COMPILE:
  - Reads `Pres/glb_bounds.json` to get tile coordinates for each GLB.
  - Converts all .glb in ./Pres to .psg (in-place), then deletes the .glb.
  - Asks for a map name (must start with DIST_, auto-added if needed).
  - Creates a folder DIST_<MapName> next to this script (the "map root").
  - Inside it creates:
        cPres_Global
        cSim_Global
        cPres_<cx>_<cy>_high (from JSON coords)
        cSim_<cx>_<cy>_high  (same tiles as Pres)
  - Moves all PSGs from ./Pres into cPres_Global.
  - Copies all PSGs from ./Sim into cSim_Global (if present).
  - Copies all files from DONOTREMOVE/cpres into cPres_Global as templates.
  - Copies all files from DONOTREMOVE/ctex into the root of DIST_<MapName>,
    renaming them to: DIST_<MapName>_Tex.<ext>
  - Runs `Stream File Tool.exe` (from DONOTREMOVE) with:
        Stream File Tool.exe pack <cPres* dirs...> --platform=p
        Stream File Tool.exe pack <cSim* dirs...>  --platform=p
    with *_Global folders first, others sorted.
  - After packing, deletes all cPres* / cSim* / cTex* folders in DIST_<MapName>.
  - Deletes `glb_bounds.json` from Pres when finished.
"""

from __future__ import annotations

import glbtopsg
import base64
import hashlib
import io
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Callable, Dict, Optional, Sequence, Set, Tuple


def _get_base_dir() -> str:
    """Return the folder that should be treated as the 'script directory'.

    - When frozen (PyInstaller one-file EXE), this is the folder containing the .exe.
    - When running as a normal script, this is the folder containing this .py file.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.realpath(__file__))

# --- UI imports ---
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
except Exception:
    tk = None  # type: ignore

try:  # Pillow is optional but improves alpha detection
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - Pillow is optional
    Image = None  # type: ignore


TEXTURE_ID_START = 0x1AC
MESH_DIFFUSE_ID_START = 0x360
MESH_TRANSPARENCY_ID_START = 0x320
MESH_ROUGHNESS_ID_START = 0x340
MESH_NORMAL_ID_START = 0x3A0
MESH_LIGHTMAP_ID_START = 0x2E0
ID_FIELD_LENGTH = 8
TEMPLATE_ID_SETS = (
    (0x00000280, 0x00000B3C),
    (0x00000830, 0x00000B6C),
)
_USED_TEMPLATE_IDS: Set[bytes] = set()


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def _generate_random_psg_stem(used: Set[str]) -> str:
    """Return a unique ``1014XXXXXXXXXXXX`` stem not present in *used*."""
    while True:
        candidate = "1014" + f"{random.randrange(0, 10 ** 12):012d}"
        if candidate not in used:
            used.add(candidate)
            return candidate


def _write_id_field(buf: bytearray, start: int, id_bytes: bytes) -> None:
    """Write *id_bytes* into *buf* starting at *start* after validation."""
    if len(id_bytes) != ID_FIELD_LENGTH:
        raise ValueError("Identifier must be exactly 8 bytes long")
    end = start + ID_FIELD_LENGTH
    if len(buf) < end:
        buf.extend(b"\x00" * (end - len(buf)))
    buf[start:end] = id_bytes


def _write_be32(buf: bytearray, offset: int, value: int) -> None:
    """Write a big-endian 32-bit integer *value* at *offset* in *buf*."""
    if not (0 <= value <= 0xFFFFFFFF):
        raise ValueError("Value must fit in 32 bits")
    end = offset + 4
    if len(buf) < end:
        buf.extend(b"\x00" * (end - len(buf)))
    struct.pack_into(">I", buf, offset, value)


def _randomize_template_ids(psg_data: bytearray) -> None:
    """Randomize paired identifier bytes at known template offsets."""

    def _new_unique_id() -> bytes:
        while True:
            candidate = os.urandom(ID_FIELD_LENGTH)
            if candidate not in _USED_TEMPLATE_IDS:
                _USED_TEMPLATE_IDS.add(candidate)
                return candidate

    def _write_pair(offsets: Tuple[int, int]) -> None:
        new_id = _new_unique_id()
        for off in offsets:
            end = off + len(new_id)
            if end > len(psg_data):
                raise ValueError(f"Template is too small to write ID at 0x{off:X}")
            psg_data[off:end] = new_id

    for pair in TEMPLATE_ID_SETS:
        _write_pair(pair)


def _randomized_template_copy(template_path: str) -> Tuple[str, str]:
    """Return a temp file path containing the template with randomized IDs."""

    with open(template_path, "rb") as f:
        psg_bytes = bytearray(f.read())

    _randomize_template_ids(psg_bytes)

    temp_dir = tempfile.mkdtemp(prefix="psg_template_")
    temp_path = os.path.join(temp_dir, os.path.basename(template_path))
    with open(temp_path, "wb") as f:
        f.write(psg_bytes)

    return temp_path, temp_dir


def _resolve_io_directory(base_dir: str, *names: str) -> str:
    """Return the first directory inside *base_dir* that matches *names*."""
    for name in names:
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            return path
    # Retry with simple case variations
    for name in names:
        for variant in {name, name.lower(), name.upper(), name.capitalize()}:
            path = os.path.join(base_dir, variant)
            if os.path.isdir(path):
                return path
    # Fall back to the first name even if it does not exist
    return os.path.join(base_dir, names[0])


def _detect_alpha_pillow(img_path: str) -> bool:
    """Return ``True`` when *img_path* contains an alpha channel."""
    if Image is None:
        return False
    try:
        with Image.open(img_path) as im:
            return "A" in im.getbands() or im.mode.endswith("A")
    except Exception:
        return False


def _detect_alpha_bytes(img_data: bytes) -> bool:
    """Return ``True`` if *img_data* decodes to an image with alpha."""
    if Image is None:
        return False
    try:
        with Image.open(io.BytesIO(img_data)) as im:
            return "A" in im.getbands() or im.mode.endswith("A")
    except Exception:
        return False


def _find_manual_texture(input_dir: str, base_name: str) -> Optional[str]:
    """Return a side-car texture next to the GLB matching *base_name* if any."""
    priority = {".dds": 0, ".png": 1, ".jpg": 2, ".jpeg": 3}
    base_lower = base_name.lower()
    best_path: Optional[str] = None
    best_score: Optional[int] = None
    try:
        for entry in os.listdir(input_dir):
            if not os.path.isfile(os.path.join(input_dir, entry)):
                continue
            name, ext = os.path.splitext(entry)
            ext_lower = ext.lower()
            if name.lower() != base_lower or ext_lower not in priority:
                continue
            score = priority[ext_lower]
            if best_score is None or score < best_score:
                best_score = score
                best_path = os.path.join(input_dir, entry)
                if score == 0:  # DDS has highest priority
                    break
    except FileNotFoundError:
        return None
    return best_path


def _build_image_registry_key(
    image_name: Optional[str],
    image_bytes: Optional[bytes],
    fallback_label: str = "",
) -> Optional[Tuple[str, str]]:
    """Return a registry key that distinguishes images by name and content.

    *image_name* is preferred, but when it is absent we can still create a
    useful key as long as *image_bytes* is provided. The optional
    *fallback_label* is used in place of the missing name so lightmaps or other
    nameless images still participate in the registry.
    """

    if image_bytes is None:
        return None
    key_name = image_name or fallback_label
    if not key_name:
        return None
    digest = hashlib.sha1(image_bytes).hexdigest()
    return key_name, digest


# ---------------------------------------------------------------------------
# GLB parsing helpers (for textures)
# ---------------------------------------------------------------------------


def _parse_glb_container(glb_path: str) -> Optional[Tuple[dict, Optional[bytes]]]:
    """Return ``(gltf_dict, bin_chunk)`` for the GLB at *glb_path* or ``None``."""
    try:
        with open(glb_path, "rb") as f:
            buf = f.read()
    except OSError:
        return None

    if len(buf) < 20:
        return None

    magic, version, length = struct.unpack_from("<4sII", buf, 0)
    if magic != b"glTF" or length > len(buf):
        return None

    offset = 12
    json_str = None
    bin_data = None
    while offset + 8 <= length:
        chunk_len, chunk_type = struct.unpack_from("<I4s", buf, offset)
        offset += 8
        chunk_data = buf[offset:offset + chunk_len]
        if chunk_type == b"JSON":
            try:
                json_str = chunk_data.decode("utf-8")
            except Exception:
                return None
        elif chunk_type == b"BIN\x00":
            bin_data = chunk_data
        offset += chunk_len

    if json_str is None:
        return None

    try:
        gltf = json.loads(json_str)
    except Exception:
        return None
    return gltf, bin_data


def _resolve_texture_image_index(gltf: dict, texture_index: Optional[int]) -> Optional[int]:
    """Map a texture index to an image index using the glTF manifest."""
    if texture_index is None:
        return None
    textures = gltf.get("textures", [])
    if not isinstance(texture_index, int) or texture_index < 0 or texture_index >= len(textures):
        return None
    tex = textures[texture_index]
    src = tex.get("source")
    if not isinstance(src, int):
        return None
    images = gltf.get("images", [])
    if src < 0 or src >= len(images):
        return None
    return src


def _load_image_from_gltf(
    glb_path: str,
    gltf: dict,
    bin_chunk: Optional[bytes],
    image_index: Optional[int],
) -> Optional[Tuple[str, bytes]]:
    """Return ``(mime_type, data)`` for *image_index* or ``None`` on failure."""
    if image_index is None:
        return None
    images = gltf.get("images", [])
    if image_index < 0 or image_index >= len(images):
        return None
    image = images[image_index]

    uri = image.get("uri")
    if isinstance(uri, str) and uri:
        if uri.startswith("data:"):
            try:
                header, payload = uri.split(",", 1)
                if "base64" in header:
                    data = base64.b64decode(payload)
                    mime = header.split(";")[0].split(":", 1)[-1]
                    return mime or "application/octet-stream", data
            except Exception:
                return None
        else:
            img_path = os.path.join(os.path.dirname(glb_path), uri)
            if not os.path.isfile(img_path):
                return None
            try:
                with open(img_path, "rb") as f:
                    data = f.read()
            except OSError:
                return None
            mime = image.get("mimeType")
            if not mime:
                ext = os.path.splitext(uri)[1].lower()
                mime = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".dds": "image/vnd-ms.dds",
                }.get(ext, "application/octet-stream")
            return mime, data

    buffer_view = image.get("bufferView")
    if not isinstance(buffer_view, int):
        return None
    buffer_views = gltf.get("bufferViews", [])
    if buffer_view < 0 or buffer_view >= len(buffer_views):
        return None
    bv = buffer_views[buffer_view]
    byte_offset = int(bv.get("byteOffset", 0))
    byte_length = int(bv.get("byteLength", 0))
    if bin_chunk is None or byte_offset < 0 or byte_length <= 0 or byte_offset + byte_length > len(bin_chunk):
        return None
    mime = image.get("mimeType") or "application/octet-stream"
    return mime, bytes(bin_chunk[byte_offset:byte_offset + byte_length])


def _extract_material_texture_indices(
    gltf: dict,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], bool]:
    """Return indices for (base, normal, roughness, emission, alpha flag)."""
    base_index: Optional[int] = None
    normal_index: Optional[int] = None
    roughness_index: Optional[int] = None
    emission_index: Optional[int] = None
    alpha_connected = False

    materials = gltf.get("materials", []) or []
    for mat in materials:
        if base_index is None:
            pbr = mat.get("pbrMetallicRoughness") or {}
            base_tex = pbr.get("baseColorTexture") or {}
            base_index = base_tex.get("index") if isinstance(base_tex, dict) else None
            if base_index is not None:
                alpha_mode = str(mat.get("alphaMode", "OPAQUE")).upper()
                if alpha_mode in {"BLEND", "MASK"}:
                    alpha_connected = True
                base_factor = pbr.get("baseColorFactor")
                if isinstance(base_factor, Sequence) and len(base_factor) >= 4:
                    try:
                        if float(base_factor[3]) < 0.999:
                            alpha_connected = True
                    except Exception:
                        pass
        if normal_index is None:
            normal_tex = mat.get("normalTexture") or {}
            if isinstance(normal_tex, dict):
                idx = normal_tex.get("index")
                if isinstance(idx, int):
                    normal_index = idx
        if roughness_index is None:
            pbr = mat.get("pbrMetallicRoughness") or {}
            rough_tex = pbr.get("metallicRoughnessTexture") or {}
            if isinstance(rough_tex, dict):
                idx = rough_tex.get("index")
                if isinstance(idx, int):
                    roughness_index = idx
        if emission_index is None:
            emissive_tex = mat.get("emissiveTexture") or {}
            if isinstance(emissive_tex, dict):
                idx = emissive_tex.get("index")
                if isinstance(idx, int):
                    emission_index = idx

        if (
            base_index is not None
            and normal_index is not None
            and roughness_index is not None
            and emission_index is not None
        ):
            break

    return base_index, normal_index, roughness_index, emission_index, alpha_connected


def _get_image_name(gltf: dict, image_index: Optional[int]) -> Optional[str]:
    """Return the human-readable name for *image_index* if one exists."""
    if image_index is None:
        return None
    images = gltf.get("images", [])
    if image_index < 0 or image_index >= len(images):
        return None
    img = images[image_index]
    name = img.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    uri = img.get("uri")
    if isinstance(uri, str) and uri:
        return os.path.basename(uri)
    return None


# ---------------------------------------------------------------------------
# DDS helpers
# ---------------------------------------------------------------------------


def _compress_image_to_dds(src_path: str, dst_path: str, compression: str) -> None:
    """Compress *src_path* into *dst_path* using ImageMagick with *compression*."""
    cmd = [
        "magick",
        src_path,
        "-define",
        f"dds:compression={compression}",
        "-define",
        "dds:mipmaps=6",
        dst_path,
    ]
    subprocess.run(cmd, check=True)


def _parse_dds_header(dds_path: str) -> Tuple[int, int, int, int, bytes, int]:
    """Return ``(width, height, fmt_code, mips, payload, block_size)`` for a DDS."""
    with open(dds_path, "rb") as f:
        data = f.read()

    if len(data) < 128 or data[:4] != b"DDS ":
        raise ValueError(f"{dds_path} is not a valid DDS file")

    height = int.from_bytes(data[12:16], "little")
    width = int.from_bytes(data[16:20], "little")
    mips = int.from_bytes(data[28:32], "little")

    fourcc = None
    for off in (84, 88, 92):
        if off + 4 <= len(data):
            cc = data[off:off + 4]
            if cc in (b"DXT1", b"DXT3", b"DXT5", b"DX10"):
                fourcc = cc
                break
    if fourcc is None:
        for cc in (b"DXT1", b"DXT3", b"DXT5", b"DX10"):
            idx = data.find(cc)
            if 0 <= idx < 128:
                fourcc = cc
                break
    if fourcc is None:
        raise ValueError(f"Could not determine FourCC in {dds_path}")

    if fourcc == b"DXT1":
        fmt = 0x86
        block_size = 8
    elif fourcc == b"DXT3":
        fmt = 0x87
        block_size = 16
    else:  # Treat DXT5/DX10 as interpolated alpha
        fmt = 0x88
        block_size = 16

    header_size = 128 + (20 if fourcc == b"DX10" else 0)
    raw_data = data[header_size:]
    return width, height, fmt, mips, raw_data, block_size


def _dds_has_alpha(fmt_code: int) -> bool:
    """Return ``True`` if the DDS format carries alpha information."""
    return fmt_code != 0x86  # 0x86 corresponds to opaque DXT1


# ---------------------------------------------------------------------------
# PSG helpers
# ---------------------------------------------------------------------------


def _read_psg_dictionary(psg_data: bytes) -> Tuple[int, int, int, list]:
    """Return dictionary metadata and entries from *psg_data*."""
    if len(psg_data) < 0x60:
        raise ValueError("PSG data too small to contain a valid dictionary")
    num_entries = struct.unpack_from(">I", psg_data, 0x20)[0]
    dict_start = struct.unpack_from(">I", psg_data, 0x30)[0]
    main_base = struct.unpack_from(">I", psg_data, 0x44)[0]
    entries = []
    for i in range(num_entries):
        off = dict_start + i * 0x18
        if off + 0x18 > len(psg_data):
            break
        ptr = struct.unpack_from(">I", psg_data, off)[0]
        size = struct.unpack_from(">I", psg_data, off + 8)[0]
        type_index = struct.unpack_from(">I", psg_data, off + 0x10)[0]
        type_id = struct.unpack_from(">I", psg_data, off + 0x14)[0]
        entries.append((off, ptr, size, type_id, type_index))
    return num_entries, dict_start, main_base, entries


def _extract_green_channel_texture(image_path: str) -> Tuple[str, Optional[str]]:
    """Return ``(path, temp_dir)`` where *path* is a green-channel-only copy."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".dds":
        return image_path, None

    if Image is not None:
        try:
            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
                _, green, _ = rgb.split()
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, "roughness.png")
                green.save(temp_path)
                return temp_path, temp_dir
        except Exception:
            pass

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "roughness.png")
    try:
        subprocess.run(
            [
                "magick",
                image_path,
                "-channel",
                "G",
                "-separate",
                temp_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return temp_path, temp_dir
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return image_path, None


def _create_texture_psg(
    template_path: str,
    image_path: Optional[str],
    out_path: str,
    use_alpha: bool,
    id_bytes: bytes,
) -> None:
    """Create a texture PSG at *out_path* based on *template_path* and *image_path*."""
    with open(template_path, "rb") as f:
        data = bytearray(f.read())

    temp_dir: Optional[str] = None
    temp_dds: Optional[str] = None
    try:
        if image_path is not None:
            ext = os.path.splitext(image_path)[1].lower()
            if ext == ".dds":
                dds_path = image_path
            else:
                compression = "dxt5" if use_alpha else "dxt1"
                temp_dir = tempfile.mkdtemp()
                temp_dds = os.path.join(temp_dir, "tmp.dds")
                _compress_image_to_dds(image_path, temp_dds, compression)
                dds_path = temp_dds

            width, height, fmt, mips, raw_data, block_size = _parse_dds_header(dds_path)

            _, _, main_base, entries = _read_psg_dictionary(data)
            base_entry = None
            tex_info_entry = None
            for entry in entries:
                entry_off, ptr, size, type_id, _ = entry
                if 0x00010030 <= type_id <= 0x0001003F and base_entry is None:
                    base_entry = (entry_off, ptr, size)
                elif type_id == 0x000200E8 and tex_info_entry is None:
                    tex_info_entry = (entry_off, ptr, size)
                if base_entry and tex_info_entry:
                    break
            if base_entry is None or tex_info_entry is None:
                raise ValueError("Template PSG missing expected texture entries")

            entry_off, ptr, old_size = base_entry
            old_base_offset = main_base + ptr
            old_base_end = old_base_offset + old_size
            new_payload = bytearray(raw_data)
            delta = len(new_payload) - old_size

            new_data = bytearray()
            new_data += data[:old_base_offset]
            new_data += new_payload
            new_data += data[old_base_end:]

            for entry in entries:
                e_off, e_ptr, _, e_type_id, _ = entry
                if 0x00010030 <= e_type_id <= 0x0001003F:
                    continue
                if e_ptr >= old_base_end:
                    struct.pack_into(">I", new_data, e_off, e_ptr + delta)

            tex_entry_off, tex_ptr, tex_size = tex_info_entry
            if tex_ptr >= old_base_end:
                new_tex_ptr = tex_ptr + delta
            else:
                new_tex_ptr = tex_ptr

            (
                orig_fmt,
                orig_mips,
                orig_dimension,
                orig_cubemap,
                orig_remap,
                orig_width,
                orig_height,
                orig_depth,
                orig_location,
                orig_pad1,
                orig_pitch,
                orig_offset,
                orig_buffer_ptr,
                orig_store_type,
                orig_store_flags,
                orig_unknown2,
                orig_pad2,
                orig_fmt2,
            ) = struct.unpack_from(
                ">BBBBIHHHBBIIIIIHBB", new_data, new_tex_ptr
            )

            blocks_w = max(1, width // 4)
            pitch = blocks_w * block_size
            mip_count = mips if mips > 0 else 1

            new_struct = struct.pack(
                ">BBBBIHHHBBIIIIIHBB",
                fmt,
                mip_count & 0xFF,
                orig_dimension,
                orig_cubemap,
                orig_remap,
                width,
                height,
                orig_depth,
                orig_location,
                orig_pad1,
                pitch,
                0,
                orig_buffer_ptr,
                orig_store_type,
                orig_store_flags,
                orig_unknown2,
                orig_pad2,
                fmt,
            )
            new_data[new_tex_ptr:new_tex_ptr + len(new_struct)] = new_struct

            data = new_data

        _write_id_field(data, TEXTURE_ID_START, id_bytes)

        payload_start = 0x248
        payload_size = len(data) - payload_start if len(data) > payload_start else 0
        _write_be32(data, 0x6C, payload_size)
        _write_be32(data, 0x1F0, payload_size)

        with open(out_path, "wb") as out_f:
            out_f.write(data)
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def _patch_mesh_psg(
    mesh_path: str,
    id_bytes: bytes,
    use_alpha: bool,
    *,
    normal_id_bytes: Optional[bytes] = None,
    roughness_id_bytes: Optional[bytes] = None,
    lightmap_id_bytes: Optional[bytes] = None,
) -> None:
    """Insert texture identifiers into the mesh PSG, including lightmaps."""
    with open(mesh_path, "rb") as f:
        mesh_data = bytearray(f.read())

    def ensure_capacity(offset: int) -> None:
        end = offset + ID_FIELD_LENGTH
        if len(mesh_data) < end:
            mesh_data.extend(b"\x00" * (end - len(mesh_data)))

    ensure_capacity(MESH_DIFFUSE_ID_START)
    _write_id_field(mesh_data, MESH_DIFFUSE_ID_START, id_bytes)

    if use_alpha:
        ensure_capacity(MESH_TRANSPARENCY_ID_START)
        _write_id_field(mesh_data, MESH_TRANSPARENCY_ID_START, id_bytes)

    if roughness_id_bytes is not None:
        ensure_capacity(MESH_ROUGHNESS_ID_START)
        _write_id_field(mesh_data, MESH_ROUGHNESS_ID_START, roughness_id_bytes)
    if normal_id_bytes is not None:
        ensure_capacity(MESH_NORMAL_ID_START)
        _write_id_field(mesh_data, MESH_NORMAL_ID_START, normal_id_bytes)
    if lightmap_id_bytes is not None:
        ensure_capacity(MESH_LIGHTMAP_ID_START)
        _write_id_field(mesh_data, MESH_LIGHTMAP_ID_START, lightmap_id_bytes)

    with open(mesh_path, "wb") as f:
        f.write(mesh_data)


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------


def _bytes_to_temp_file(data: bytes, suffix: str) -> Tuple[str, str]:
    """Write *data* to a temporary file and return ``(path, temp_dir)``."""
    temp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(temp_dir, f"image{suffix}")
    with open(tmp_path, "wb") as f:
        f.write(data)
    return tmp_path, temp_dir


def convert_glb_directory(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    template_dir: Optional[str] = None,
    glb_converter: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """Convert all GLBs in *input_dir* to PSGs in *output_dir*."""
    def _log(msg: str) -> None:
        if log is not None:
            log(msg)
        else:
            print(msg)

    script_dir = _get_base_dir()
    if template_dir is None:
        template_dir = os.path.join(script_dir, "DONOTREMOVE")

    if input_dir is None:
        input_dir = _resolve_io_directory(script_dir, "input", "Input")
    if output_dir is None:
        output_dir = _resolve_io_directory(script_dir, "output", "Output")

    mesh_template = os.path.join(template_dir, "Mesh.psg")
    alpha_mesh_template = os.path.join(template_dir, "AlphaMesh.psg")
    tex_template = os.path.join(template_dir, "Texture.psg")
    if glb_converter is None:
        glb_converter = os.path.join(script_dir, "glbtopsg.py")

    if not os.path.isfile(mesh_template):
        raise FileNotFoundError(f"Missing mesh template: {mesh_template}")
    if not os.path.isfile(alpha_mesh_template):
        raise FileNotFoundError(f"Missing alpha mesh template: {alpha_mesh_template}")
    if not os.path.isfile(tex_template):
        raise FileNotFoundError(f"Missing texture template: {tex_template}")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    psg_converter = glbtopsg.PSGConverterCLI()

    used_output_names = {
        os.path.splitext(entry)[0]
        for entry in os.listdir(output_dir)
        if entry.lower().endswith(".psg")
    }
    diffuse_registry: Dict[Tuple[str, str], Tuple[bytes, str]] = {}
    normal_registry: Dict[Tuple[str, str], Tuple[bytes, str]] = {}
    roughness_registry: Dict[Tuple[str, str], Tuple[bytes, str]] = {}
    lightmap_registry: Dict[Tuple[str, str], Tuple[bytes, str]] = {}

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".glb"):
            continue
        glb_path = os.path.join(input_dir, fname)
        base_name = os.path.splitext(fname)[0]

        mesh_stem = _generate_random_psg_stem(used_output_names)
        mesh_out = os.path.join(output_dir, f"{mesh_stem}.psg")

        base_bytes: Optional[bytes] = None
        base_mime: Optional[str] = None
        normal_bytes: Optional[bytes] = None
        normal_mime: Optional[str] = None
        rough_bytes: Optional[bytes] = None
        rough_mime: Optional[str] = None
        emission_bytes: Optional[bytes] = None
        emission_mime: Optional[str] = None
        emission_image_name: Optional[str] = None
        base_image_name: Optional[str] = None
        normal_image_name: Optional[str] = None
        rough_image_name: Optional[str] = None
        alpha_from_material = False

        gltf_data = _parse_glb_container(glb_path)
        if gltf_data is not None:
            gltf, bin_chunk = gltf_data
            base_idx, normal_idx, rough_idx, emission_idx, alpha_from_material = _extract_material_texture_indices(
                gltf
            )
            base_image_index = _resolve_texture_image_index(gltf, base_idx)
            base_image_name = _get_image_name(gltf, base_image_index)
            base_image = _load_image_from_gltf(
                glb_path,
                gltf,
                bin_chunk,
                base_image_index,
            )
            if base_image is not None:
                base_mime, base_bytes = base_image
            normal_image_index = _resolve_texture_image_index(gltf, normal_idx)
            normal_image_name = _get_image_name(gltf, normal_image_index)
            normal_image = _load_image_from_gltf(
                glb_path,
                gltf,
                bin_chunk,
                normal_image_index,
            )
            if normal_image is not None:
                normal_mime, normal_bytes = normal_image
            rough_image_index = _resolve_texture_image_index(gltf, rough_idx)
            rough_image_name = _get_image_name(gltf, rough_image_index)
            rough_image = _load_image_from_gltf(
                glb_path,
                gltf,
                bin_chunk,
                rough_image_index,
            )
            if rough_image is not None:
                rough_mime, rough_bytes = rough_image
            emission_image_index = _resolve_texture_image_index(gltf, emission_idx)
            emission_image_name = _get_image_name(gltf, emission_image_index)
            emission_image = _load_image_from_gltf(
                glb_path, gltf, bin_chunk, emission_image_index
            )
            if emission_image is not None:
                emission_mime, emission_bytes = emission_image

        manual_image = _find_manual_texture(input_dir, base_name)

        cleanup_dirs = []
        texture_has_alpha = False
        if manual_image:
            ext = os.path.splitext(manual_image)[1].lower()
            if ext == ".dds":
                try:
                    _, _, fmt_code, _, _, _ = _parse_dds_header(manual_image)
                    texture_has_alpha = _dds_has_alpha(fmt_code)
                except Exception:
                    texture_has_alpha = False
            else:
                texture_has_alpha = _detect_alpha_pillow(manual_image)
        elif base_bytes is not None:
            texture_has_alpha = _detect_alpha_bytes(base_bytes)

        base_registry_key = (
            _build_image_registry_key(base_image_name, base_bytes, "diffuse")
            if manual_image is None
            else None
        )
        diffuse_entry = diffuse_registry.get(base_registry_key) if base_registry_key else None
        if diffuse_entry is not None:
            diffuse_id_bytes, diffuse_tex_name = diffuse_entry
            diffuse_tex_out = os.path.join(output_dir, diffuse_tex_name)
            diffuse_needs_creation = False
        else:
            diffuse_id_bytes = random.getrandbits(64).to_bytes(8, "big")
            diffuse_tex_stem = _generate_random_psg_stem(used_output_names)
            diffuse_tex_name = f"{diffuse_tex_stem}.psg"
            diffuse_tex_out = os.path.join(output_dir, diffuse_tex_name)
            diffuse_needs_creation = True

        image_path: Optional[str] = None
        try:
            if diffuse_needs_creation:
                if manual_image:
                    image_path = manual_image
                elif base_bytes is not None:
                    ext_map = {
                        "image/png": ".png",
                        "image/jpeg": ".jpg",
                        "image/jpg": ".jpg",
                        "image/vnd-ms.dds": ".dds",
                    }
                    suffix = ext_map.get((base_mime or "").lower(), ".bin")
                    image_path, temp_dir = _bytes_to_temp_file(base_bytes, suffix)
                    cleanup_dirs.append(temp_dir)
                else:
                    image_path = None

            mesh_use_alpha = alpha_from_material

            normal_id_bytes: Optional[bytes] = None
            normal_tex_out: Optional[str] = None
            normal_tex_name: Optional[str] = None
            normal_needs_creation = False
            normal_key = _build_image_registry_key(normal_image_name, normal_bytes, "normal")
            normal_entry = normal_registry.get(normal_key) if normal_key else None
            if normal_entry is not None:
                normal_id_bytes, stored_name = normal_entry
                normal_tex_name = stored_name
                normal_tex_out = os.path.join(output_dir, stored_name)
            elif normal_bytes is not None:
                normal_id_bytes = random.getrandbits(64).to_bytes(8, "big")
                normal_tex_stem = _generate_random_psg_stem(used_output_names)
                normal_tex_name = f"{normal_tex_stem}.psg"
                normal_tex_out = os.path.join(output_dir, normal_tex_name)
                normal_needs_creation = True

            roughness_id_bytes: Optional[bytes] = None
            roughness_tex_out: Optional[str] = None
            rough_tex_name: Optional[str] = None
            roughness_needs_creation = False
            rough_key = _build_image_registry_key(rough_image_name, rough_bytes, "roughness")
            rough_entry = roughness_registry.get(rough_key) if rough_key else None
            if rough_entry is not None:
                roughness_id_bytes, stored_name = rough_entry
                rough_tex_name = stored_name
                roughness_tex_out = os.path.join(output_dir, stored_name)
            elif rough_bytes is not None:
                roughness_id_bytes = random.getrandbits(64).to_bytes(8, "big")
                rough_tex_stem = _generate_random_psg_stem(used_output_names)
                rough_tex_name = f"{rough_tex_stem}.psg"
                roughness_tex_out = os.path.join(output_dir, rough_tex_name)
                roughness_needs_creation = True

            mesh_template_path = alpha_mesh_template if mesh_use_alpha else mesh_template

            try:
                randomized_template_path, template_tmp_dir = _randomized_template_copy(
                    mesh_template_path
                )
                cleanup_dirs.append(template_tmp_dir)
            except Exception as exc:
                _log(f"Error preparing template for {fname}: {exc}")
                continue

            try:
                psg_converter.run_conversion(
                    gltf_path=glb_path,
                    bin_path=None,
                    psg_template_path=randomized_template_path,
                    output_path=mesh_out,
                    scale_xyz=256.0,
                )
            except Exception as exc:
                _log(f"Error converting {fname}: {exc}")
                continue

            if diffuse_needs_creation:
                try:
                    _create_texture_psg(
                        tex_template,
                        image_path,
                        diffuse_tex_out,
                        mesh_use_alpha,
                        diffuse_id_bytes,
                    )
                    if base_registry_key:
                        diffuse_registry[base_registry_key] = (diffuse_id_bytes, diffuse_tex_name)
                except Exception as exc:
                    _log(f"Error creating texture for {fname}: {exc}")
                    continue

            if normal_needs_creation and normal_tex_out and normal_id_bytes:
                try:
                    ext_map = {
                        "image/png": ".png",
                        "image/jpeg": ".jpg",
                        "image/jpg": ".jpg",
                        "image/vnd-ms.dds": ".dds",
                    }
                    suffix = ext_map.get((normal_mime or "").lower(), ".bin")
                    if normal_bytes is None:
                        raise ValueError("Missing normal bytes during texture creation")
                    normal_image_path, temp_dir = _bytes_to_temp_file(normal_bytes, suffix)
                    cleanup_dirs.append(temp_dir)
                    _create_texture_psg(tex_template, normal_image_path, normal_tex_out, False, normal_id_bytes)
                    if normal_key:
                        normal_registry[normal_key] = (
                            normal_id_bytes,
                            normal_tex_name or os.path.basename(normal_tex_out),
                        )
                except Exception as exc:
                    _log(f"Error creating normal texture for {fname}: {exc}")
                    normal_id_bytes = None
                    normal_tex_out = None
                    normal_tex_name = None

            if roughness_needs_creation and roughness_tex_out and roughness_id_bytes:
                try:
                    ext_map = {
                        "image/png": ".png",
                        "image/jpeg": ".jpg",
                        "image/jpg": ".jpg",
                        "image/vnd-ms.dds": ".dds",
                    }
                    suffix = ext_map.get((rough_mime or "").lower(), ".bin")
                    if rough_bytes is None:
                        raise ValueError("Missing roughness bytes during texture creation")
                    roughness_image_path, temp_dir = _bytes_to_temp_file(rough_bytes, suffix)
                    cleanup_dirs.append(temp_dir)
                    processed_path, tmp_dir = _extract_green_channel_texture(roughness_image_path)
                    if tmp_dir:
                        cleanup_dirs.append(tmp_dir)
                    _create_texture_psg(tex_template, processed_path, roughness_tex_out, False, roughness_id_bytes)
                    if rough_key:
                        roughness_registry[rough_key] = (
                            roughness_id_bytes,
                            rough_tex_name or os.path.basename(roughness_tex_out),
                        )
                except Exception as exc:
                    _log(f"Error creating roughness texture for {fname}: {exc}")
                    roughness_id_bytes = None
                    roughness_tex_out = None
                    rough_tex_name = None

            lightmap_id_bytes: Optional[bytes] = None
            lightmap_tex_name: Optional[str] = None
            if emission_image_name:
                lightmap_key = _build_image_registry_key(
                    emission_image_name, emission_bytes, "lightmap"
                )
                registry_entry = lightmap_registry.get(lightmap_key) if lightmap_key else None
                if registry_entry is not None:
                    lightmap_id_bytes, lightmap_tex_name = registry_entry
                elif emission_bytes is not None:
                    lightmap_id_bytes = random.getrandbits(64).to_bytes(8, "big")
                    ext_map = {
                        "image/png": ".png",
                        "image/jpeg": ".jpg",
                        "image/jpg": ".jpg",
                        "image/vnd-ms.dds": ".dds",
                    }
                    suffix = ext_map.get((emission_mime or "").lower(), ".bin")
                    lightmap_image_path, temp_dir = _bytes_to_temp_file(emission_bytes, suffix)
                    cleanup_dirs.append(temp_dir)
                    lightmap_use_alpha = _detect_alpha_bytes(emission_bytes)
                    lightmap_tex_stem = _generate_random_psg_stem(used_output_names)
                    lightmap_tex_out = os.path.join(output_dir, f"{lightmap_tex_stem}.psg")
                    try:
                        _create_texture_psg(
                            tex_template,
                            lightmap_image_path,
                            lightmap_tex_out,
                            lightmap_use_alpha,
                            lightmap_id_bytes,
                        )
                        lightmap_tex_name = os.path.basename(lightmap_tex_out)
                        if lightmap_key:
                            lightmap_registry[lightmap_key] = (
                                lightmap_id_bytes,
                                lightmap_tex_name,
                            )
                    except Exception as exc:
                        _log(f"Error creating lightmap texture for {fname}: {exc}")
                        lightmap_id_bytes = None
                        lightmap_tex_name = None

            try:
                _patch_mesh_psg(
                    mesh_out,
                    diffuse_id_bytes,
                    mesh_use_alpha,
                    normal_id_bytes=normal_id_bytes,
                    roughness_id_bytes=roughness_id_bytes,
                    lightmap_id_bytes=lightmap_id_bytes,
                )
            except Exception as exc:
                _log(f"Error patching mesh {mesh_out}: {exc}")
                continue

            textures_exported = [diffuse_tex_name]
            if normal_tex_name:
                textures_exported.append(normal_tex_name)
            if rough_tex_name:
                textures_exported.append(rough_tex_name)
            if lightmap_tex_name:
                textures_exported.append(lightmap_tex_name)

            alpha_note = " with alpha" if mesh_use_alpha else ""
            _log(
                f"Converted {fname} -> {os.path.basename(mesh_out)}, "
                f"{', '.join(textures_exported)}{alpha_note}"
            )
        finally:
            for temp_dir in cleanup_dirs:
                if temp_dir and os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Bounds/tile helpers using glb_bounds.json
# ---------------------------------------------------------------------------


def _load_bounds_json(pres_dir: str, log: Callable[[str], None]) -> Dict[str, Tuple[float, float, float, float]]:
    """Load min/max XY bounds per GLB (by stem) from glb_bounds.json."""
    json_path = os.path.join(pres_dir, "glb_bounds.json")
    if not os.path.isfile(json_path):
        log(f"No glb_bounds.json found in Pres ({json_path}); chunk folders will not be created.")
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        log(f"Failed to read glb_bounds.json: {exc}")
        return {}

    result: Dict[str, Tuple[float, float, float, float]] = {}
    for key, val in data.items():
        try:
            vmin = val["min"]
            vmax = val["max"]
            min_x = float(vmin[0])
            min_y = float(vmin[1])
            max_x = float(vmax[0])
            max_y = float(vmax[1])
            result[str(key)] = (min_x, max_x, min_y, max_y)
        except Exception:
            continue
    return result


def _get_tile_centers_from_bounds(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    chunk_size: float = 100.0,
) -> Set[Tuple[int, int]]:
    """Return a set of (cx, cy) tile centres overlapped by the given bounds."""
    size = float(chunk_size)
    eps = 1e-4

    start_kx = math.floor(min_x / size)
    end_kx = math.floor((max_x - eps) / size)
    start_ky = math.floor(min_y / size)
    end_ky = math.floor((max_y - eps) / size)

    tiles: Set[Tuple[int, int]] = set()
    for kx in range(start_kx, end_kx + 1):
        for ky in range(start_ky, end_ky + 1):
            cx = int(round((kx + 0.5) * size))
            cy = int(round((ky + 0.5) * size))
            tiles.add((cx, cy))
    return tiles


# ---------------------------------------------------------------------------
# Stream File Tool helpers
# ---------------------------------------------------------------------------


def _pack_and_cleanup(map_root: str, donotremove_dir: str, log: Callable[[str], None]) -> None:
    """Pack cSim* and cPres* folders with Stream File Tool.exe and delete them.

    Command line shape (for each group):
        Stream File Tool.exe pack "c:/yourfolder/cPres_Global" "c:/yourfolder/cPres_50_50_high" --platform=p

    We:
      - collect cPres* and cSim* dirs inside the DIST_ map root
      - order with *_Global first, then others sorted by name
      - run:
            Stream File Tool.exe pack <all cPres dirs...> --platform=p
            Stream File Tool.exe pack <all cSim dirs...>  --platform=p
      - finally delete all cPres*, cSim*, cTex* folders inside the DIST_ folder.
    """
    exe_path = os.path.join(donotremove_dir, "Stream File Tool.exe")
    if not os.path.isfile(exe_path):
        log(f"Stream File Tool.exe not found at: {exe_path}, skipping PSF packing.")
        return

    try:
        pres_dirs: list[str] = []
        sim_dirs: list[str] = []

        # Collect cPres* and cSim* folders from the DIST_ map root
        for entry in os.scandir(map_root):
            if not entry.is_dir():
                continue
            name = os.path.basename(entry.path)
            if name.startswith("cPres"):
                pres_dirs.append(entry.path)
            elif name.startswith("cSim"):
                sim_dirs.append(entry.path)

        def order_dirs(dirs: Sequence[str], global_name: str) -> list[str]:
            globals_ = [d for d in dirs if os.path.basename(d) == global_name]
            others = [d for d in dirs if d not in globals_]
            return globals_ + sorted(others, key=lambda p: os.path.basename(p).lower())

        pres_dirs = order_dirs(pres_dirs, "cPres_Global")
        sim_dirs = order_dirs(sim_dirs, "cSim_Global")

        tool_cwd = os.path.dirname(exe_path)

        # Pack PRES first
        if pres_dirs:
            log("Running Stream File Tool.exe for Pres group:")
            for d in pres_dirs:
                log(f"  {d}")
            cmd_pres = [exe_path, "pack"] + pres_dirs + ["--platform=p"]
            subprocess.run(cmd_pres, cwd=tool_cwd, check=False)

        # Then pack SIM
        if sim_dirs:
            log("Running Stream File Tool.exe for Sim group:")
            for d in sim_dirs:
                log(f"  {d}")
            cmd_sim = [exe_path, "pack"] + sim_dirs + ["--platform=p"]
            subprocess.run(cmd_sim, cwd=tool_cwd, check=False)

    except Exception as exc:
        log(f"Stream File Tool.exe packing failed: {exc}")
    finally:
        # Clean up folders after packing
        log("Deleting cPres*, cSim*, and cTex* folders from DIST_ map folder...")
        for entry in os.scandir(map_root):
            if not entry.is_dir():
                continue
            name = os.path.basename(entry.path)
            if name.startswith("cPres") or name.startswith("cSim") or name.startswith("cTex"):
                try:
                    shutil.rmtree(entry.path, ignore_errors=True)
                    log(f"  Deleted {name}")
                except Exception as exc:
                    log(f"  Failed to delete {name}: {exc}")


# ---------------------------------------------------------------------------
# UI + high-level compile flow
# ---------------------------------------------------------------------------


def _run_compile(map_name_raw: str, log: Callable[[str], None]) -> None:
    """High-level COMPILE button handler."""
    script_dir = _get_base_dir()
    donotremove_dir = os.path.join(script_dir, "DONOTREMOVE")

    pres_dir = os.path.join(script_dir, "Pres")
    sim_source_dir = os.path.join(script_dir, "Sim")
    cpres_templates_dir = os.path.join(donotremove_dir, "cpres")
    ctex_dir = os.path.join(donotremove_dir, "ctex")

    if not os.path.isdir(pres_dir):
        raise FileNotFoundError(f"Pres folder not found: {pres_dir}")

    # Normalise map name
    name = map_name_raw.strip()
    if not name:
        raise ValueError("Map name cannot be empty")
    if not name.startswith("DIST_"):
        name = "DIST_" + name
    map_root = os.path.join(script_dir, name)
    os.makedirs(map_root, exist_ok=True)

    cpres_global = os.path.join(map_root, "cPres_Global")
    csim_global = os.path.join(map_root, "cSim_Global")
    os.makedirs(cpres_global, exist_ok=True)
    os.makedirs(csim_global, exist_ok=True)

    log(f"Map root: {map_root}")

    # Copy CTEX content into root of DIST_ folder, renaming to DIST_<MapName>_Tex.<ext>
    if os.path.isdir(ctex_dir):
        log("Copying CTEX files into map root with DIST_<MapName>_Tex naming...")
        map_base = os.path.basename(map_root)  # e.g. DIST_Test
        new_base = f"{map_base}_Tex"
        for fname in os.listdir(ctex_dir):
            src_file = os.path.join(ctex_dir, fname)
            if not os.path.isfile(src_file):
                continue
            _, ext = os.path.splitext(fname)
            dst_file = os.path.join(map_root, new_base + ext)
            try:
                shutil.copy2(src_file, dst_file)
                log(f"  Copied {fname} -> {os.path.basename(dst_file)}")
            except Exception as exc:
                log(f"  Failed to copy CTEX file {fname}: {exc}")
    else:
        log("No DONOTREMOVE/ctex folder found; skipping CTEX copy.")

    # Load bounds JSON once
    bounds_lookup = _load_bounds_json(pres_dir, log)

    log("Scanning GLBs in Pres for tile coordinates (from glb_bounds.json)...")

    tile_coords: Set[Tuple[int, int]] = set()
    glb_files: list[str] = []
    for entry in os.listdir(pres_dir):
        if entry.lower().endswith(".glb"):
            glb_path = os.path.join(pres_dir, entry)
            glb_files.append(glb_path)
            stem = os.path.splitext(entry)[0]
            bounds = bounds_lookup.get(stem)
            if bounds is None:
                log(f"  {entry}: no bounds in glb_bounds.json; skipping chunk coords.")
                continue
            min_x, max_x, min_y, max_y = bounds
            tiles = _get_tile_centers_from_bounds(min_x, max_x, min_y, max_y)
            if tiles:
                log(f"  {entry}: tiles {sorted(tiles)}")
            tile_coords.update(tiles)

    if not glb_files:
        log("No GLBs found in Pres. Nothing to convert.")

    # Convert GLBs -> PSGs in-place inside ./Pres
    if glb_files:
        log("Converting GLBs to PSGs...")
        convert_glb_directory(
            input_dir=pres_dir,
            output_dir=pres_dir,
            template_dir=donotremove_dir,
            glb_converter=os.path.join(script_dir, "glbtopsg.py"),
            log=log,
        )

    # Delete GLBs after conversion
    for glb_path in glb_files:
        try:
            os.remove(glb_path)
            log(f"Deleted GLB: {os.path.basename(glb_path)}")
        except Exception as exc:
            log(f"Failed to delete {os.path.basename(glb_path)}: {exc}")

    # Move PSGs from Pres -> cPres_Global
    log("Moving PSGs from Pres to cPres_Global...")
    for entry in os.listdir(pres_dir):
        if not entry.lower().endswith(".psg"):
            continue
        src = os.path.join(pres_dir, entry)
        dst = os.path.join(cpres_global, entry)
        try:
            shutil.move(src, dst)
            log(f"  Moved {entry} -> cPres_Global")
        except Exception as exc:
            log(f"  Failed to move {entry}: {exc}")

    # Copy any PSGs from ./Sim into cSim_Global
    if os.path.isdir(sim_source_dir):
        log("Copying Sim PSGs into cSim_Global...")
        for root, _dirs, files in os.walk(sim_source_dir):
            for f in files:
                if not f.lower().endswith(".psg"):
                    continue
                src = os.path.join(root, f)
                dst = os.path.join(csim_global, f)
                try:
                    shutil.move(src, dst)
                    log(f"  Copied {f} from Sim -> cSim_Global")
                except Exception as exc:
                    log(f"  Failed to copy Sim PSG {f}: {exc}")
    else:
        log("No ./Sim folder found; skipping Sim PSG copy.")

    # Copy static cPres templates from DONOTREMOVE/cpres into cPres_Global
    if os.path.isdir(cpres_templates_dir):
        log("Copying static cPres templates into cPres_Global...")
        for fname in os.listdir(cpres_templates_dir):
            src_file = os.path.join(cpres_templates_dir, fname)
            if not os.path.isfile(src_file):
                continue
            dst_file = os.path.join(cpres_global, fname)
            try:
                shutil.copy2(src_file, dst_file)
                log(f"  Copied template {fname}")
            except Exception as exc:
                log(f"  Failed to copy template {fname}: {exc}")
    else:
        log("No DONOTREMOVE/cpres folder found; skipping static templates.")

    # Create chunk folders based on JSON-derived tiles
    if tile_coords:
        log("Creating chunk folders from tile coordinates...")
        for cx, cy in sorted(tile_coords):
            pres_chunk = os.path.join(map_root, f"cPres_{cx}_{cy}_high")
            sim_chunk = os.path.join(map_root, f"cSim_{cx}_{cy}_high")
            os.makedirs(pres_chunk, exist_ok=True)
            os.makedirs(sim_chunk, exist_ok=True)
            log(f"  Created {os.path.basename(pres_chunk)} and {os.path.basename(sim_chunk)}")
    else:
        log("No tile coordinates could be derived from JSON; no chunk folders created.")

    # Delete glb_bounds.json now that we're done
    bounds_json_path = os.path.join(pres_dir, "glb_bounds.json")
    if os.path.isfile(bounds_json_path):
        try:
            os.remove(bounds_json_path)
            log("Deleted glb_bounds.json from Pres.")
        except Exception as exc:
            log(f"Failed to delete glb_bounds.json: {exc}")

    # Run Stream File Tool.exe to pack and then delete folders
    _pack_and_cleanup(map_root, donotremove_dir, log)

    log("Compile complete.")


def _start_ui() -> None:
    if tk is None:
        print("Tkinter not available; running headless convert_glb_directory().")
        convert_glb_directory()
        return

    root = tk.Tk()
    root.title("GLB -> PSG Compiler")

    main_frame = ttk.Frame(root, padding=10)
    main_frame.grid(row=0, column=0, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(2, weight=1)

    ttk.Label(main_frame, text="Map Name (must start with DIST_):").grid(
        row=0, column=0, sticky="w"
    )
    map_var = tk.StringVar(value="DIST_")
    map_entry = ttk.Entry(main_frame, textvariable=map_var)
    map_entry.grid(row=0, column=1, sticky="ew", padx=(5, 0))

    ttk.Label(main_frame, text="Log:").grid(row=1, column=0, sticky="w", pady=(10, 0))
    log_widget = scrolledtext.ScrolledText(main_frame, height=20, width=80, state="disabled")
    log_widget.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(2, 0))

    def log(msg: str) -> None:
        log_widget.configure(state="normal")
        log_widget.insert("end", msg + "\n")
        log_widget.see("end")
        log_widget.configure(state="disabled")
        root.update_idletasks()

    def on_compile() -> None:
        map_name_raw = map_var.get()
        log_widget.configure(state="normal")
        log_widget.delete("1.0", "end")
        log_widget.configure(state="disabled")

        try:
            _run_compile(map_name_raw, log)
            messagebox.showinfo("Done", "Compile finished successfully.")
        except Exception as exc:
            log(f"ERROR: {exc}")
            messagebox.showerror("Error", str(exc))

    compile_btn = ttk.Button(main_frame, text="COMPILE", command=on_compile)
    compile_btn.grid(row=3, column=0, columnspan=2, pady=(10, 0))

    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    _start_ui()
