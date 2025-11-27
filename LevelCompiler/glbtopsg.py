#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command-line glTF/GLB -> PSG converter (Skinned)
Parity with the provided PyQt UI version.

Adds world-space bounding box writing:
- Writes min(vec3) to 0xA60 and max(vec3) to 0xA70 (big-endian floats).
- Assumes in-game Y is up (no Y/Z swap).

Usage:
  python glbtopsg.py input.glb template.psg output.psg 256
  python glbtopsg.py input.gltf template.psg output.psg 256 --bin input.bin
"""

import sys
import os
import struct
import traceback
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

try:
    import numpy
    import pygltflib
except ImportError as e:
    print(f"Error: A required library is missing. -> {e}")
    print("Please install the required libraries using: pip install pygltflib numpy")
    sys.exit(1)


@dataclass
class VDElem:
    vertex_type: int
    num_components: int
    stream: int
    offset: int
    stride: int
    type: int
    class_id: int


@dataclass
class VertexLayout:
    stride: int = 0
    elements: List[VDElem] = field(default_factory=list)


class PsgTemplateParser:
    RW_GRAPHICS_VERTEXDESCRIPTOR = 0x000200E9
    RW_GRAPHICS_VERTEXBUFFER = 0x000200EA
    RW_GRAPHICS_INDEXBUFFER = 0x000200EB
    PEGASUS_OPTIMESHDATA = 0x00EB0023  # RenderOptimeshData

    def __init__(self, file_path: str):
        with open(file_path, 'rb') as f:
            self.data = f.read()

        self.vdes_offset = -1
        self.vertex_offset = -1
        self.face_offset = -1
        self.vbuff_dict_ptr = -1
        self.ibuff_dict_ptr = -1
        self.main_baseresource_size = 0x44
        self.graphics_baseresource_size = 0x6C
        self.vertex_buffer_size_offset = -1
        self.index_count_offset = -1
        self.optimesh_index_offset = -1

        # Donor skeleton info
        self.bone_names: List[str] = []
        self.bone_palette: List[int] = []

        self._parse_dictionary_and_skeleton()
        self.layout = self._parse_vdes()

    def _u16_be(self, offset: int) -> int:
        return struct.unpack('>H', self.data[offset:offset+2])[0]

    def _u32_be(self, offset: int) -> int:
        return struct.unpack('>I', self.data[offset:offset+4])[0]

    def _is_base_resource(self, type_id: int) -> bool:
        return 0x00010030 <= type_id <= 0x0001003F

    def _parse_dictionary_and_skeleton(self):
        try:
            num_entries = self._u32_be(0x20)
            dict_start = self._u32_be(0x30)
            main_base = self._u32_be(0x44)

            dict_entries = []
            for i in range(num_entries):
                entry_offset = dict_start + (i * 0x18)
                entry = {
                    "ptr": self._u32_be(entry_offset + 0x00),
                    "size": self._u32_be(entry_offset + 0x08),
                    "type_id": self._u32_be(entry_offset + 0x14),
                    "offset": entry_offset
                }
                dict_entries.append(entry)

            carrier_entry = self._find_carrier(dict_entries, main_base)
            if carrier_entry:
                self._parse_carrier(carrier_entry, main_base)
            else:
                self.bone_names = []
                self.bone_palette = []

            palette_entry = next((e for e in dict_entries if e["type_id"] == self.PEGASUS_OPTIMESHDATA), None)
            if palette_entry:
                self._parse_bone_palette(palette_entry, main_base)
            else:
                self.bone_palette = list(range(len(self.bone_names)))

            for entry in dict_entries:
                type_id = entry["type_id"]
                ptr = entry["ptr"]
                block_start = (main_base + ptr) if self._is_base_resource(type_id) else ptr

                if type_id == self.RW_GRAPHICS_VERTEXDESCRIPTOR and self.vdes_offset == -1:
                    self.vdes_offset = block_start

                elif type_id == self.RW_GRAPHICS_VERTEXBUFFER and self.vertex_offset == -1:
                    br_index = self._u32_be(block_start)
                    br_entry = dict_entries[br_index]
                    br_ptr = br_entry["ptr"]
                    br_type_id = br_entry["type_id"]
                    self.vertex_offset = (main_base + br_ptr) if self._is_base_resource(br_type_id) else br_ptr
                    self.vertex_buffer_size_offset = block_start + 8
                    self.vbuff_dict_ptr = br_entry["offset"]

                elif type_id == self.RW_GRAPHICS_INDEXBUFFER and self.face_offset == -1:
                    br_index = self._u32_be(block_start)
                    br_entry = dict_entries[br_index]
                    br_ptr = br_entry["ptr"]
                    br_type_id = br_entry["type_id"]
                    self.face_offset = (main_base + br_ptr) if self._is_base_resource(br_type_id) else br_ptr
                    self.index_count_offset = block_start + 8
                    self.ibuff_dict_ptr = br_entry["offset"]

                elif type_id == self.PEGASUS_OPTIMESHDATA and self.optimesh_index_offset == -1:
                    self.optimesh_index_offset = block_start + 0x64

            if self.vdes_offset == -1:
                raise ValueError("Could not find a Vertex Descriptor (0x000200E9) in the PSG template.")
            if self.vertex_offset == -1:
                raise ValueError("Could not find a Vertex Buffer (0x000200EA) in the PSG template.")
            if self.face_offset == -1:
                raise ValueError("Could not find an Index Buffer (0x000200EB) in the PSG template.")

        except (IndexError, struct.error) as e:
            raise ValueError(f"Failed to parse PSG dictionary. The template may be corrupt or invalid. Details: {e}")

    def _find_carrier(self, dict_entries, main_base) -> Optional[dict]:
        for entry in dict_entries:
            block_start = (main_base + entry["ptr"]) if self._is_base_resource(entry["type_id"]) else entry["ptr"]
            block_end = block_start + entry["size"]

            header_offset = block_start + 0x20
            if header_offset + 0x24 > len(self.data):
                continue

            bone_count = self._u16_be(header_offset + 0x14)
            if not (0 < bone_count <= 512):
                continue

            off_ibm = self._u32_be(header_offset + 0x00)
            off_tbl_idx = self._u32_be(header_offset + 0x08)

            ibm_abs = block_start + off_ibm
            idx_abs = block_start + off_tbl_idx

            if (ibm_abs + bone_count * 64 <= block_end) and (idx_abs + bone_count * 4 <= block_end):
                return entry
        return None

    def _parse_carrier(self, carrier_entry, main_base):
        block_start = (main_base + carrier_entry["ptr"]) if self._is_base_resource(carrier_entry["type_id"]) else carrier_entry["ptr"]
        header_offset = block_start + 0x20

        bone_count = self._u16_be(header_offset + 0x14)
        off_tbl_idx = self._u32_be(header_offset + 0x08)
        idx_abs = block_start + off_tbl_idx

        self.bone_names = []
        for i in range(bone_count):
            rel_offset = self._u32_be(idx_abs + 4 * i)
            name_offset = block_start + rel_offset
            end_offset = self.data.find(b'\x00', name_offset)
            name = self.data[name_offset:end_offset].decode('ascii', errors='ignore')
            self.bone_names.append(name)

    def _parse_bone_palette(self, palette_entry, main_base):
        block_start = (main_base + palette_entry["ptr"]) if self._is_base_resource(palette_entry["type_id"]) else palette_entry["ptr"]
        palette_offset = block_start + 0x6C

        self.bone_palette = []
        p = palette_offset
        while p + 1 < len(self.data):
            global_index = self._u16_be(p)
            if global_index == 0xFFFF or global_index >= len(self.bone_names):
                break
            self.bone_palette.append(global_index)
            p += 2

    def _parse_vdes(self) -> VertexLayout:
        header_offset = self.vdes_offset
        num_elements = struct.unpack('>H', self.data[header_offset + 10:header_offset + 12])[0]

        elements_offset = header_offset + 16
        parsed_elements: List[VDElem] = []
        strides = set()

        for i in range(num_elements):
            elem_offset = elements_offset + (i * 8)
            elem_data = self.data[elem_offset:elem_offset+8]
            e = VDElem(
                vertex_type=elem_data[0],
                num_components=elem_data[1],
                stream=elem_data[2],
                offset=elem_data[3],
                stride=struct.unpack('>H', elem_data[4:6])[0],
                type=elem_data[6],
                class_id=elem_data[7]
            )
            parsed_elements.append(e)
            if e.stride > 0:
                strides.add(e.stride)

        if not strides:
            raise ValueError("Vertex descriptor in template has no valid stride defined.")

        resolved_stride = max(strides)
        return VertexLayout(stride=resolved_stride, elements=parsed_elements)


class PSGConverterCLI:
    # ----------------- Transform helpers (instance methods) -----------------

    def _node_local_matrix(self, node: pygltflib.Node) -> numpy.ndarray:
        """Build a 4x4 row-major matrix from a node's TRS or matrix."""
        if node.matrix:
            # glTF stores column-major; transpose to row-major
            return numpy.array(node.matrix, dtype=numpy.float32).reshape(4, 4).T

        t = numpy.array(node.translation or [0.0, 0.0, 0.0], dtype=numpy.float32)
        s = numpy.array(node.scale or [1.0, 1.0, 1.0], dtype=numpy.float32)
        q = numpy.array(node.rotation or [0.0, 0.0, 0.0, 1.0], dtype=numpy.float32)  # x, y, z, w

        x, y, z, w = q
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        rot = numpy.array([
            [1.0 - 2.0*(yy + zz),     2.0*(xy - wz),         2.0*(xz + wy),       0.0],
            [    2.0*(xy + wz),   1.0 - 2.0*(xx + zz),       2.0*(yz - wx),       0.0],
            [    2.0*(xz - wy),       2.0*(yz + wx),     1.0 - 2.0*(xx + yy),     0.0],
            [0.0,                   0.0,                   0.0,                   1.0]
        ], dtype=numpy.float32)

        smat = numpy.diag([s[0], s[1], s[2], 1.0]).astype(numpy.float32)
        m = rot @ smat
        m[0:3, 3] = t
        return m

    def _world_matrix_for_mesh(self, gltf: pygltflib.GLTF2, mesh_index: int) -> numpy.ndarray:
        """Find the first node that references mesh_index and compute its full world matrix (with parents)."""
        parents: Dict[int, int] = {}
        for i, node in enumerate(gltf.nodes or []):
            for c in (node.children or []):
                parents[c] = i

        mesh_nodes = [i for i, n in enumerate(gltf.nodes or []) if n.mesh == mesh_index]
        if not mesh_nodes:
            return numpy.eye(4, dtype=numpy.float32)

        node_id = mesh_nodes[0]
        chain = []
        cur = node_id
        while cur is not None:
            chain.append(cur)
            cur = parents.get(cur, None)
        chain.reverse()

        world = numpy.eye(4, dtype=numpy.float32)
        for nid in chain:
            world = world @ self._node_local_matrix(gltf.nodes[nid])
        return world

    def _apply_world_transform(self, positions: numpy.ndarray, world: numpy.ndarray) -> numpy.ndarray:
        """positions: (N,3), world: 4x4 (row-major). Returns transformed (N,3)."""
        R = world[:3, :3]
        t = world[:3, 3]
        return positions @ R.T + t

    # ----------------- Existing skin/palette utilities -----------------

    @staticmethod
    def normalize_bone_name(name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        return ''.join(ch for ch in name if ch.isalnum()).lower()

    def remap_skin_to_donor_palette(self, gltf_joints, gltf_weights, glb_bone_map: Dict[int, str],
                                    donor_bone_names: List[str], donor_bone_palette: List[int]):
        donor_name_to_global_idx = {self.normalize_bone_name(name): i for i, name in enumerate(donor_bone_names)}

        global_idx_to_palette_idx: Dict[int, int] = {}
        for palette_idx, global_idx in enumerate(donor_bone_palette):
            if global_idx not in global_idx_to_palette_idx:
                global_idx_to_palette_idx[global_idx] = palette_idx

        gltf_to_palette_map: Dict[int, int] = {}
        for gltf_idx, gltf_name in glb_bone_map.items():
            norm_name = self.normalize_bone_name(gltf_name)
            global_idx = donor_name_to_global_idx.get(norm_name)
            palette_idx = global_idx_to_palette_idx.get(global_idx) if global_idx is not None else None
            if palette_idx is not None:
                gltf_to_palette_map[gltf_idx] = palette_idx

        final_palette_indices = []
        final_weights = []
        for indices, weights in zip(gltf_joints, gltf_weights):
            weight_by_palette_idx: Dict[int, float] = {}
            for i in range(4):
                w = float(weights[i])
                if w <= 1e-6:
                    continue
                gltf_joint_idx = int(indices[i])
                palette_idx = gltf_to_palette_map.get(gltf_joint_idx)
                if palette_idx is not None:
                    weight_by_palette_idx[palette_idx] = weight_by_palette_idx.get(palette_idx, 0.0) + w

            sorted_pairs = sorted(weight_by_palette_idx.items(), key=lambda x: x[1], reverse=True)[:4]
            palette_indices_per_vertex = [0, 0, 0, 0]
            weights_per_vertex = [0.0, 0.0, 0.0, 0.0]
            for i, (pal_idx, w) in enumerate(sorted_pairs):
                palette_indices_per_vertex[i] = int(pal_idx)
                weights_per_vertex[i] = float(w)

            total_weight = sum(weights_per_vertex)
            if total_weight > 1e-6:
                inv = 1.0 / total_weight
                weights_per_vertex = [w * inv for w in weights_per_vertex]
            else:
                palette_indices_per_vertex = [0, 0, 0, 0]
                weights_per_vertex = [1.0, 0.0, 0.0, 0.0]

            final_palette_indices.append(palette_indices_per_vertex)
            final_weights.append(weights_per_vertex)

        return numpy.array(final_palette_indices, dtype=numpy.uint8), numpy.array(final_weights, dtype=numpy.float32)

    @staticmethod
    def pack_normal_dec3n(n) -> bytes:
        nx, ny, nz = n
        nx = max(-1.0, min(1.0, float(nx)))
        ny = max(-1.0, min(1.0, float(ny)))
        nz = max(-1.0, min(1.0, float(nz)))
        ix = int(round(nx * 1023.0))
        iy = int(round(ny * 1023.0))
        iz = int(round(nz * 511.0))
        ix &= (1 << 11) - 1
        iy &= (1 << 11) - 1
        iz &= (1 << 10) - 1
        packed_val = (iz << 22) | (iy << 11) | ix
        return struct.pack('>I', packed_val)

    def make_vertex_bin_dynamic(self, vertices,
                                uvs0,
                                uvs1,
                                normals, tangents, binormals,
                                joints, weights,
                                layout: VertexLayout,
                                scale_xyz: float = 256.0) -> bytearray:
        output = bytearray()
        elem_map = {
            'XYZ': 0, 'WEIGHTS': 1, 'NORMAL': 2, 'VERTEXCOLOR': 3, 'SPECULAR': 4,
            'BONEINDICES': 7, 'TEX0': 8, 'TEX1': 9, 'TEX2': 10, 'TEX3': 11, 'TEX4': 12, 'TEX5': 13,
            'TANGENT': 14, 'BINORMAL': 15
        }
        is_skinned = joints is not None and weights is not None

        for i in range(len(vertices)):
            vertex_bytes = bytearray(layout.stride)
            for elem in layout.elements:
                packed_data = b''
                if elem.type == elem_map['XYZ']:
                    x_s, y_s, z_s = [max(-32768, min(32767, int(float(c) * scale_xyz))) for c in vertices[i]]
                    if elem.vertex_type in [0x01, 0x05]:
                        packed_data = struct.pack('>hhh', x_s, y_s, z_s)
                    elif elem.vertex_type == 0x02:
                        packed_data = struct.pack('>fff', float(vertices[i][0]), float(vertices[i][1]), float(vertices[i][2]))

                elif elem.type == elem_map['NORMAL']:
                    if elem.vertex_type == 0x06:
                        packed_data = self.pack_normal_dec3n(normals[i])

                elif elem.type == elem_map['TANGENT']:
                    if elem.vertex_type == 0x06:
                        packed_data = self.pack_normal_dec3n(tangents[i])

                elif elem.type == elem_map['BINORMAL']:
                    if elem.vertex_type == 0x06:
                        packed_data = self.pack_normal_dec3n(binormals[i])

                elif elem.type == elem_map['TEX0']:
                    u, v = uvs0[i]
                    if elem.vertex_type == 0x03:
                        packed_data = struct.pack('>ee', numpy.float16(u), numpy.float16(v))
                    elif elem.vertex_type in [0x01, 0x05]:
                        # PSG TEX0 in S1: game expects [-1,1]
                        u_n = (float(u) * 2.0) - 1.0
                        v_n = (float(v) * 2.0) - 1.0
                        u_s = int(round(max(-1.0, min(1.0, u_n)) * 32767.0))
                        v_s = int(round(max(-1.0, min(1.0, v_n)) * 32767.0))
                        packed_data = struct.pack('>hh', u_s, v_s)

                elif elem.type == elem_map['TEX1']:
                    if uvs1 is not None:
                        u, v = uvs1[i]
                    else:
                        u, v = uvs0[i]
                    u_n = float(u)
                    v_n = float(v)
                    if elem.vertex_type == 0x03:
                        packed_data = struct.pack('>ee', numpy.float16(u_n), numpy.float16(v_n))
                    elif elem.vertex_type in [0x01, 0x05]:
                        u_s = int(round(max(-1.0, min(1.0, u_n)) * 32767.0))
                        v_s = int(round(max(-1.0, min(1.0, v_n)) * 32767.0))
                        packed_data = struct.pack('>hh', u_s, v_s)

                elif elem.type == elem_map['WEIGHTS']:
                    w = weights[i] if is_skinned else [1.0, 0.0, 0.0, 0.0]
                    if elem.vertex_type == 0x02:
                        packed_data = struct.pack('>ffff', float(w[0]), float(w[1]), float(w[2]), float(w[3]))
                    elif elem.vertex_type in [0x04, 0x07]:
                        w_u8 = [int(round(max(0.0, min(1.0, float(c))) * 255.0)) for c in w]
                        packed_data = struct.pack('>BBBB', *w_u8[:4])

                elif elem.type == elem_map['BONEINDICES']:
                    j = joints[i] if is_skinned else [0, 0, 0, 0]
                    if elem.vertex_type in [0x04, 0x07]:
                        j_u8 = [int(max(0, min(255, int(v)))) for v in j]
                        packed_data = struct.pack('>BBBB', *j_u8[:4])

                elif elem.type == elem_map['VERTEXCOLOR']:
                    if elem.vertex_type in [0x04, 0x07]:
                        packed_data = struct.pack('>BBBB', 255, 255, 255, 255)

                elif elem.type == elem_map['SPECULAR']:
                    if elem.vertex_type in [0x04, 0x07]:
                        packed_data = struct.pack('>BBBB', 0, 0, 0, 255)

                if packed_data:
                    vertex_bytes[elem.offset:elem.offset + len(packed_data)] = packed_data
            output.extend(vertex_bytes)
        return output

    @staticmethod
    def make_face_bin(faces) -> bytearray:
        output = bytearray()
        for face in faces:
            for idx in face:
                output.extend(struct.pack('>H', int(idx)))
        return output

    # ----------------- Modified: instance method so we can use helpers -----------------

    def parse_gltf_to_data(self, gltf_path: str, bin_path: Optional[str]) -> Tuple[
        List[numpy.ndarray], List[numpy.ndarray], List[numpy.ndarray], List[numpy.ndarray],
        List[numpy.ndarray], List[numpy.ndarray], List[List[int]], Optional[numpy.ndarray],
        Optional[numpy.ndarray], Optional[Dict[int, str]], numpy.ndarray
    ]:
        gltf = pygltflib.GLTF2.load(gltf_path)

        blob = None
        if gltf_path.lower().endswith('.glb'):
            blob = gltf.binary_blob()
        elif bin_path and os.path.exists(bin_path):
            with open(bin_path, 'rb') as f:
                blob = f.read()

        if blob is None:
            raise ValueError("Could not load binary data.")
        if not gltf.meshes:
            raise ValueError("No meshes found in file.")

        mesh_index = 0
        primitive = gltf.meshes[mesh_index].primitives[0]

        def get_accessor_data(accessor_id):
            accessor = gltf.accessors[accessor_id]
            buffer_view = gltf.bufferViews[accessor.bufferView]
            offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
            dtype_map = {
                5120: numpy.int8,
                5121: numpy.uint8,
                5122: numpy.int16,
                5123: numpy.uint16,
                5125: numpy.uint32,
                5126: numpy.float32
            }
            dtype = dtype_map[accessor.componentType]
            num_components = {
                'SCALAR': 1,
                'VEC2': 2,
                'VEC3': 3,
                'VEC4': 4
            }[accessor.type]
            data = numpy.frombuffer(blob, dtype=dtype, count=accessor.count * num_components, offset=offset)
            return data.reshape(accessor.count, num_components) if num_components > 1 else data

        raw_vertices = get_accessor_data(primitive.attributes.POSITION)
        raw_normals = get_accessor_data(primitive.attributes.NORMAL)
        raw_uvs0 = get_accessor_data(primitive.attributes.TEXCOORD_0) if primitive.attributes.TEXCOORD_0 is not None else numpy.zeros((len(raw_vertices), 2), dtype=numpy.float32)
        raw_uvs1 = get_accessor_data(primitive.attributes.TEXCOORD_1) if getattr(primitive.attributes, "TEXCOORD_1", None) is not None else None

        indices = get_accessor_data(primitive.indices)
        faces_indices = indices.reshape(-1, 3)

        raw_joints, raw_weights, glb_bone_map = None, None, None

        # World transform from node hierarchy for this mesh:
        world_matrix = self._world_matrix_for_mesh(gltf, mesh_index)

        if primitive.attributes.JOINTS_0 is not None and primitive.attributes.WEIGHTS_0 is not None:
            raw_joints = get_accessor_data(primitive.attributes.JOINTS_0)
            raw_weights = get_accessor_data(primitive.attributes.WEIGHTS_0)

            weights_accessor = gltf.accessors[primitive.attributes.WEIGHTS_0]
            if weights_accessor.componentType == 5121:
                raw_weights = raw_weights.astype(numpy.float32) / 255.0
            elif weights_accessor.componentType == 5123:
                raw_weights = raw_weights.astype(numpy.float32) / 65535.0

            skin_index = None
            for node in gltf.nodes:
                if node.mesh == mesh_index and node.skin is not None:
                    skin_index = node.skin
                    break

            if skin_index is None:
                raise ValueError("Skinned mesh data found, but no node in the GLB uses this mesh with a skin.")

            if gltf.skins and len(gltf.skins) > skin_index:
                skin = gltf.skins[skin_index]
                glb_bone_map = {i: gltf.nodes[joint_index].name for i, joint_index in enumerate(skin.joints)}
            else:
                raise ValueError("Skinned data found, but no valid skin definition was found in the GLB.")

        # Tangent/binormal accumulate
        tangent_acc = numpy.zeros_like(raw_vertices)
        for i0, i1, i2 in faces_indices:
            p0, p1, p2 = raw_vertices[[i0, i1, i2]]
            uv0, uv1, uv2 = raw_uvs0[[i0, i1, i2]]
            edge1, edge2 = p1 - p0, p2 - p0
            delta_uv1, delta_uv2 = uv1 - uv0, uv2 - uv0
            f = delta_uv1[0] * delta_uv2[1] - delta_uv2[0] * delta_uv1[1]
            if abs(f) > 1e-6:
                r = 1.0 / f
                tangent = (edge1 * delta_uv2[1] - edge2 * delta_uv1[1]) * r
                tangent_acc[[i0, i1, i2]] += tangent

        t_ortho = tangent_acc - raw_normals * numpy.sum(tangent_acc * raw_normals, axis=1, keepdims=True)
        final_raw_tangents = t_ortho / (numpy.linalg.norm(t_ortho, axis=1, keepdims=True) + 1e-9)
        final_raw_binormals = numpy.cross(raw_normals, final_raw_tangents)

        final_data = {
            "vertices": [],
            "uvs0": [],
            "uvs1": [],
            "normals": [],
            "tangents": [],
            "binormals": [],
            "joints": [],
            "weights": []
        }
        is_skinned = raw_joints is not None

        for v_idx in indices:
            final_data["vertices"].append(raw_vertices[v_idx])
            final_data["normals"].append(raw_normals[v_idx])
            final_data["uvs0"].append(raw_uvs0[v_idx])
            if raw_uvs1 is not None:
                final_data["uvs1"].append(raw_uvs1[v_idx])
            else:
                final_data["uvs1"].append(raw_uvs0[v_idx])
            final_data["tangents"].append(final_raw_tangents[v_idx])
            final_data["binormals"].append(final_raw_binormals[v_idx])
            if is_skinned:
                final_data["joints"].append(raw_joints[v_idx])
                final_data["weights"].append(raw_weights[v_idx])

        final_faces = numpy.arange(len(indices)).reshape(-1, 3).tolist()
        joints_out = final_data["joints"] if is_skinned else None
        weights_out = final_data["weights"] if is_skinned else None

        return (
            final_data["vertices"],
            final_data["uvs0"],
            final_data["uvs1"],
            final_data["normals"],
            final_data["tangents"],
            final_data["binormals"],
            final_faces,
            joints_out,
            weights_out,
            glb_bone_map,
            world_matrix,  # return world transform for bounds
        )

    def run_conversion(self, gltf_path: str, bin_path: Optional[str],
                       psg_template_path: str, output_path: str, scale_xyz: float):
        if not os.path.isfile(gltf_path):
            raise FileNotFoundError(f"Input glTF/GLB not found: {gltf_path}")
        if not os.path.isfile(psg_template_path):
            raise FileNotFoundError(f"PSG template not found: {psg_template_path}")
        if gltf_path.lower().endswith('.gltf') and not bin_path:
            raise ValueError("A .bin file is required when using a .gltf file (pass with --bin).")

        template = PsgTemplateParser(psg_template_path)

        (
            final_vertices,
            final_uvs0,
            final_uvs1,
            final_normals,
            final_tangents,
            final_binormals,
            final_faces,
            final_joints,
            final_weights,
            glb_bone_map,
            world_matrix
        ) = self.parse_gltf_to_data(gltf_path, bin_path)

        is_skinned = final_joints is not None
        remapped_joints, remapped_weights = None, None
        if is_skinned:
            if not template.bone_names or not template.bone_palette:
                raise ValueError("Skinning data found in GLB, but no skeleton or palette was loaded from the donor PSG.")
            remapped_joints, remapped_weights = self.remap_skin_to_donor_palette(
                final_joints, final_weights, glb_bone_map, template.bone_names, template.bone_palette)

        vertex_data = self.make_vertex_bin_dynamic(
            final_vertices,
            final_uvs0,
            final_uvs1,
            final_normals,
            final_tangents,
            final_binormals,
            remapped_joints,
            remapped_weights,
            template.layout,
            scale_xyz=float(scale_xyz)
        )
        face_data = self.make_face_bin(final_faces)

        with open(psg_template_path, 'rb') as f:
            psg_data = bytearray(f.read())

        v_offset = template.vertex_offset

        original_file_end = struct.unpack(">I", psg_data[template.main_baseresource_size:template.main_baseresource_size+4])[0]
        psg_data = psg_data[0:original_file_end]

        psg_data[template.graphics_baseresource_size:template.graphics_baseresource_size+4] = struct.pack(">I", len(vertex_data) + len(face_data))
        psg_data[template.vertex_buffer_size_offset:template.vertex_buffer_size_offset+4] = struct.pack(">I", len(vertex_data))
        psg_data[template.index_count_offset:template.index_count_offset+4] = struct.pack(">I", len(final_faces) * 3)
        if template.optimesh_index_offset > 0:
            psg_data[template.optimesh_index_offset:template.optimesh_index_offset+4] = struct.pack(">I", len(final_faces) * 3)

        psg_data.extend(b'\x00' * (len(vertex_data) + len(face_data)))
        psg_data[v_offset:v_offset + len(vertex_data)] = vertex_data
        psg_data[template.vbuff_dict_ptr+8:template.vbuff_dict_ptr+12] = struct.pack(">I", len(vertex_data))

        new_f_offset = v_offset + len(vertex_data)
        psg_data[template.ibuff_dict_ptr:template.ibuff_dict_ptr+4] = struct.pack(">I", len(vertex_data))
        psg_data[template.ibuff_dict_ptr+8:template.ibuff_dict_ptr+12] = struct.pack(">I", len(face_data))
        psg_data[new_f_offset:new_f_offset + len(face_data)] = face_data

        # ----- NEW: compute and write world-space bounding box to 0xA60/0xA70 -----
        # Use the vertices actually referenced by indices (already ordered)
        verts_np = numpy.array(final_vertices, dtype=numpy.float32)  # (N,3) object-space
        verts_world = self._apply_world_transform(verts_np, world_matrix)  # world-space (Y is up)

        min_bounds = numpy.min(verts_world, axis=0)  # [minX, minY, minZ]
        max_bounds = numpy.max(verts_world, axis=0)  # [maxX, maxY, maxZ]

        # Debug (optional): print world-space bounds we are writing
        # print(f"[DEBUG] min_bounds (world): {min_bounds}")
        # print(f"[DEBUG] max_bounds (world): {max_bounds}")

        # --- Write same bounding box to all header regions ---
        bbox_offsets = [
            (0x900, 0x910),
            (0x9C0, 0x9D0),
            (0xA60, 0xA70),
        ]

        for min_off, max_off in bbox_offsets:
            for off in (min_off, max_off):
                if len(psg_data) < off + 12:
                    raise ValueError(f"PSG too small for bbox write at 0x{off:X}")
            psg_data[min_off:min_off + 12] = struct.pack(">fff", *min_bounds)
            psg_data[max_off:max_off + 12] = struct.pack(">fff", *max_bounds)
        # -----------------------------------------------------


        with open(output_path, 'wb') as f:
            f.write(psg_data)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Convert glTF/GLB to PSG using a donor PSG template (parity with UI).")
    p.add_argument("input", help="Input .glb or .gltf")
    p.add_argument("template", help="Donor PSG template")
    p.add_argument("output", help="Output PSG path")
    p.add_argument("scale", type=float, help="Vertex scale, e.g. 256.0")
    p.add_argument("--bin", dest="bin", default=None, help="External .bin for .gltf")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        PSGConverterCLI().run_conversion(
            gltf_path=args.input,
            bin_path=args.bin,
            psg_template_path=args.template,
            output_path=args.output,
            scale_xyz=args.scale
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
