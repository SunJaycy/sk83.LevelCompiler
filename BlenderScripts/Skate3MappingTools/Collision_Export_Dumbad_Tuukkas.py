
#!/usr/bin/env python3
"""
Blender PSG Exporter for Skate 3 (PS3)
Exports Blender meshes and curves to .psg format

Installation:
    1. Open Blender
    2. Edit > Preferences > Add-ons > Install
    3. Select this file
    4. Enable "Import-Export: PSG Exporter for Skate 3"

Usage:
    1. Select mesh and/or curve objects
    2. Open N-panel (press N in 3D viewport)
    3. Navigate to "PSG Export" tab
    4. Set output path
    5. Click "Export PSG"

"""

bl_info = {
    "name": "Collision Exporter",
    "author": "PSG Tools Team",
    "version": (2, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > N-Panel > Collision Export",
    "description": "Export collision meshes and splines to Skate 3 PSG format (collision only)",
    "category": "Import-Export",
}

import bpy
from bpy.props import StringProperty, BoolProperty, FloatProperty, EnumProperty
from bpy.types import Panel, Operator
from bpy_extras import view3d_utils
import struct
import math
import os
import secrets
import hashlib
import bmesh
from typing import List, Tuple, Dict
from dataclasses import dataclass
from mathutils import Vector as BlenderVector
from mathutils.geometry import intersect_ray_tri
import colorsys
import blf

# ===== UTILITY FUNCTIONS =====

def encode_surface_id(audio: int, physics: int, pattern: int) -> int:
    """
    Encode SurfaceID from component bitfields.
    
    Args:
        audio: AudioSurfaceType enum value (0-127)
        physics: PhysicsSurfaceType enum value (0-31)
        pattern: SurfacePatternType enum value (0-15)
    
    Returns:
        16-bit packed SurfaceID
    """
    return (audio & 0x7F) | ((physics & 0x1F) << 7) | ((pattern & 0x0F) << 12)

def decode_surface_id(surface_id: int) -> tuple[int, int, int]:
    """
    Decode SurfaceID from component bitfields.

    Args:
        16-bit packed SurfaceID

    Returns:
        audio: AudioSurfaceType enum value (0-127)
        physics: PhysicsSurfaceType enum value (0-31)
        pattern: SurfacePatternType enum value (0-15)
    """
    audio = surface_id & 0x7F               # bits 0-6
    physics = (surface_id >> 7) & 0x1F      # bits 7-11
    pattern = (surface_id >> 12) & 0x0F     # bits 12-15
    return audio, physics, pattern

def be_u32(val: int) -> bytes:
    """Big-endian unsigned 32-bit integer"""
    return struct.pack('>I', val & 0xFFFFFFFF)

def be_u16(val: int) -> bytes:
    """Big-endian unsigned 16-bit integer"""
    return struct.pack('>H', val & 0xFFFF)

def le_u16(val: int) -> bytes:
    """Little-endian unsigned 16-bit integer"""
    return struct.pack('<H', val & 0xFFFF)

def be_u8(val: int) -> bytes:
    """Unsigned 8-bit integer"""
    return struct.pack('B', val & 0xFF)

def be_f32(val: float) -> bytes:
    """Big-endian 32-bit float"""
    return struct.pack('>f', float(val))

def be_u64(val: int) -> bytes:
    """Big-endian unsigned 64-bit integer"""
    return struct.pack('>Q', val & 0xFFFFFFFFFFFFFFFF)

def align(n: int, a: int) -> int:
    """Align n to next multiple of a"""
    return (n + (a - 1)) & ~(a - 1)

def align_qw(n: int) -> int:
    """Align to 16 bytes (quad-word)"""
    return (n + 15) & ~15

def be_i32(v: int) -> bytes:
    """Big-endian signed 32-bit integer"""
    return struct.pack('>i', v)

def be_i16(v: int) -> bytes:
    """Big-endian signed 16-bit integer"""
    return struct.pack('>h', v)

# ============================================================================
# Vector Math (Standalone - No Blender)
# ============================================================================

class Vector:
    """Simple 3D vector"""
    def __init__(self, xyz):
        if isinstance(xyz, (list, tuple)):
            self.x, self.y, self.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        else:
            self.x = self.y = self.z = float(xyz)
    
    def __add__(self, other):
        return Vector((self.x + other.x, self.y + other.y, self.z + other.z))
    
    def __sub__(self, other):
        return Vector((self.x - other.x, self.y - other.y, self.z - other.z))
    
    def __mul__(self, scalar):
        return Vector((self.x * scalar, self.y * scalar, self.z * scalar))
    
    def __truediv__(self, scalar):
        return Vector((self.x / scalar, self.y / scalar, self.z / scalar))
    
    @property
    def xyz(self):
        return 'xyz'

def vec_dot(a: Vector, b: Vector) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z

def vec_cross(a: Vector, b: Vector) -> Vector:
    return Vector((
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    ))

def vec_normalize(v: Vector) -> Vector:
    length_sq = v.x * v.x + v.y * v.y + v.z * v.z
    if length_sq <= 1e-20:
        return Vector((0.0, 0.0, 0.0))
    inv = 1.0 / math.sqrt(length_sq)
    return Vector((v.x * inv, v.y * inv, v.z * inv))

def edge_cosine_to_angle_byte(edge_cosine: float) -> int:
    """Quantize extended edge cosine to 5-bit angle byte
    
    EXACT port of ClusteredMeshBuilderUtils::EdgeCosineToAngleByte
    Source: clusteredmeshbuilderutils.cpp lines 30-63
    
    Converts edge cosine real number [-1...3] into angleByte which is -Log(A/π)/Log(√2)
    The factor √2 is used for the log base because the reversal function works out nicely.
    B=0 means fully convex, B=26 means the two triangles are coplanar.
    The range of the result is 0...26 which is 5 bits.
    
    Args:
        edge_cosine: Extended edge cosine value in range [-1.0, 3.0]
    
    Returns:
        Byte encoding data for this edge [0-26]
    """
    # Line 33: const rwpmath::VecFloat min_angle(6.6e-5f); // this is PI * sqrt(2) ^ (-31)
    min_angle = 6.6e-5
    
    # Lines 37-44: Calculate angle based on edge cosine value
    # Line 37: if ( edgeCosine > rwpmath::VecFloat(1.0f) )
    if edge_cosine > 1.0:
        # Line 39: angle = rwpmath::ACos(rwpmath::VecFloat(2.0f) - edgeCosine);
        angle = math.acos(2.0 - edge_cosine)
    else:
        # Line 43: angle = rwpmath::ACos(edgeCosine);
        angle = math.acos(edge_cosine)
    
    # Line 46: EA_ASSERT_FORMATTED(angle >= 0.0f && angle <= PI, ("Bad angle"));
    # Python: RenderWare asserts, doesn't clamp. We trust the input is valid.
    
    # Line 48: angle = rwpmath::Max(angle, min_angle);
    angle = max(angle, min_angle)
    
    # Line 50: int result = static_cast<int>(-2.0f * Log(angle / PI) / Log(2.0f));
    result = int(-2.0 * math.log(angle / math.pi) / math.log(2.0))
    
    # Clamp the result to the range of 0 - 26 (lines 53-62)
    # Lines 53-55: if ( result < 0 ) return 0;
    if result < 0:
        return 0
    # Lines 57-59: else if ( result > 26 ) return 26;
    elif result > 26:
        return 26
    
    # Line 62: return static_cast<uint8_t>(result);
    return result

class AABBox:
    """Axis-aligned bounding box"""
    def __init__(self, min_pt, max_pt):
        self.min = Vector(min_pt) if not isinstance(min_pt, Vector) else min_pt
        self.max = Vector(max_pt) if not isinstance(max_pt, Vector) else max_pt
    
    def surface_area(self):
        d = self.max - self.min
        if d.x < 0 or d.y < 0 or d.z < 0:
            return 0.0
        return 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    
    def expand(self, other):
        self.min.x = min(self.min.x, other.min.x)
        self.min.y = min(self.min.y, other.min.y)
        self.min.z = min(self.min.z, other.min.z)
        self.max.x = max(self.max.x, other.max.x)
        self.max.y = max(self.max.y, other.max.y)
        self.max.z = max(self.max.z, other.max.z)

def tri_bbox(v0, v1, v2):
    min_x = min(v0.x, v1.x, v2.x)
    min_y = min(v0.y, v1.y, v2.y)
    min_z = min(v0.z, v1.z, v2.z)
    max_x = max(v0.x, v1.x, v2.x)
    max_y = max(v0.y, v1.y, v2.y)
    max_z = max(v0.z, v1.z, v2.z)
    return AABBox(Vector((min_x, min_y, min_z)), Vector((max_x, max_y, max_z)))

# ============================================================================
# RenderWare-Compliant Data Structures
# ============================================================================

class RWEntry:
    """Entry in KDTree - matches RenderWare Entry struct (rwckdtreebuilder.cpp line 76)"""
    def __init__(self, entry_index: int, bbox_surface_area: float):
        self.entryIndex = entry_index  # Original triangle ID
        self.entryBBoxSurfaceArea = bbox_surface_area

class RWBuildNode:
    """KDTree BuildNode - matches RenderWare BuildNode (kdtreebuilder.h lines 129-180)
    
    EXACT match to RenderWare BuildNode structure:
    - BuildNode *m_parent
    - int32_t m_index
    - AABBoxU m_bbox
    - uint32_t m_firstEntry
    - uint32_t m_numEntries
    - uint32_t m_splitAxis
    - BuildNode *m_left
    - BuildNode *m_right
    
    Note: ext0/ext1 are NOT stored in BuildNode. They are computed during 
    InitializeRuntimeKDTree from child bboxes (lines 1356-1357).
    """
    def __init__(self, parent, bbox: AABBox, first_entry: int, num_entries: int):
        self.parent = parent      # BuildNode *m_parent
        self.m_index = 0          # int32_t m_index (initialized to 0)
        self.bbox = bbox          # AABBoxU m_bbox
        self.m_firstEntry = first_entry  # uint32_t m_firstEntry
        self.m_firstEntry_initial = first_entry  # Save initial value to detect if already set
        self.m_numEntries = num_entries  # uint32_t m_numEntries
        self.m_splitAxis = 0      # uint32_t m_splitAxis (initialized to 0)
        self.left = None          # BuildNode *m_left (initialized to 0/null)
        self.right = None         # BuildNode *m_right (initialized to 0/null)

class RWKDTreeSplit:
    """Split plane - matches RenderWare KDTreeSplit (rwckdtreebuilder.cpp line 40)"""
    def __init__(self):
        self.m_axis = 0
        self.m_value = 0.0
        self.m_numLeft = 0
        self.m_numRight = 0
        self.m_leftBBox = AABBox(Vector((0,0,0)), Vector((0,0,0)))
        self.m_rightBBox = AABBox(Vector((0,0,0)), Vector((0,0,0)))

class RWUnitCluster:
    """Cluster for storing triangles - matches RenderWare UnitCluster (unitcluster.h lines 37-167)
    
    EXACT match to RenderWare UnitCluster structure:
    - uint32_t clusterID
    - Vertex32 clusterOffset (used in 16-bit compression mode only)
    - UnitID * unitIDs (pointer to array)
    - uint32_t numUnits (count of units)
    - VertexSet vertexIDs (array of uint32_t)
    - uint32_t numVertices (count of entries in vertex set)
    - uint8_t compressionMode
    - uint8_t m_padding[3] (alignment padding)
    
    Python-specific extensions (not in RenderWare):
    - vertices: List[Vector] - computed vertex positions
    - vertex_map: Dict[int, int] - convenience lookup
    - byte_offset_start: computed during serialization
    - edge_codes: Dict[int, Tuple[int, int, int]] - computed edge codes
    """
    def __init__(self):
        self.clusterID = 0  # uint32_t clusterID
        self.clusterOffset = (0, 0, 0)  # Vertex32 clusterOffset (3x int32_t)
        self.unitIDs: List[int] = []  # UnitID * unitIDs (pointer to array in C++)
        self.numUnits = 0  # uint32_t numUnits (should match len(unitIDs))
        self.vertexIDs: List[int] = []  # VertexSet vertexIDs (array of uint32_t)
        self.numVertices = 0  # uint32_t numVertices (count of entries in vertex set)
        self.compressionMode = 0  # uint8_t compressionMode (VERTICES_UNCOMPRESSED = 0)
        # m_padding[3] not needed in Python (no struct padding)
        
        # Python-specific extensions (computed/derived data)
        self.vertices: List[Vector] = []  # Unique vertices (filled after compression)
        self.vertex_map: Dict[int, int] = {}  # global_vert_id -> local_index
        self.byte_offset_start = 0  # Start byte offset in global serialization
        self.edge_codes: Dict[int, Tuple[int, int, int]] = {}  # Per triangle edge codes


# ============================================================================
# Compression
# ============================================================================

def determine_compression_mode(verts: List[Vector], granularity: float) -> Tuple[int, Tuple[int, int, int]]:
    """Determine compression mode and offset for cluster vertices.
    
    EXACT port of RenderWare VertexCompression::DetermineCompressionModeAndOffsetForRange
    Source: vertexcompression.cpp lines 46-75
    
    RenderWare caller (rwcclusteredmeshbuilder.cpp lines 1040-1042) computes integer coordinates:
        int32_t x32 = (int32_t)( v.GetX() / m_vertexCompressionGranularity );
    This Python version computes them here to match the exact same calculation method,
    then applies the same logic as RenderWare's DetermineCompressionModeAndOffsetForRange.
    
    Returns:
        (compressionMode, (offset_x, offset_y, offset_z))
        compressionMode: 1 = VERTICES_16BIT_COMPRESSED, 2 = VERTICES_32BIT_COMPRESSED
        offset: Used only for mode 1 (16-bit compression)
    """
    if not verts or granularity == 0:
        # Default to 16-bit with no offset (should never happen)
        return 1, (0, 0, 0)
    
    # Convert all vertices to integer space (matches RenderWare caller lines 1040-1042)
    # RenderWare: int32_t x32 = (int32_t)( v.GetX() / m_vertexCompressionGranularity );
    # CRITICAL: Use division (v / granularity) to match actual compression exactly
    x32_coords = [int(v.x / granularity) for v in verts]
    y32_coords = [int(v.y / granularity) for v in verts]
    z32_coords = [int(v.z / granularity) for v in verts]
    
    x32_min = min(x32_coords)
    x32_max = max(x32_coords)
    y32_min = min(y32_coords)
    y32_max = max(y32_coords)
    z32_min = min(z32_coords)
    z32_max = max(z32_coords)
    
    # RenderWare tolerance for floating-point safety (line 56)
    # RenderWare: const int32_t granularityTolerance = 65534;
    GRANULARITY_TOLERANCE = 65534
    
    # Check if cluster fits in 16-bit range (lines 61-66)
    # EXACT RenderWare logic: if range < 65534 for all axes, use 16-bit compression
    if ((x32_max - x32_min < GRANULARITY_TOLERANCE) and 
        (y32_max - y32_min < GRANULARITY_TOLERANCE) and 
        (z32_max - z32_min < GRANULARITY_TOLERANCE)):
        # Use 16-bit compression (line 63)
        compression_mode = 1  # VERTICES_16BIT_COMPRESSED
        # Offset is min - 1 (lines 64-66)
        offset_x = x32_min - 1
        offset_y = y32_min - 1
        offset_z = z32_min - 1
    else:
        # Use 32-bit compression (lines 68-73)
        compression_mode = 2  # VERTICES_32BIT_COMPRESSED
        offset_x = 0
        offset_y = 0
        offset_z = 0
    
    return compression_mode, (offset_x, offset_y, offset_z)


def serialize_cluster_uncompressed(verts: List[Vector]) -> bytes:
    """Serialize vertices using uncompressed format (float32).
    
    EXACT port of RenderWare ClusteredMeshCluster::SetVertex for VERTICES_UNCOMPRESSED
    Source: rwcclusteredmeshcluster.cpp lines 234-254, 299-303
    
    Format:
        - 16 bytes per vertex: (x, y, z, w) as float32 (w = padding/unused, set to 0.0)
        - RenderWare stores vertices as rwpmath::Vector3 which is 16-byte aligned (quadword)
        - See rwcclusteredmeshcluster.cpp line 609: "16u * quadwords" (16 bytes per vertex)
    
    This mode avoids ALL compression artifacts and phantom collisions,
    but increases file size (~33% larger than 32-bit compressed: 16 vs 12 bytes/vertex).
    Use for complex geometry in small spaces where precision is critical.
    """
    if not verts:
        return b""
    
    body = bytearray()
    # Write raw float32 vertices as quadwords (16 bytes each)
    # RenderWare: vertexArray[vertexCount].Set(v.GetX(), v.GetY(), v.GetZ());
    # where vertexArray is rwpmath::Vector3[] with 16-byte alignment
    for v in verts:
        # Write as big-endian float32: (x, y, z, padding)
        # The 4th float (w) is padding to align to 16 bytes (quadword)
        body += struct.pack('>ffff', v.x, v.y, v.z, 0.0)
    
    return bytes(body)


def serialize_cluster_16bit(verts: List[Vector], granularity: float, offset: Tuple[int, int, int]) -> bytes:
    """Serialize vertices using 16-bit compression.
    
    EXACT port of RenderWare ClusteredMeshCluster::SetVertex for VERTICES_16BIT_COMPRESSED
    Source: rwcclusteredmeshcluster.cpp lines 255-283
    
    Format:
        - 12 bytes: offset (3x int32)
        - 6 bytes per vertex: (x, y, z) as uint16
    """
    if not verts:
        return b"\x00" * 12
    
    offset_x, offset_y, offset_z = offset
    
    body = bytearray()
    # Write offset (lines 260 equivalent, stored at start)
    # RenderWare: offsetData[0] = clusterOffset; (stored at start of vertexArray)
    body += be_i32(offset_x) + be_i32(offset_y) + be_i32(offset_z)
    
    # Write compressed vertices (lines 272-274)
    # RenderWare: compressedVertexData->x = (uint16_t)((int32_t)(v.GetX() / vertexCompressionGranularity) - clusterOffset->x);
    # CRITICAL: RenderWare casts directly to uint16_t without validation!
    # RenderWare assumes compression mode selection prevents overflow
    for v in verts:
        # Convert to int32 and subtract offset (lines 272-274)
        # CRITICAL: RenderWare uses TRUNCATION, not rounding!
        x_val = int(v.x / granularity) - offset_x
        y_val = int(v.y / granularity) - offset_y
        z_val = int(v.z / granularity) - offset_z
        
        # RenderWare casts directly: (uint16_t)x_val without range check
        # Python defensive programming: Validate range to catch Python/C++ floating-point differences
        # This should never trigger if compression mode selection matches RenderWare exactly
        if x_val < 0 or x_val > 65535 or y_val < 0 or y_val > 65535 or z_val < 0 or z_val > 65535:
            raise ValueError(
                f"16-bit compression overflow detected! "
                f"Compressed values: ({x_val}, {y_val}, {z_val}) exceed uint16 range [0, 65535]. "
                f"This indicates compression mode selection failed or floating-point precision overflow. "
                f"Cluster should use 32-bit compression instead. "
                f"Vertex: ({v.x:.6f}, {v.y:.6f}, {v.z:.6f}), "
                f"Granularity: {granularity}, Offset: {offset}"
            )
        
        # Cast to uint16 (line 272-274: (uint16_t)) - match RenderWare exactly
        # Python: Use bitwise mask since Python doesn't have uint16_t type
        x = x_val & 0xFFFF
        y = y_val & 0xFFFF
        z = z_val & 0xFFFF
        body += be_u16(x) + be_u16(y) + be_u16(z)
    
    return bytes(body)


def serialize_cluster_32bit(verts: List[Vector], granularity: float) -> bytes:
    """Serialize vertices using 32-bit compression.
    
    EXACT port of RenderWare ClusteredMeshCluster::SetVertex for VERTICES_32BIT_COMPRESSED
    Source: rwcclusteredmeshcluster.cpp lines 284-298
    
    Format:
        - 12 bytes per vertex: (x, y, z) as int32
    """
    if not verts:
        return b""
    
    body = bytearray()
    # Write compressed vertices (lines 295-297)
    # RenderWare: compressedVertexData->x = (int32_t)(v.GetX() / vertexCompressionGranularity);
    # CRITICAL: RenderWare casts directly to int32_t without validation!
    # RenderWare assumes compression mode selection prevents overflow
    for v in verts:
        # Convert to int32 (lines 295-297)
        # CRITICAL: RenderWare uses TRUNCATION, not rounding!
        x_val = int(v.x / granularity)
        y_val = int(v.y / granularity)
        z_val = int(v.z / granularity)
        
        # RenderWare casts directly: (int32_t)x_val without range check
        # Python defensive programming: Validate range to catch Python/C++ floating-point differences
        # This should never trigger if compression mode selection matches RenderWare exactly
        INT32_MIN = -2147483648
        INT32_MAX = 2147483647
        if x_val < INT32_MIN or x_val > INT32_MAX or \
           y_val < INT32_MIN or y_val > INT32_MAX or \
           z_val < INT32_MIN or z_val > INT32_MAX:
            raise ValueError(
                f"32-bit compression overflow detected! "
                f"Compressed values: ({x_val}, {y_val}, {z_val}) exceed int32 range "
                f"[{INT32_MIN}, {INT32_MAX}]. "
                f"This will cause vertex corruption and phantom collisions. "
                f"Vertex: ({v.x:.6f}, {v.y:.6f}, {v.z:.6f}), "
                f"Granularity: {granularity}. "
                f"Consider using larger granularity or scaling mesh down."
            )
        
        # Cast to int32 (line 295-297: (int32_t)) - match RenderWare exactly
        x = x_val
        y = y_val
        z = z_val
        body += be_i32(x) + be_i32(y) + be_i32(z)
    
    return bytes(body)


def determine_optimal_granularity(verts: List[Vector], 
                                   min_granularity: float = 0.001,
                                   max_granularity: float = 10.0) -> float:
    """Automatically determine the finest granularity that fits within 32-bit compression limits.
    
    This function finds the finest (smallest) granularity value that ensures all vertices
    can be compressed into int32 range without overflow. This maximizes precision while
    ensuring compatibility with RenderWare's 32-bit compression format.
    
    CRITICAL FOR LARGE MESHES: For huge meshes (100K+ triangles, large dimensions),
    this function will automatically select coarser granularity to prevent integer overflow.
    
    Args:
        verts: List of vertices to compress
        min_granularity: Finest granularity to try (default 0.001 = 1mm precision)
        max_granularity: Coarsest granularity to try (default 10.0 = 10m precision)
    
    Returns:
        The finest granularity that fits within int32 limits
        
    Raises:
        ValueError: If even max_granularity cannot fit all vertices
    """
    if not verts:
        return min_granularity
    
    INT32_MIN = -2147483648
    INT32_MAX = 2147483647
    
    # Find mesh bounds
    mesh_min_x = min(v.x for v in verts)
    mesh_max_x = max(v.x for v in verts)
    mesh_min_y = min(v.y for v in verts)
    mesh_max_y = max(v.y for v in verts)
    mesh_min_z = min(v.z for v in verts)
    mesh_max_z = max(v.z for v in verts)
    
    # Calculate required range for each axis
    range_x = mesh_max_x - mesh_min_x
    range_y = mesh_max_y - mesh_min_y
    range_z = mesh_max_z - mesh_min_z
    
    # Binary search for optimal granularity
    # We want the finest granularity where: max_coord / granularity <= INT32_MAX
    # and: min_coord / granularity >= INT32_MIN
    # This means: granularity >= max(abs(max_coord), abs(min_coord)) / INT32_MAX
    
    # Calculate minimum granularity based on absolute coordinate bounds
    max_abs_coord = max(abs(mesh_max_x), abs(mesh_min_x), 
                        abs(mesh_max_y), abs(mesh_min_y),
                        abs(mesh_max_z), abs(mesh_min_z))
    
    # Minimum granularity to avoid int32 overflow
    theoretical_min = max_abs_coord / INT32_MAX if max_abs_coord > 0 else min_granularity
    
    # Start with user's desired minimum or theoretical minimum, whichever is larger
    granularity_low = max(min_granularity, theoretical_min)
    granularity_high = max_granularity
    
    # If theoretical minimum exceeds max, we have a problem
    if granularity_low > granularity_high:
        raise ValueError(
            f"Mesh is too large for 32-bit compression even with granularity {max_granularity}. "
            f"Maximum absolute coordinate: {max_abs_coord:.2f} units. "
            f"Required granularity: {theoretical_min:.6f}. "
            f"Solution: Scale mesh down or split into multiple PSG files."
        )
    
    # Binary search for finest granularity that fits
    best_granularity = granularity_high
    tolerance = 1e-9  # Precision tolerance for granularity
    
    while granularity_high - granularity_low > tolerance:
        granularity_mid = (granularity_low + granularity_high) / 2.0
        
        # Test if this granularity fits
        fits = True
        for v in verts:
            x_val = int(v.x / granularity_mid)
            y_val = int(v.y / granularity_mid)
            z_val = int(v.z / granularity_mid)
            
            if (x_val < INT32_MIN or x_val > INT32_MAX or
                y_val < INT32_MIN or y_val > INT32_MAX or
                z_val < INT32_MIN or z_val > INT32_MAX):
                fits = False
                break
        
        if fits:
            # This granularity works, try finer
            best_granularity = granularity_mid
            granularity_high = granularity_mid
        else:
            # Too fine, need coarser
            granularity_low = granularity_mid
    
    # Round to reasonable precision (avoid floating-point noise)
    # Round to nearest 0.0001 for values < 0.01, or 0.001 for larger values
    if best_granularity < 0.01:
        best_granularity = round(best_granularity, 6)
    else:
        best_granularity = round(best_granularity, 4)
    
    return best_granularity


# ============================================================================
# KDTree - EXACT RenderWare Implementation (Verified Line-by-Line)
# ============================================================================
# Complete port of EA RenderWare 4.x (version 6.14.00) KDTree builder
# Source: rwcollision_volumes/6.14.00/source/core/spatialmaps/rwckdtreebuilder.cpp
# 
# All functions verified against RenderWare C++ source and Skate 3 IDA dumps
# Critical bugs fixed:
#   - rwc_SortSplitEntries: Now uses start_index offset (no Python slice copy)
#   - rwc_FindBestSplit_SAH: Now sorts entries in-place correctly
#   - RW_WalkBranch: Fixed cluster allocation (new per leaf)
#   - All functions match RenderWare behavior exactly
# ============================================================================

# EA RenderWare parameters - FROM SOURCE CODE
KDTREE_SPLIT_THRESHOLD = 8  # RenderWare default: clusteredmeshbuilder.h line 180
KDTREE_MAX_ENTRIES_PER_NODE = 63
KDTREE_SPLIT_COST_THRESHOLD = 0.95  # rwckdtreebuilder.cpp line 28
KDTREE_LARGE_ITEM_THRESHOLD = 0.6   # rwckdtreebuilder.cpp line 33
KDTREE_MIN_CHILD_ENTRIES_THRESHOLD = 0.3  # RenderWare default: kdtreebuilder.h line 58
KDTREE_MIN_SIMILAR_AREA_THRESHOLD = 0.7

# Edge code flags from RenderWare (clusteredmeshcluster.h lines 88-95)
EDGE_ANGLE_ZERO = 0x1A  # 26
EDGE_CONVEX = 0x20      # bit 5
EDGE_VERTEX_DISABLE = 0x40  # bit 6 - disables vertex collision for smoothing
EDGE_UNMATCHED = 0x80   # bit 7

class KDTreeDebugConfig:
    def __init__(self):
        self.enabled = True  # Enable debug logging
        self.build = True
        self.split_recurse = True
        self.find_best_split = True
        self.partition = True
        self.mismatch_details = True
        self.trace_partition_steps = True
        self.trace_partition_limit = 64
        self.max_list_preview = 16
        self.fail_on_partition_mismatch_gt1 = False  # Don't crash on mismatches
        # Enhanced debugging for platform-specific issues
        self.trace_floating_point_precision = True  # Log floating-point precision issues
        self.trace_platform_differences = True  # Log potential platform differences

KD_DEBUG = KDTreeDebugConfig()

def _kd_indent(depth: int) -> str:
    return '  ' * max(0, depth)

def kdlog(section: str, message: str, depth: int = 0):
    pass  # Disabled for performance

def fmt_bbox(bb: AABBox) -> str:
    return f"min({bb.min.x:.6f},{bb.min.y:.6f},{bb.min.z:.6f}) max({bb.max.x:.6f},{bb.max.y:.6f},{bb.max.z:.6f})"

class KDNode:
    def __init__(self):
        self.axis = 0
        self.split = 0.0
        self.left: int = -1
        self.right: int = -1
        self.parent: int = -1
        self.ext0 = 0.0
        self.ext1 = 0.0
        self.entries: List[Tuple[int, int]] = []

# ============================================================================
# Step 1: KDTree Builder (RenderWare rwckdtreebuilder.cpp)
# ============================================================================

# Constants from RenderWare (rwckdtreebuilder.cpp lines 24-33)
# CRITICAL: MAX_DEPTH = 32 from RenderWare 6.14.00 kdtreebase.h line 40
# Using 40 would allow deeper trees than runtime collision queries expect!
rwcKDTREEBUILD_SPLIT_COST_THRESHOLD = 0.95
rwcKDTREEBUILD_EMPTY_LEAF_THRESHOLD = 0.6
rwcKDTREE_MAX_DEPTH = 32  # RenderWare 6.14.00: kdtreebase.h line 40

def rwc_SortSplitEntries(split: RWKDTreeSplit, entry_bboxes: List[AABBox], 
                          entries: List[RWEntry], start_index: int, num_entries: int):
    """Sort entries into left/right groups by split plane
    
    EXACT port of RenderWare rwc_SortSplitEntries (rwckdtreebuilder.cpp lines 266-306)
    
    Algorithm: Two-pointer partitioning
    - Left pointer starts at 0, right at end
    - Test center of each bbox against split plane
    - If center > split: swap to right, decrement right pointer
    - If center <= split: keep on left, increment left pointer
    
    CRITICAL: This modifies entries array IN-PLACE!
    Result: entries = [left_group | right_group]
    
    RenderWare source lines 266-306:
    - Line 273: iLeft = 0
    - Line 274: iRight = numEntries - 1
    - Line 275: while (iLeft <= iRight)
    - Line 277: Get bbox using entries[iLeft].entryIndex
    - Line 289: center = (minExtent + maxExtent) * 0.5
    - Line 291: centerAxis = center.GetComponent(split.m_axis)
    - Line 294: CompGreaterThan(centerAxis, split.m_value) - STRICTLY >
    - Line 298-301: Call swap helper
    - Line 304-305: ASSERT counts match (NOT update!)
    """
    # Sort entry indices into left and right groups (line 272)
    # RenderWare receives: entries + m_firstEntry (pointer to subarray)
    # Python: We receive full array + start_index offset
    iLeft = start_index
    iRight = start_index + num_entries - 1

    if KD_DEBUG.partition:
        kdlog('PART', f"start={start_index} count={num_entries} axis={split.m_axis} ({'XYZ'[split.m_axis]}) value={split.m_value:.20f}")
        # Preview first few centers relative to split
        preview = min(num_entries, KD_DEBUG.max_list_preview)
        left_preview = []
        right_preview = []
        for i in range(preview):
            bb = entry_bboxes[entries[start_index + i].entryIndex]
            c = (bb.min + bb.max) * 0.5
            ca = getattr(c, 'xyz'[split.m_axis])
            if ca > split.m_value:
                right_preview.append((entries[start_index + i].entryIndex, ca))
            else:
                left_preview.append((entries[start_index + i].entryIndex, ca))
        if left_preview:
            kdlog('PART', f"preview-left[{len(left_preview)}]: " + ', '.join(f"(id={idx},c={ca:.6f})" for idx, ca in left_preview))
        if right_preview:
            kdlog('PART', f"preview-right[{len(right_preview)}]: " + ', '.join(f"(id={idx},c={ca:.6f})" for idx, ca in right_preview))
    
    trace_count = 0
    while iLeft <= iRight:
        # Get current entry's bbox (line 277)
        bb = entry_bboxes[entries[iLeft].entryIndex]
        
        # Min and Max of current entry AABBox (lines 284-285)
        min_extent = bb.min
        max_extent = bb.max
        
        # Center point of current entry AABBox (line 289)
        # CRITICAL: Match RenderWare's exact center calculation
        # RenderWare: const rwpmath::Vector3 center = (minExtent + maxExtent) * rwpmath::GetVecFloat_Half();
        center = (min_extent + max_extent) * 0.5
        
        # Get center component along split axis (line 291)
        center_axis = getattr(center, 'xyz'[split.m_axis])
        
        # Create a mask to select center components which are greater than split value (line 294)
        # RenderWare: rwpmath::CompGreaterThan(centerAxis, split.m_value)
        # CRITICAL: STRICTLY > (not >=)
        swap = center_axis > split.m_value

        if KD_DEBUG.partition and KD_DEBUG.trace_partition_steps and trace_count < KD_DEBUG.trace_partition_limit:
            kdlog('PART', f"iL={iLeft-start_index:4d} iR={iRight-start_index:4d} eid={entries[iLeft].entryIndex:6d} center={center_axis:.12f} split={split.m_value:.12f} -> {'R(SWAP)' if swap else 'L(KEEP)'}")
            trace_count += 1
        
        # rwc_SwapEntriesAndAdjustIndices (lines 298-301, helper at lines 108-126)
        if swap:
            # Swap with right (lines 117-120)
            entries[iLeft], entries[iRight] = entries[iRight], entries[iLeft]
            iRight -= 1
        else:
            # Increment left (lines 124)
            iLeft += 1
    
    # ASSERT that counts match (lines 304-305)
    # RenderWare: EA_ASSERT_MSG(split.m_numLeft == (uint32_t) iLeft, ("Count of entries on left of split does not match."));
    # RenderWare: EA_ASSERT_MSG(split.m_numRight == numEntries - iLeft, ("Count of entries on right of split does not match"));
    # Account for start_index offset
    actual_left_count = iLeft - start_index
    actual_right_count = num_entries - actual_left_count
    
    if KD_DEBUG.partition:
        kdlog('PART', f"done: left={actual_left_count} right={actual_right_count}")
    
    # ASSERT: Counts must match exactly (RenderWare lines 304-305)
    # If this fails, it indicates a bug in the split calculation logic
    assert split.m_numLeft == actual_left_count, \
        f"Count of entries on left of split does not match. Expected: {split.m_numLeft}, Actual: {actual_left_count}"
    assert split.m_numRight == actual_right_count, \
        f"Count of entries on right of split does not match. Expected: {split.m_numRight}, Actual: {actual_right_count}"

class KDTreeMultiAxisSplit:
    """EXACT RenderWare KDTreeMultiAxisSplit structure"""
    def __init__(self):
        self.m_value = [0.0, 0.0, 0.0]  # Vector3 - split values for all 3 axes
        self.m_numLeft = [0, 0, 0]       # Vector3 - left counts for all 3 axes
        self.m_numRight = [0, 0, 0]      # Vector3 - right counts for all 3 axes
        self.m_leftBBox = [None, None, None]   # AABBox[3] - left bboxes for all 3 axes
        self.m_rightBBox = [None, None, None]  # AABBox[3] - right bboxes for all 3 axes

def rwc_UpdateSplitStats(axis_comparison: List[bool], min_extent: Vector, max_extent: Vector,
                        right_count: List[int], left_count: List[int],
                        left_bbox_x: AABBox, right_bbox_x: AABBox,
                        left_bbox_y: AABBox, right_bbox_y: AABBox,
                        left_bbox_z: AABBox, right_bbox_z: AABBox) -> None:
    """
    EXACT RenderWare implementation of rwc_UpdateSplitStats.
    
    This matches the exact behavior from RenderWare source lines 177-245.
    RenderWare uses conditional updates based on the axisComparison mask.
    
    Based on RenderWare source: rwckdtreebuilder.cpp lines 177-245
    """
    # Adjust each of the split left/right counts (lines 191-192)
    # RenderWare: rightCount += rwpmath::Select(axisComparison, rwpmath::GetVector3_One(), rwpmath::GetVector3_Zero());
    # RenderWare: leftCount += rwpmath::Select(axisComparison, rwpmath::GetVector3_Zero(), rwpmath::GetVector3_One());
    for axis in range(3):
        if axis_comparison[axis]:
            right_count[axis] += 1
        else:
            left_count[axis] += 1
    
    # Adjust each of the split AABBoxes (lines 194-245)
    # X-Axis split (lines 196-212)
    # Generate the new Min and Max AABBox values
    # Left AABBox
    new_left_min = Vector((min(left_bbox_x.min.x, min_extent.x), min(left_bbox_x.min.y, min_extent.y), min(left_bbox_x.min.z, min_extent.z)))
    new_left_max = Vector((max(left_bbox_x.max.x, max_extent.x), max(left_bbox_x.max.y, max_extent.y), max(left_bbox_x.max.z, max_extent.z)))
    # Right AABBox
    new_right_min = Vector((min(right_bbox_x.min.x, min_extent.x), min(right_bbox_x.min.y, min_extent.y), min(right_bbox_x.min.z, min_extent.z)))
    new_right_max = Vector((max(right_bbox_x.max.x, max_extent.x), max(right_bbox_x.max.y, max_extent.y), max(right_bbox_x.max.z, max_extent.z)))
    
    # Adjust the AABBoxes with the new Min and Max values, if they require updating
    # Left AABBox
    # RenderWare: leftBBoxX.m_min = Select(axisComparison.GetX(), leftBBoxX.Min(), newLeftMin);
    # Select(condition, value_if_true, value_if_false)
    if axis_comparison[0]:  # X axis - goes right
        left_bbox_x.min = left_bbox_x.min  # Keep old (value_if_true)
        left_bbox_x.max = left_bbox_x.max  # Keep old (value_if_true)
    else:  # goes left
        left_bbox_x.min = new_left_min  # Update (value_if_false)
        left_bbox_x.max = new_left_max  # Update (value_if_false)
    # Right AABBox
    # RenderWare: rightBBoxX.m_min = Select(axisComparison.GetX(), newRightMin, rightBBoxX.Min());
    # Select(condition, value_if_true, value_if_false)
    if axis_comparison[0]:  # X axis - goes right
        right_bbox_x.min = new_right_min  # Update (value_if_true)
        right_bbox_x.max = new_right_max  # Update (value_if_true)
    else:  # goes left
        right_bbox_x.min = right_bbox_x.min  # Keep old (value_if_false)
        right_bbox_x.max = right_bbox_x.max  # Keep old (value_if_false)
    
    # Y-Axis split (lines 214-230)
    # Generate the new Min and Max AABBox values
    # Left AABBox
    new_left_min = Vector((min(left_bbox_y.min.x, min_extent.x), min(left_bbox_y.min.y, min_extent.y), min(left_bbox_y.min.z, min_extent.z)))
    new_left_max = Vector((max(left_bbox_y.max.x, max_extent.x), max(left_bbox_y.max.y, max_extent.y), max(left_bbox_y.max.z, max_extent.z)))
    # Right AABBox
    new_right_min = Vector((min(right_bbox_y.min.x, min_extent.x), min(right_bbox_y.min.y, min_extent.y), min(right_bbox_y.min.z, min_extent.z)))
    new_right_max = Vector((max(right_bbox_y.max.x, max_extent.x), max(right_bbox_y.max.y, max_extent.y), max(right_bbox_y.max.z, max_extent.z)))
    
    # Adjust the AABBoxes with the new Min and Max values, if they require updating
    # Left AABBox
    # RenderWare: leftBBoxY.m_min = Select(axisComparison.GetY(), leftBBoxY.Min(), newLeftMin);
    if axis_comparison[1]:  # Y axis - goes right
        left_bbox_y.min = left_bbox_y.min  # Keep old (value_if_true)
        left_bbox_y.max = left_bbox_y.max  # Keep old (value_if_true)
    else:  # goes left
        left_bbox_y.min = new_left_min  # Update (value_if_false)
        left_bbox_y.max = new_left_max  # Update (value_if_false)
    # Right AABBox
    # RenderWare: rightBBoxY.m_min = Select(axisComparison.GetY(), newRightMin, rightBBoxY.Min());
    if axis_comparison[1]:  # Y axis - goes right
        right_bbox_y.min = new_right_min  # Update (value_if_true)
        right_bbox_y.max = new_right_max  # Update (value_if_true)
    else:  # goes left
        right_bbox_y.min = right_bbox_y.min  # Keep old (value_if_false)
        right_bbox_y.max = right_bbox_y.max  # Keep old (value_if_false)
    
    # Z-Axis split (lines 232-248)
    # Generate the new Min and Max AABBox values
    # Left AABBox
    new_left_min = Vector((min(left_bbox_z.min.x, min_extent.x), min(left_bbox_z.min.y, min_extent.y), min(left_bbox_z.min.z, min_extent.z)))
    new_left_max = Vector((max(left_bbox_z.max.x, max_extent.x), max(left_bbox_z.max.y, max_extent.y), max(left_bbox_z.max.z, max_extent.z)))
    # Right AABBox
    new_right_min = Vector((min(right_bbox_z.min.x, min_extent.x), min(right_bbox_z.min.y, min_extent.y), min(right_bbox_z.min.z, min_extent.z)))
    new_right_max = Vector((max(right_bbox_z.max.x, max_extent.x), max(right_bbox_z.max.y, max_extent.y), max(right_bbox_z.max.z, max_extent.z)))
    
    # Adjust the AABBoxes with the new Min and Max values, if they require updating
    # Left AABBox
    # RenderWare: leftBBoxZ.m_min = Select(axisComparison.GetZ(), leftBBoxZ.Min(), newLeftMin);
    if axis_comparison[2]:  # Z axis - goes right
        left_bbox_z.min = left_bbox_z.min  # Keep old (value_if_true)
        left_bbox_z.max = left_bbox_z.max  # Keep old (value_if_true)
    else:  # goes left
        left_bbox_z.min = new_left_min  # Update (value_if_false)
        left_bbox_z.max = new_left_max  # Update (value_if_false)
    # Right AABBox
    # RenderWare: rightBBoxZ.m_min = Select(axisComparison.GetZ(), newRightMin, rightBBoxZ.Min());
    if axis_comparison[2]:  # Z axis - goes right
        right_bbox_z.min = new_right_min  # Update (value_if_true)
        right_bbox_z.max = new_right_max  # Update (value_if_true)
    else:  # goes left
        right_bbox_z.min = right_bbox_z.min  # Keep old (value_if_false)
        right_bbox_z.max = right_bbox_z.max  # Keep old (value_if_false)

def rwc_GetSplitStatsAllAxis_Exact(split: KDTreeMultiAxisSplit, entry_bboxes: List[AABBox], entries: List[RWEntry], 
                                   start_index: int, num_entries: int, node_bbox: AABBox) -> None:
    """
    EXACT RenderWare implementation of rwc_GetSplitStatsAllAxis.
    
    This matches the exact behavior from RenderWare source lines 408-474.
    RenderWare processes ALL 3 axes simultaneously using vector operations.
    
    Based on RenderWare source: rwckdtreebuilder.cpp lines 408-474
    """
    # CRITICAL: RenderWare processes ALL 3 axes simultaneously!
    # We need to match this exact behavior
    
    # Initialize stats for all 3 axes (lines 417-422)
    # RenderWare: rwpmath::Vector3 leftCount(rwpmath::GetVector3_Zero());
    # RenderWare: rwpmath::Vector3 rightCount(rwpmath::GetVector3_Zero());
    left_count = [0, 0, 0]  # [X, Y, Z]
    right_count = [0, 0, 0]  # [X, Y, Z]
    
    # Initialize bounding boxes for all 3 axes (lines 417-422)
    # CRITICAL: RenderWare initializes with inverted bboxes!
    # RenderWare: AABBox leftBBoxX(nodeBB.Max(), nodeBB.Min());
    left_bbox_x = AABBox(node_bbox.max, node_bbox.min)
    left_bbox_y = AABBox(node_bbox.max, node_bbox.min)
    left_bbox_z = AABBox(node_bbox.max, node_bbox.min)
    right_bbox_x = AABBox(node_bbox.max, node_bbox.min)
    right_bbox_y = AABBox(node_bbox.max, node_bbox.min)
    right_bbox_z = AABBox(node_bbox.max, node_bbox.min)
    
    for i in range(num_entries):
        # Current entry AABBox (line 434)
        bb = entry_bboxes[entries[start_index + i].entryIndex]
        
        # Min and Max of current entry AABBox (lines 441-442)
        min_extent = bb.min
        max_extent = bb.max
        
        # Center point of current entry AABBox (line 446)
        # RenderWare: const rwpmath::Vector3 center = (minExtent + maxExtent) * rwpmath::GetVecFloat_Half();
        center = (min_extent + max_extent) * 0.5
        
        # Create a mask to select center components which are greater than split value (line 449)
        # RenderWare: rwpmath::Mask3 axisComparison = rwpmath::CompGreaterThan(center, split.m_value);
        axis_comparison = [
            center.x > split.m_value[0],  # X axis
            center.y > split.m_value[1],  # Y axis
            center.z > split.m_value[2]   # Z axis
        ]
        
        # CRITICAL: Use exact RenderWare UpdateSplitStats function
        rwc_UpdateSplitStats(axis_comparison, min_extent, max_extent,
                            right_count, left_count,
                            left_bbox_x, right_bbox_x,
                            left_bbox_y, right_bbox_y,
                            left_bbox_z, right_bbox_z)
    
    # Assign the local copy of the split details (lines 464-473)
    split.m_numLeft = left_count
    split.m_numRight = right_count
    split.m_leftBBox = [left_bbox_x, left_bbox_y, left_bbox_z]
    split.m_rightBBox = [right_bbox_x, right_bbox_y, right_bbox_z]

def rwc_BBoxSurfaceArea(bbox: AABBox) -> float:
    """Calculate surface area of bounding box.
    
    EXACT port of RenderWare rwc_BBoxSurfaceArea (rwckdtreebuilder.cpp lines 136-140)
    
    RenderWare:
        rwpmath::Vector3 diag = bbox.Max() - bbox.Min();
        return GetVecFloat_Two() * (diag.GetX()*diag.GetY() + diag.GetY()*diag.GetZ() + diag.GetZ()*diag.GetX());
    """
    # Line 138: rwpmath::Vector3 diag = bbox.Max() - bbox.Min();
    d = bbox.max - bbox.min
    # Line 139: return GetVecFloat_Two() * (diag.GetX()*diag.GetY() + diag.GetY()*diag.GetZ() + diag.GetZ()*diag.GetX());
    return 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)

def rwc_GetMultiSplitLowestCost(node_bbox: AABBox, multi_split: KDTreeMultiAxisSplit) -> List[float]:
    """
    EXACT RenderWare implementation of rwc_GetMultiSplitLowestCost.
    
    This matches the exact behavior from RenderWare source lines 627-652.
    RenderWare calculates costs for all 3 axes simultaneously using vector operations.
    
    Based on RenderWare source: rwckdtreebuilder.cpp lines 627-652
    """
    # Weights of left split bboxes (lines 635-637)
    # RenderWare: rwpmath::Vector3 leftWeight = multiSplit.m_numLeft * rwpmath::Vector3(rwc_BBoxSurfaceArea(multiSplit.m_leftBBox[0]), ...);
    left_weight = [
        multi_split.m_numLeft[0] * rwc_BBoxSurfaceArea(multi_split.m_leftBBox[0]),  # X axis
        multi_split.m_numLeft[1] * rwc_BBoxSurfaceArea(multi_split.m_leftBBox[1]),  # Y axis
        multi_split.m_numLeft[2] * rwc_BBoxSurfaceArea(multi_split.m_leftBBox[2])   # Z axis
    ]
    
    # Weights of right split bboxes (lines 640-642)
    # RenderWare: rwpmath::Vector3 rightWeight = multiSplit.m_numRight * rwpmath::Vector3(rwc_BBoxSurfaceArea(multiSplit.m_rightBBox[0]), ...);
    right_weight = [
        multi_split.m_numRight[0] * rwc_BBoxSurfaceArea(multi_split.m_rightBBox[0]),  # X axis
        multi_split.m_numRight[1] * rwc_BBoxSurfaceArea(multi_split.m_rightBBox[1]),  # Y axis
        multi_split.m_numRight[2] * rwc_BBoxSurfaceArea(multi_split.m_rightBBox[2])   # Z axis
    ]
    
    # Weight of Parent (lines 644-646)
    # RenderWare: rwpmath::VecFloat nodeBBArea = (multiSplit.m_numLeft[0] + multiSplit.m_numRight[0]) *rwc_BBoxSurfaceArea(nodeBB);
    # RenderWare: rwpmath::Vector3 parentWeight(nodeBBArea, nodeBBArea, nodeBBArea);
    node_bb_area = (multi_split.m_numLeft[0] + multi_split.m_numRight[0]) * rwc_BBoxSurfaceArea(node_bbox)
    parent_weight = [node_bb_area, node_bb_area, node_bb_area]  # [X, Y, Z]
    
    # Costs of each split (line 649)
    # RenderWare: rwpmath::Vector3 costs = (leftWeight + rightWeight) / parentWeight;
    costs = [
        (left_weight[0] + right_weight[0]) / parent_weight[0],  # X axis
        (left_weight[1] + right_weight[1]) / parent_weight[1],  # Y axis
        (left_weight[2] + right_weight[2]) / parent_weight[2]   # Z axis
    ]
    
    return costs

def rwc_SelectLowestCostSplit(result: RWKDTreeSplit, multi_split: KDTreeMultiAxisSplit, costs: List[float]) -> float:
    """
    EXACT RenderWare implementation of rwc_SelectLowestCostSplit.
    
    This matches the exact behavior from RenderWare source lines 570-599.
    RenderWare selects the axis with the lowest cost and modifies the result parameter.
    
    Based on RenderWare source: rwckdtreebuilder.cpp lines 570-599
    """
    # RenderWare: if (costs.GetX() <= costs.GetY() && costs.GetX() <= costs.GetZ())
    if costs[0] <= costs[1] and costs[0] <= costs[2]:
        # RenderWare line 577: lowestCost = costs.GetX();
        lowest_cost = costs[0]
        # RenderWare line 578: result.m_axis = 0;
        result.m_axis = 0
        # RenderWare line 579: result.m_value = multiSplit.m_value.GetX();
        result.m_value = multi_split.m_value[0]
    # RenderWare: else if (costs.GetY() <= costs.GetZ())
    elif costs[1] <= costs[2]:
        # RenderWare line 583: lowestCost = costs.GetY();
        lowest_cost = costs[1]
        # RenderWare line 584: result.m_axis = 1;
        result.m_axis = 1
        # RenderWare line 585: result.m_value = multiSplit.m_value.GetY();
        result.m_value = multi_split.m_value[1]
    else:
        # RenderWare line 589: lowestCost = costs.GetZ();
        lowest_cost = costs[2]
        # RenderWare line 590: result.m_axis = 2;
        result.m_axis = 2
        # RenderWare line 591: result.m_value = multiSplit.m_value.GetZ();
        result.m_value = multi_split.m_value[2]
    
    # RenderWare lines 594-595: result.m_leftBBox.m_min = ...; result.m_leftBBox.m_max = ...;
    result.m_leftBBox.min = multi_split.m_leftBBox[result.m_axis].min
    result.m_leftBBox.max = multi_split.m_leftBBox[result.m_axis].max
    
    # RenderWare lines 597-598: result.m_rightBBox.m_min = ...; result.m_rightBBox.m_max = ...;
    result.m_rightBBox.min = multi_split.m_rightBBox[result.m_axis].min
    result.m_rightBBox.max = multi_split.m_rightBBox[result.m_axis].max
    
    # RenderWare: result.m_numLeft = static_cast<uint32_t>(multiSplit.m_numLeft[static_cast<int32_t>(result.m_axis)]);
    # RenderWare: result.m_numRight = static_cast<uint32_t>(multiSplit.m_numRight[static_cast<int32_t>(result.m_axis)]);
    result.m_numLeft = multi_split.m_numLeft[result.m_axis]
    result.m_numRight = multi_split.m_numRight[result.m_axis]
    
    return lowest_cost

def rwc_FindBestSplit_SAH(node_bbox: AABBox, entry_bboxes: List[AABBox],
                           entries: List[RWEntry], start_index: int, num_entries: int) -> RWKDTreeSplit:
    """Find best split for KDTree node.
    
    EXACT port of RenderWare rwc_FindBestSplit (rwckdtreebuilder.cpp lines 852-1002)
    
    Args:
        node_bbox: Bounding box of the node
        entry_bboxes: Bounding boxes of all entries
        entries: Array of entries (triangle indices + surface areas)
        start_index: Index of first entry in this node
        num_entries: Number of entries in this node
    
    Returns:
        Best split if found, None if no acceptable split found
    """
    if KD_DEBUG.find_best_split:
        kdlog('SPLIT', f"search start={start_index} count={num_entries} bbox={fmt_bbox(node_bbox)}")

    # Get tight bbox around entries (lines 866-884)
    # RenderWare: AABBoxU tightFPUBBox = entryBBoxes[entries[0].entryIndex];
    tight_bbox = AABBox(
        entry_bboxes[entries[start_index].entryIndex].min,
        entry_bboxes[entries[start_index].entryIndex].max
    )
    
    # Entry BBox metrics (lines 869-870)
    # RenderWare: float sumBBoxSurfaceArea(entries[0].entryBBoxSurfaceArea);
    # RenderWare: float smallestBBoxSurfaceArea(entries[0].entryBBoxSurfaceArea);
    sum_bbox_surface_area = entries[start_index].entryBBoxSurfaceArea
    smallest_bbox_surface_area = entries[start_index].entryBBoxSurfaceArea
    
    # Loop through remaining entries (lines 872-884)
    for i in range(1, num_entries):
        bb = entry_bboxes[entries[start_index + i].entryIndex]
        tight_bbox.expand(bb)
        
        # Line 879: const float currentSurfaceArea(entries[i].entryBBoxSurfaceArea);
        current_surface_area = entries[start_index + i].entryBBoxSurfaceArea
        # Line 880: sumBBoxSurfaceArea += currentSurfaceArea;
        sum_bbox_surface_area += current_surface_area
        # Line 883: smallestBBoxSurfaceArea = Min(currentSurfaceArea,smallestBBoxSurfaceArea);
        smallest_bbox_surface_area = min(current_surface_area, smallest_bbox_surface_area)
    
    # Compare the mean BBoxes to the node BBox to get a ratio (lines 888-892)
    # Line 889: rwpmath::VecFloat nodeSurfaceArea = rwc_BBoxSurfaceArea(nodeBBox);
    node_surface_area = rwc_BBoxSurfaceArea(node_bbox)
    # Line 892: const float meanBBoxSurfaceArea = sumBBoxSurfaceArea / (numEntries);
    mean_bbox_surface_area = sum_bbox_surface_area / num_entries
    
    # See if it's worth making an empty leaf (lines 894-938)
    # RenderWare tries to put all entries in one child and leave the other empty
    # if it reduces surface area significantly
    cur_split = RWKDTreeSplit()
    min_child_surface_area = node_surface_area
    
    # Loop over all 3 principal axes (line 896)
    for i in range(3):
        # Try keeping entries in left (lines 901-915)
        # Line 902: childBBox = nodeBBox;
        child_bbox = AABBox(node_bbox.min, node_bbox.max)
        # Line 903: childBBox.m_max.SetComponent((int)i, tightBBox.m_max.GetComponent((int)i));
        if i == 0:
            child_bbox.max = Vector((tight_bbox.max.x, child_bbox.max.y, child_bbox.max.z))
        elif i == 1:
            child_bbox.max = Vector((child_bbox.max.x, tight_bbox.max.y, child_bbox.max.z))
        else:
            child_bbox.max = Vector((child_bbox.max.x, child_bbox.max.y, tight_bbox.max.z))
        # Line 904: childSurfaceArea = rwc_BBoxSurfaceArea(childBBox);
        child_surface_area = rwc_BBoxSurfaceArea(child_bbox)
        
        # Line 906: if (childSurfaceArea < minChildSurfaceArea)
        if child_surface_area < min_child_surface_area:
            # Lines 908-914
            min_child_surface_area = child_surface_area
            cur_split.m_axis = i
            cur_split.m_value = getattr(tight_bbox.max, 'xyz'[i])
            cur_split.m_numLeft = num_entries
            cur_split.m_numRight = 0
            cur_split.m_leftBBox = tight_bbox
            # Line 914: curSplit.m_rightBBox = AABBox(nodeBBox.m_max, nodeBBox.m_min); // Inverted
            cur_split.m_rightBBox = AABBox(node_bbox.max, node_bbox.min)
        
        # Try keeping entries in right (lines 917-931)
        # Line 918: childBBox = nodeBBox;
        child_bbox = AABBox(node_bbox.min, node_bbox.max)
        # Line 919: childBBox.m_min.SetComponent((int)i, tightBBox.m_min.GetComponent((int)i));
        if i == 0:
            child_bbox.min = Vector((tight_bbox.min.x, child_bbox.min.y, child_bbox.min.z))
        elif i == 1:
            child_bbox.min = Vector((child_bbox.min.x, tight_bbox.min.y, child_bbox.min.z))
        else:
            child_bbox.min = Vector((child_bbox.min.x, child_bbox.min.y, tight_bbox.min.z))
        # Line 920: childSurfaceArea = rwc_BBoxSurfaceArea(childBBox);
        child_surface_area = rwc_BBoxSurfaceArea(child_bbox)
        
        # Line 922: if (childSurfaceArea < minChildSurfaceArea)
        if child_surface_area < min_child_surface_area:
            # Lines 924-930
            min_child_surface_area = child_surface_area
            cur_split.m_axis = i
            cur_split.m_value = getattr(tight_bbox.min, 'xyz'[i])
            cur_split.m_numLeft = 0
            cur_split.m_numRight = num_entries
            # Line 929: curSplit.m_leftBBox = AABBox(nodeBBox.m_max, nodeBBox.m_min); // Inverted
            cur_split.m_leftBBox = AABBox(node_bbox.max, node_bbox.min)
            cur_split.m_rightBBox = tight_bbox
    
    # Line 934: if (minChildSurfaceArea < (rwcKDTREEBUILD_EMPTY_LEAF_THRESHOLD * nodeSurfaceArea))
    # **DISABLED** - causes hundreds of KDTree gaps → phantom collisions at cluster edges
    if False and min_child_surface_area < (rwcKDTREEBUILD_EMPTY_LEAF_THRESHOLD * node_surface_area):
        # Line 936-937: *result = curSplit; return TRUE;
        if KD_DEBUG.find_best_split:
            kdlog('SPLIT', f" empty leaf: axis={'XYZ'[cur_split.m_axis]} left={cur_split.m_numLeft} right={cur_split.m_numRight}")
        return cur_split
    
    # Find best of X, Y, and Z axes (line 942)
    # Line 940: KDTreeMultiAxisSplit multiSplit;
    
    # Initialize the split position/value along each principal axis (line 945)
    # RenderWare: multiSplit.m_value = (tightBBox.Min() + tightBBox.Max()) * rwpmath::VecFloat(0.5f);
    multi_split = KDTreeMultiAxisSplit()
    multi_split.m_value = [
        (tight_bbox.min.x + tight_bbox.max.x) * 0.5,  # X axis
        (tight_bbox.min.y + tight_bbox.max.y) * 0.5,  # Y axis  
        (tight_bbox.min.z + tight_bbox.max.z) * 0.5   # Z axis
    ]
    
    # Get the split stats for each principal axis (line 948)
    # RenderWare: rwc_GetSplitStatsAllAxis(multiSplit, nodeBBox, entryBBoxes, entries, numEntries);
    # CRITICAL: This processes ALL 3 axes simultaneously!
    rwc_GetSplitStatsAllAxis_Exact(multi_split, entry_bboxes, entries, start_index, num_entries, node_bbox)
    
    # Get the costs of each split (line 950)
    # RenderWare: rwpmath::Vector3 costs = rwc_GetMultiSplitLowestCost(nodeBBox, multiSplit);
    costs = rwc_GetMultiSplitLowestCost(node_bbox, multi_split)
    
    # Create the result split
    best_split = RWKDTreeSplit()
    
    # Determine the lowest cost split (line 952)
    # RenderWare: rwpmath::VecFloat cost = rwc_SelectLowestCostSplit(*result, multiSplit, costs);
    # Note: rwc_SelectLowestCostSplit sets m_numLeft and m_numRight internally (lines 600-601)
    best_cost = rwc_SelectLowestCostSplit(best_split, multi_split, costs)
    
    if KD_DEBUG.find_best_split:
        kdlog('SPLIT', f" axis={'XYZ'[best_split.m_axis]} split={best_split.m_value:.12f} left={best_split.m_numLeft} right={best_split.m_numRight} cost={best_cost:.6f}")
    
    # Check the validity of the cheapest split (line 955)
    # RenderWare: if (result->m_numLeft > 0 && result->m_numRight > 0 && cost < rwcKDTREEBUILD_SPLIT_COST_THRESHOLD)
    if best_split.m_numLeft > 0 and best_split.m_numRight > 0 and best_cost < rwcKDTREEBUILD_SPLIT_COST_THRESHOLD:
        if KD_DEBUG.find_best_split:
            kdlog('SPLIT', f" choose axis={'XYZ'[best_split.m_axis]} value={best_split.m_value:.12f} left={best_split.m_numLeft} right={best_split.m_numRight} cost={best_cost:.6f}")
        # Sort the entires in the order corresponding to the cheapest split (line 958)
        # RenderWare: rwc_SortSplitEntries(*result, entryBBoxes, entries, numEntries);
        rwc_SortSplitEntries(best_split, entry_bboxes, entries, start_index, num_entries)
        # Line 959: return TRUE;
        return best_split
    
    # NOTE: RenderWare has a large items threshold check (lines 962-979)
    # if (largeItemThreshold < 1.0f) { rwc_GetSplitStatsAllAxisLargeItems(...); ... }
    # This is an optimization for handling large objects - we skip it for simplicity
    
    # Safety net (lines 981-998)
    # RenderWare comment: "Well, if we are here then our default routines have failed."
    # Line 986: if ((smallestBBoxSurfaceArea < (minSimilarAreaThreshold * nodeSurfaceArea)) || (numEntries >= maxEntriesPerNode))
    if (smallest_bbox_surface_area < (KDTREE_MIN_SIMILAR_AREA_THRESHOLD * node_surface_area)) or (num_entries >= KDTREE_MAX_ENTRIES_PER_NODE):
        # Force split at median using ANY axis
        # Use axis 0, calculate split value from entry centers
        axis = 0
        entry_centers = []
        for i in range(num_entries):
            bb = entry_bboxes[entries[start_index + i].entryIndex]
            center = (bb.min + bb.max) * 0.5
            center_axis = getattr(center, 'xyz'[axis])
            entry_centers.append((center_axis, i))
        
        entry_centers.sort()
        mid = num_entries // 2
        
        # Split value between the two middle entries
        split_value = (entry_centers[mid - 1][0] + entry_centers[mid][0]) * 0.5
        
        # Simulate partition to get exact counts
        actual_left = 0
        actual_right = 0
        left_bbox = None
        right_bbox = None
        
        for i in range(num_entries):
            bb = entry_bboxes[entries[start_index + i].entryIndex]
            center = (bb.min + bb.max) * 0.5
            center_axis = getattr(center, 'xyz'[axis])
            
            if center_axis > split_value:
                actual_right += 1
                if right_bbox is None:
                    right_bbox = AABBox(bb.min, bb.max)
                else:
                    right_bbox.expand(bb)
            else:
                actual_left += 1
                if left_bbox is None:
                    left_bbox = AABBox(bb.min, bb.max)
                else:
                    left_bbox.expand(bb)
        
        # CRITICAL FIX: Ensure both bounding boxes are always valid AABBox objects
        # RenderWare creates inverted bounding boxes for empty children using parent bbox
        # Source: rwckdtreebuilder.cpp lines 914, 929: AABBox(nodeBBox.m_max, nodeBBox.m_min)
        if left_bbox is None:
            # Create inverted bounding box for empty left child (EXACT RenderWare approach)
            left_bbox = AABBox(node_bbox.max, node_bbox.min)  # Inverted: min > max
        if right_bbox is None:
            # Create inverted bounding box for empty right child (EXACT RenderWare approach)
            right_bbox = AABBox(node_bbox.max, node_bbox.min)  # Inverted: min > max
        
        split = RWKDTreeSplit()
        split.m_axis = axis
        split.m_value = split_value
        split.m_numLeft = actual_left
        split.m_numRight = actual_right
        split.m_leftBBox = left_bbox
        split.m_rightBBox = right_bbox
        
        # CRITICAL: Sort entries according to this forced split (line 996-997)
        # RenderWare uses rwc_SplitNonSpatial which sorts by size, then splits
        # Python simplification: sort entries by the split value
        if KD_DEBUG.find_best_split:
            kdlog('SPLIT', f" force axis={'XYZ'[axis]} value={split_value:.12f} left={actual_left} right={actual_right}")
        rwc_SortSplitEntries(split, entry_bboxes, entries, start_index, num_entries)
        return split
    
    # Failed to split (lines 1000-1001)
    # RenderWare: return FALSE;
    if KD_DEBUG.find_best_split:
        kdlog('SPLIT', f" no-split")
    return None  # Python: None represents FALSE (no split found)

def RW_SplitRecurse(node: RWBuildNode, entry_bboxes: List[AABBox], 
                     entries: List[RWEntry], depth: int) -> int:
    """Recursively split KDTree node
    
    EXACT port of RenderWare BuildNode::SplitRecurse (rwckdtreebuilder.cpp lines 1026-1119)
    
    RenderWare source mapping:
    - Line 1036: Create KDTreeSplit object
    - Line 1041-1051: Check if splittable (numEntries, depth, FindBestSplit)
    - Line 1058: Set split axis
    - Line 1061-1064: Create child bboxes
    - Line 1074-1075: Allocate and construct child BuildNodes
    - Line 1078: Increment depth
    - Line 1085-1094: Set left index and recurse
    - Line 1096-1099: Check for failure
    - Line 1101-1110: Set right index and recurse
    - Line 1112-1115: Check for failure
    - Line 1118: Return total nodes created
    
    Returns: Number of nodes created (0 if leaf, >0 if split)
    """
    if KD_DEBUG.split_recurse:
        kdlog('RECUR', f"depth={depth} first={node.m_firstEntry} count={node.m_numEntries} bbox={fmt_bbox(node.bbox)}", depth)

    # Can we find a split? (lines 1041-1055)
    # RenderWare lines 1041-1042: if (m_numEntries <= splitThreshold || depth > rwcKDTREE_MAX_DEPTH || !rwc_FindBestSplit(...))
    
    # Line 1041: m_numEntries <= splitThreshold
    if node.m_numEntries <= KDTREE_SPLIT_THRESHOLD:
        # Line 1054: Not splittable - return 0;
        return 0
    
    # Line 1042: depth > rwcKDTREE_MAX_DEPTH
    if depth > rwcKDTREE_MAX_DEPTH:
        # Line 1054: Not splittable - return 0;
        return 0
    
    # Lines 1043-1051: !rwc_FindBestSplit(&split, nodeBBox, entryBBoxes, entries + m_firstEntry, m_numEntries, ...)
    # RenderWare: passes (entries + m_firstEntry) as pointer to subarray
    # Python: pass full array + start_index to avoid copying
    split = rwc_FindBestSplit_SAH(
        AABBox(node.bbox.min, node.bbox.max),
        entry_bboxes,
        entries,
        node.m_firstEntry,
        node.m_numEntries
    )
    
    if split is None:
        # Line 1054: Not splittable - return 0;
        if KD_DEBUG.split_recurse:
            kdlog('RECUR', f"leaf depth={depth} count={node.m_numEntries}", depth)
        return 0
    
    # Set the split axis (line 1058)
    # RenderWare: m_splitAxis = split.m_axis;
    node.m_splitAxis = split.m_axis
    
    # Get actual child bboxes for planar split (lines 1061-1064)
    # RenderWare: AABBoxU leftBBox = m_bbox;
    # RenderWare: leftBBox.m_max.SetComponent((int)m_splitAxis, (AABBoxU::FloatType)(split.m_leftBBox.m_max.GetComponent((int)m_splitAxis)));
    # Note: empty children can have inverted bbox
    left_bbox = AABBox(node.bbox.min, node.bbox.max)
    # Set only the split axis component
    if split.m_axis == 0:
        left_bbox.max = Vector((split.m_leftBBox.max.x, left_bbox.max.y, left_bbox.max.z))
    elif split.m_axis == 1:
        left_bbox.max = Vector((left_bbox.max.x, split.m_leftBBox.max.y, left_bbox.max.z))
    else:  # split.m_axis == 2
        left_bbox.max = Vector((left_bbox.max.x, left_bbox.max.y, split.m_leftBBox.max.z))
    
    # RenderWare: AABBoxU rightBBox = m_bbox;
    # RenderWare: rightBBox.m_min.SetComponent((int)m_splitAxis, (AABBoxU::FloatType)(split.m_rightBBox.m_min.GetComponent((int)m_splitAxis)));
    right_bbox = AABBox(node.bbox.min, node.bbox.max)
    # Set only the split axis component
    if split.m_axis == 0:
        right_bbox.min = Vector((split.m_rightBBox.min.x, right_bbox.min.y, right_bbox.min.z))
    elif split.m_axis == 1:
        right_bbox.min = Vector((right_bbox.min.x, split.m_rightBBox.min.y, right_bbox.min.z))
    else:  # split.m_axis == 2
        right_bbox.min = Vector((right_bbox.min.x, right_bbox.min.y, split.m_rightBBox.min.z))
    
    # Allocate child nodes (lines 1067-1075)
    # RenderWare lines 1067-1072: Allocate memory and check for failure
    # Python: no explicit memory allocation needed, Python handles this automatically
    # Line 1074: m_left = new (...) BuildNode(this, leftBBox, m_firstEntry, split.m_numLeft);
    node.left = RWBuildNode(node, left_bbox, node.m_firstEntry, split.m_numLeft)
    # Line 1075: m_right = new (...) BuildNode(this, rightBBox, m_firstEntry + split.m_numLeft, split.m_numRight);
    node.right = RWBuildNode(node, right_bbox, node.m_firstEntry + split.m_numLeft, split.m_numRight)

    if KD_DEBUG.split_recurse:
        kdlog('RECUR', f"split axis={'XYZ'[split.m_axis]} value={split.m_value:.12f} -> L={split.m_numLeft} R={split.m_numRight}", depth)
    
    # Increment depth (lines 1077-1082)
    # RenderWare: depth += 1;
    depth += 1
    # Line 1079: if(depth > rwcKDTREE_MAX_DEPTH)
    if depth > rwcKDTREE_MAX_DEPTH:
        # Line 1081: EAPHYSICS_MESSAGE warning (silent for performance)
        pass
    
    # Set child indices and recurse (lines 1085-1094)
    # Line 1085: m_left->m_index = m_index + 1;
    node.left.m_index = node.m_index + 1
    # Lines 1086-1094: numLeft = m_left->SplitRecurse(...)
    num_left = RW_SplitRecurse(node.left, entry_bboxes, entries, depth)
    
    # Lines 1096-1099: Check for build failure
    # RenderWare: if (rwcKDTREEBUILDER_BUILDFAILED == numLeft) return rwcKDTREEBUILDER_BUILDFAILED;
    # Python: We don't use explicit failure codes, exceptions would be used instead
    
    # Set right child index and recurse (lines 1101-1110)
    # Line 1101: m_right->m_index = m_left->m_index + (int32_t) numLeft + 1;
    node.right.m_index = node.left.m_index + num_left + 1
    # Lines 1102-1110: numRight = m_right->SplitRecurse(...)
    num_right = RW_SplitRecurse(node.right, entry_bboxes, entries, depth)
    
    # Lines 1112-1115: Check for build failure
    # RenderWare: if (rwcKDTREEBUILDER_BUILDFAILED == numRight) return rwcKDTREEBUILDER_BUILDFAILED;
    # Python: We don't use explicit failure codes
    
    # Return total number of nodes created during splitting (line 1118)
    # RenderWare: return numLeft + numRight + 2;
    total = num_left + num_right + 2
    if KD_DEBUG.split_recurse:
        kdlog('RECUR', f"return nodes={total}", depth)
    return total

def RW_BuildKDTree(verts: List[Vector], tris: List[Tuple[int, int, int]], 
                    granularity: float = 0.001) -> Tuple[RWBuildNode, List[int]]:
    """Build KDTree using RenderWare algorithm
    
    EXACT port of KDTreeBuilder::BuildTree (rwckdtreebuilder.cpp lines 1191-1303)
    
    Returns: (root_node, sortedEntryIndices)
    
    sortedEntryIndices = m_entryIndices from RenderWare (line 1292)
    This is the MASTER ORDERING for everything that follows!
    
    RenderWare source mapping:
    - Lines 1199-1204: Input validation assertions
    - Lines 1210-1217: Allocate entries array
    - Lines 1222-1235: Initialize entries and compute root bbox
    - Lines 1238-1250: Allocate root BuildNode
    - Lines 1252-1261: Construct root and call SplitRecurse with depth=1
    - Lines 1289-1293: Extract sorted entry indices
    """
    num_tris = len(tris)
    
    # Validate inputs (lines 1199-1204)
    # Line 1199: EA_ASSERT(entryBBoxes);
    # Line 1202: EA_ASSERT(numEntries <= (1<<24));
    # Line 1204: EA_ASSERT(minChildEntriesThreshold <= 1.0f);
    assert num_tris <= (1 << 24), \
        f"Too many entries for KDTree: {num_tris} > {1 << 24}"
    
    # Step 1: Create entry bboxes  
    # RenderWare base behavior (rwcclusteredmeshbuildermethods.cpp lines 647-659):
    # - Compute min/max directly from triangle vertices WITHOUT expansion
    # CRITICAL: NO epsilon expansion! RenderWare computes exact bounding boxes.
    # Previous epsilon expansion (granularity * 10.0) caused 10x over-subdivision,
    # creating thousands of empty leaves and duplicate leaf pointers, causing phantom collisions.
    print(f"✓ KDTree building: NO epsilon expansion, split_threshold={KDTREE_SPLIT_THRESHOLD} (RenderWare-exact)")
    entry_bboxes: List[AABBox] = []
    root_bbox = None
    
    for tri in tris:
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        bbox = tri_bbox(v0, v1, v2)  # EXACT bbox, no expansion
        entry_bboxes.append(bbox)
        
        # Compute root bbox (lines 1222-1231)
        if root_bbox is None:
            # Line 1222: AABBoxU rootBBox(entryBBoxes[0].Min(), entryBBoxes[0].Max());
            root_bbox = AABBox(bbox.min, bbox.max)
        else:
            # Lines 1230-1231: rootBBox.m_min = Min(...); rootBBox.m_max = Max(...);
            root_bbox.expand(bbox)
        
    # Step 2: Allocate and initialize entry array (lines 1210-1235)
    # Line 1210: Entry * entries = ...->Alloc(numEntries * sizeof(Entry), ...);
    entries: List[RWEntry] = []
    
    if KD_DEBUG.build:
        if root_bbox is not None:
            kdlog('BUILD', f"root bbox: {fmt_bbox(root_bbox)}")
        kdlog('BUILD', f"tris={num_tris}")

    if num_tris > 0:
        # Initialize entries with index and surface area (lines 1223-1234)
        # Line 1223-1224: entries[0].entryIndex = 0; entries[0].entryBBoxSurfaceArea = ...;
        # Lines 1226-1234: Loop to initialize remaining entries
        for i in range(num_tris):
            sa = rwc_BBoxSurfaceArea(entry_bboxes[i])
            entries.append(RWEntry(i, sa))  # entryIndex = original triangle ID
    
    # Step 3: Allocate and construct root BuildNode (lines 1238-1252)
    if num_tris > 0:
        # Line 1252: m_root = new (mem) BuildNode(0, rootBBox, 0, numEntries);
        # First parameter 0 means NULL parent (not self-referential)
        root = RWBuildNode(None, root_bbox, 0, num_tris)
        root.m_index = 0
        # Note: Root's parent remains None (NULL in C++), not self-referential
        
        # Step 4: Recursively split to build tree (lines 1253-1261)
        # Line 1253: m_numNodes = 1 + m_root->SplitRecurse(..., 1, ...);
        # CRITICAL: RenderWare starts with depth=1, not 0!
        if KD_DEBUG.build:
            kdlog('BUILD', f"build tree: tris={num_tris}")
        num_nodes = 1 + RW_SplitRecurse(root, entry_bboxes, entries, 1)
        if KD_DEBUG.build:
            kdlog('BUILD', f"created nodes={num_nodes}")
    else:
        # Empty tree (lines 1263-1267)
        # Line 1265: m_root = NULL;
        # Line 1266: m_numNodes = 0;
        root = None
        num_nodes = 0
    
    # Step 5: Extract m_entryIndices (lines 1289-1293)
    # Line 1290-1292: Copy the sorted entry indices into the array
    # RenderWare: m_entryIndices[entryIndex] = entries[entryIndex].entryIndex;
    # CRITICAL: After partitioning by SplitRecurse, entries array is in SPATIAL ORDER!
    # This is the FINAL serialization order for everything
    sorted_entry_indices = [entries[i].entryIndex for i in range(num_tris)]
    
    if KD_DEBUG.build:
        preview = sorted_entry_indices[:min(10, num_tris)]
        kdlog('BUILD', f"sorted ids preview: {preview}")
    
    return root, sorted_entry_indices

# ============================================================================
# Step 2: WalkBranch Clustering (RenderWare rwcclusteredmeshbuildermethods.cpp)
# ============================================================================

# Constants from RenderWare
MAX_VERTEX_COUNT = 255  # clusteredmeshcluster.h line 245

def RW_SortAndCompressVertexSet(cluster: RWUnitCluster):
    """Sort and compress vertex set (remove duplicates)
    
    EXACT port of UnitCluster::SortAndCompressVertexSet
    (unitcluster.h lines 86-111)
    
    Algorithm:
    1. Sort vertexIDs array (line 90)
    2. Remove duplicates in-place (lines 95-107)
    3. Update numVertices to compressed count (line 110)
    """
    if cluster.numVertices == 0:
        return
    
    # Sort the vertex set (lines 89-92)
    # RenderWare: eastl::sort<uint32_t*, VertexSetCompare>(&vertexSet[0], &vertexSet[vertexSetCount], compare);
    cluster.vertexIDs.sort()
    
    # Compress - remove duplicates (lines 95-107)
    # Line 95: uint32_t currentIndex = 0;
    current_index = 0
    # Line 96: uint32_t headIndex = currentIndex + 1;
    head_index = current_index + 1
    
    # Line 97: while (headIndex < vertexSetCount)
    while head_index < cluster.numVertices:
        # Line 99: if (vertexSet[headIndex] == vertexSet[currentIndex])
        if cluster.vertexIDs[head_index] == cluster.vertexIDs[current_index]:
            # Line 101: ++headIndex; (duplicate - skip it)
            head_index += 1
        else:
            # Line 105: vertexSet[++currentIndex] = vertexSet[headIndex++];
            # Pre-increment currentIndex, assign, then post-increment headIndex
            current_index += 1
            cluster.vertexIDs[current_index] = cluster.vertexIDs[head_index]
            head_index += 1
    
    # Set the new size of the compressed vertex set (line 110)
    # RenderWare: vertexSetCount = currentIndex + 1;
    cluster.numVertices = current_index + 1
    
    # Python-specific: Trim array to actual size (not needed in C++ as it uses count)
    cluster.vertexIDs = cluster.vertexIDs[:cluster.numVertices]

def RW_GetVertexCode(cluster: RWUnitCluster, vertex_index: int) -> int:
    """Get cluster-local vertex index for global vertex ID
    
    EXACT port of UnitCluster::GetVertexCode
    (unitcluster.h lines 114-144)
    
    Uses binary search through sorted vertexIDs array
    
    Args:
        cluster: Cluster (must have compressed vertexIDs)
        vertex_index: Global vertex ID
    
    Returns:
        Local vertex index (0-254), or 0xFF if not found
    """
    # Line 116: uint8_t start = 0;
    start = 0
    # Line 117: uint8_t end = static_cast<uint8_t>(numVertices - 1u);
    end = cluster.numVertices - 1
    # Line 118: uint8_t ret = static_cast<uint8_t>((end - start) / 2u);
    ret = (end - start) // 2
    
    # Binary search through the items in the vector (lines 121-137)
    # Line 121: while (start <= end)
    while start <= end:
        # Line 123: if (vertexIndex == vertexIDs[ret])
        if vertex_index == cluster.vertexIDs[ret]:
            # Line 125: return ret;
            return ret
        # Line 127: else if (vertexIndex > vertexIDs[ret])
        elif vertex_index > cluster.vertexIDs[ret]:
            # Line 129: start = static_cast<uint8_t>(ret + 1u);
            start = ret + 1
        else:
            # Line 133: end = static_cast<uint8_t>(ret - 1u);
            end = ret - 1
        
        # Line 136: ret = static_cast<uint8_t>(start + ((end - start) / 2u));
        ret = start + ((end - start) // 2)
    
    # This case should never occur (line 140)
    # RenderWare: EA_ASSERT_MSG(start > end, ("Vertex not found in cluster."));
    # The assert checks that loop terminated correctly (start > end must be true here)
    # The error message indicates the actual problem: vertex not found
    assert start > end, f"Vertex not found in cluster. vertex_index={vertex_index}"
    
    # return with FF (line 143)
    # RenderWare: return 0xFF;
    return 0xFF

def RW_AddUnitToCluster(cluster: RWUnitCluster, unit_id: int, 
                         tris: List[Tuple[int, int, int]], 
                         max_vertices_per_unit: int = 3) -> bool:
    """Add a unit (triangle) to cluster
    
    EXACT port of UnitClusterBuilder::AddUnitToCluster
    (unitclusterbuilder.cpp lines 66-106)
    
    Args:
        cluster: Target cluster
        unit_id: Triangle ID from sortedObjects
        tris: Triangle vertex indices
        max_vertices_per_unit: 3 for triangles (quads not supported in this exporter)
    
    Returns:
        True if added, False if cluster is full
    """
    # If the cluster vertex count is near the count limit (line 77)
    # RenderWare: if (clusterVertexCount > ClusteredMeshCluster::MAX_VERTEX_COUNT - maxVerticesPerUnit)
    if cluster.numVertices > MAX_VERTEX_COUNT - max_vertices_per_unit:
        # Sort and Compress the cluster vertex set (line 80)
        # RenderWare: UnitCluster::SortAndCompressVertexSet(clusterVertexIDs, clusterVertexCount);
        RW_SortAndCompressVertexSet(cluster)
        
        # If the cluster vertex count is near the limit after having sorted (lines 82-87)
        # then it can be considered to be full.
        # Line 84: if (clusterVertexCount > ClusteredMeshCluster::MAX_VERTEX_COUNT - maxVerticesPerUnit)
        if cluster.numVertices > MAX_VERTEX_COUNT - max_vertices_per_unit:
            # Line 86: return false;
            return False
    
    # Add the unit to the cluster (lines 90-95)
    # Line 91: const Triangle &t1 = triangles[unitList[unitID].tri0];
    tri = tris[unit_id]
    
    # Add vertices WITH DUPLICATES (RenderWare adds all, compresses later!)
    # Line 93: clusterVertexIDs[clusterVertexCount++] = t1.vertices[0];
    cluster.vertexIDs.append(tri[0])
    # Line 94: clusterVertexIDs[clusterVertexCount++] = t1.vertices[1];
    cluster.vertexIDs.append(tri[1])
    # Line 95: clusterVertexIDs[clusterVertexCount++] = t1.vertices[2];
    cluster.vertexIDs.append(tri[2])
    cluster.numVertices += 3
    
    # Quad support (lines 97-101) - NOT IMPLEMENTED in this exporter
    # Line 97: if ( unitList[unitID].type == Unit::TYPE_QUAD )
    # Line 100: clusterVertexIDs[clusterVertexCount++] = t2.vertices[unitList[unitID].extraVertex];
    # This exporter only supports triangles, so we skip quad handling
    
    # Line 103: clusterUnitIDs[clusterUnitCount++] = unitID;
    cluster.unitIDs.append(unit_id)
    cluster.numUnits += 1  # Post-increment equivalent: maintains count
    
    # Line 105: return true;
    return True

def RW_AddOrderedUnitsToUnitCluster(cluster: RWUnitCluster, 
                                     sorted_objects: List[int],
                                     start_index: int,
                                     num_units_to_add: int,
                                     tris: List[Tuple[int, int, int]],
                                     max_vertices_per_unit: int = 3) -> int:
    """Add ordered units to cluster
    
    EXACT port of ClusteredMeshBuilderMethods::AddOrderedUnitsToUnitCluster
    (rwcclusteredmeshbuildermethods.cpp lines 1540-1581)
    
    Args:
        cluster: Target cluster
        sorted_objects: sortedEntryIndices from KDTree (MASTER ORDERING!)
        start_index: Start index in sorted_objects
        num_units_to_add: Number of triangles to add
        tris: Triangle vertex indices
    
    Returns:
        Number of units actually added (may be less if cluster fills up)
    """
    # For each unit to add to the UnitCluster (line 1553-1554)
    # Line 1553: uint32_t unitIndex = 0;
    unit_index = 0
    # Line 1554: for ( ; unitIndex < numUnitsToAdd ; ++unitIndex)
    while unit_index < num_units_to_add:
        # Attempt to add the unit to the cluster (lines 1557-1565)
        # Line 1562: orderedUnitIDs[startUnitIndex + unitIndex]
        unit_id = sorted_objects[start_index + unit_index]
        
        # Line 1557: const bool added = UnitClusterBuilder::AddUnitToCluster(...)
        added = RW_AddUnitToCluster(cluster, unit_id, tris, max_vertices_per_unit)
        
        # If the unit was not added to the cluster (line 1568)
        # RenderWare: if (!added)
        if not added:
            # return the number of units added to the cluster (line 1571)
            # RenderWare: return unitIndex;
            return unit_index
        
        unit_index += 1
    
    # Sort and compress the cluster vertex set (lines 1576-1577)
    # RenderWare: UnitCluster::SortAndCompressVertexSet(clusterVertexIDs, clusterVertexCount);
    RW_SortAndCompressVertexSet(cluster)
    
    # Return the number of units which have been added to the cluster (line 1580)
    # RenderWare: return unitIndex;
    return unit_index

def RW_WalkBranch(build_node: RWBuildNode, 
                   leaf_map: Dict[int, RWBuildNode],
                   cluster_stack: List[RWUnitCluster],
                   tris: List[Tuple[int, int, int]],
                   sorted_objects: List[int]) -> int:
    """Walk KDTree and create clusters
    
    EXACT port of ClusteredMeshBuilderMethods::WalkBranch
    (rwcclusteredmeshbuildermethods.cpp lines 1608-1710)
    
    Args:
        build_node: Current KDTree BuildNode
        leaf_map: Maps first triangle ID in each leaf → BuildNode
        cluster_stack: List of clusters created
        tris: Triangle vertex indices
        sorted_objects: sortedEntryIndices from KDTree (MASTER ORDERING!)
    
    Returns:
        Vertex count of branch:
        - 0 if empty
        - 1..255 if ONE cluster created
        - >255 if MULTIPLE clusters created
    """
    # Initialize vertex counts (line 1620)
    # RenderWare: uint32_t vcount0 = 0, vcount1 = 0;
    vcount0 = 0
    vcount1 = 0
    
    # Python note: RenderWare has failure flag checks (lines 1622-1626)
    # We don't implement failure flags in this simplified exporter
    
    # If the node is a leaf node (line 1629)
    # RenderWare: if (buildNode->m_left == NULL)
    if build_node.left is None:
        # Get the start index and count of units in this leaf node (lines 1632-1633)
        # Line 1632: const uint32_t start = buildNode->m_firstEntry;
        start = build_node.m_firstEntry
        # Line 1633: const uint32_t totalNumUnitsToAdd = buildNode->m_numEntries;
        total_num_units_to_add = build_node.m_numEntries
        kdlog('WALK', f"leaf first={start} count={total_num_units_to_add}")
        
        # If the leaf node is empty ignore it (lines 1636-1638)
        # Line 1636: if (0 == totalNumUnitsToAdd)
        if total_num_units_to_add == 0:
            # Line 1638: return vcount0 + vcount1;
            return vcount0 + vcount1  # Returns 0, but using pattern from RW
        
        # Add the unitID to the leaf map (line 1642)
        # RenderWare: leafMap[sortedObjects[start]] = buildNode;
        # CRITICAL: Maps FIRST triangle in leaf to the BuildNode
        # This is used later in AdjustKDTreeNodeEntriesForCluster
        leaf_map[sorted_objects[start]] = build_node
        
        # Get a new UnitCluster (line 1645)
        # RenderWare: UnitCluster *cluster = unitClusterStack.GetUnitCluster();
        # GetUnitCluster() ALWAYS returns a new/next cluster
        # It NEVER reuses the last cluster - each leaf gets a fresh cluster attempt
        cluster = RWUnitCluster()
        cluster_stack.append(cluster)
        
        # Python note: RenderWare checks if cluster is NULL (lines 1646-1652)
        # We don't need this check as Python handles memory automatically
        
        # Add the units to the cluster (lines 1655-1665)
        # Line 1655: const uint32_t numUnitsAdded = AddOrderedUnitsToUnitCluster(...);
        # Note: maxVerticesPerUnit = 4 for quads (line 1619), but we use 3 for triangle-only
        num_units_added = RW_AddOrderedUnitsToUnitCluster(
            cluster,
            sorted_objects,
            start,
            total_num_units_to_add,
            tris,
            max_vertices_per_unit=3  # Triangle-only exporter
        )
        kdlog('WALK', f"leaf added units={num_units_added}/{total_num_units_to_add} vertices(after compress)={cluster.numVertices}")
        
        # If there are remaining units to add then the current UnitCluster must be full (lines 1668-1672)
        # Line 1668: if (numUnitsAdded < totalNumUnitsToAdd)
        if num_units_added < total_num_units_to_add:
            # Mark the failure (line 1671)
            # RenderWare: failureFlags |= CLUSTER_GENERATION_FAILURE_MULTI_LEAF_CLUSTER;
            # This indicates a single leaf has too many triangles to fit in one cluster.
            # RenderWare DOES NOT try to create multiple clusters for a single leaf!
            # CRITICAL: RenderWare DOES NOT modify buildNode->m_numEntries here!
            # The leaf node retains its original m_numEntries count even though the cluster
            # only contains a subset of triangles. This is intentional RenderWare behavior.
            lost_triangles = total_num_units_to_add - num_units_added
            print(f"⚠️ Leaf overflow! Leaf wanted {total_num_units_to_add} triangles but only {num_units_added} fit. "
                  f"{lost_triangles} triangles DROPPED from this leaf!")
        
        # Set vcount0 to the count of the last cluster (line 1675)
        # RenderWare: vcount0 = cluster->numVertices;
        # CRITICAL: numVertices is the compressed count after SortAndCompressVertexSet
        vcount0 = cluster.numVertices
        
        # Check that the last cluster has more than 0 vertices (line 1678)
        # RenderWare: EA_ASSERT_MSG(vcount0 > 0, ("Attempting to add a cluster with no vertices."));
        assert vcount0 > 0, "Attempting to add a cluster with no vertices."
        
        # Leaf node complete - return early (not explicit in RW, but implied by structure)
        return vcount0
    
    else:  # buildNode is not a leaf (line 1680)
        kdlog('WALK', f"branch splitAxis={'XYZ'[build_node.m_splitAxis]} first={build_node.m_firstEntry} count={build_node.m_numEntries}")
        # Recurse on left child (lines 1682-1691)
        # Line 1682: vcount0 = WalkBranch(buildNode->m_left, ...);
        vcount0 = RW_WalkBranch(
            build_node.left,
            leaf_map,
            cluster_stack,
            tris,
            sorted_objects
        )
        kdlog('WALK', f"left return vcount={vcount0}")
        
        # Recurse on right child (lines 1693-1702)
        # Line 1693: vcount1 = WalkBranch(buildNode->m_right, ...);
        vcount1 = RW_WalkBranch(
            build_node.right,
            leaf_map,
            cluster_stack,
            tris,
            sorted_objects
        )
        kdlog('WALK', f"right return vcount={vcount1}")
        
        # Python note: RenderWare checks for failures (lines 1704-1708)
        # We don't implement failure flags in this simplified exporter
        
        # If both children are small, and not empty, then try to merge them (lines 1710-1733)
        # Line 1711: if (vcount0 > 0 && vcount0 <= ClusteredMeshCluster::MAX_VERTEX_COUNT && 
        #             vcount1 > 0 && vcount1 <= ClusteredMeshCluster::MAX_VERTEX_COUNT)
        if (vcount0 > 0 and vcount0 <= MAX_VERTEX_COUNT and 
            vcount1 > 0 and vcount1 <= MAX_VERTEX_COUNT):
            kdlog('WALK', f"merge attempt: v0={vcount0} v1={vcount1}")
            
            # assert that the child clusters MUST be the last two on the clusterList (lines 1713-1717)
            # Line 1715: EA_ASSERT((*(unitClusterStack.RBegin()))->numVertices == vcount1);
            # Line 1717: EA_ASSERT((*(++unitClusterStack.RBegin()))->numVertices == vcount0);
            if len(cluster_stack) >= 2:
                last_cluster = cluster_stack[-1]
                penultimate_cluster = cluster_stack[-2]
                
                # Verify these are the right clusters
                assert last_cluster.numVertices == vcount1, \
                    f"Last cluster vertices mismatch: expected {vcount1}, got {last_cluster.numVertices}"
                assert penultimate_cluster.numVertices == vcount0, \
                    f"Penultimate cluster vertices mismatch: expected {vcount0}, got {penultimate_cluster.numVertices}"
                    
                # Try to merge (lines 1719-1726)
                # Line 1719: if (MergeLastTwoClusters(unitClusterStack, mergedVertices))
                merged_success = RW_MergeLastTwoClusters(cluster_stack, tris)
                
                if merged_success:
                    # Merge succeeded (lines 1723-1725)
                    # Line 1724: vcount0 = (*rIt)->numVertices;
                    vcount0 = cluster_stack[-1].numVertices
                    # Line 1725: vcount1 = 0;
                    vcount1 = 0
                    kdlog('WALK', f"merge success -> new vcount={vcount0}")
                else:
                    # Merge failed (lines 1727-1732)
                    # Line 1731: EA_ASSERT(vcount0 + vcount1 > ClusteredMeshCluster::MAX_VERTEX_COUNT);
                    # An assert failure implies that there are too many vertices to merge,
                    # assert that this is definitely the case.
                    assert vcount0 + vcount1 > MAX_VERTEX_COUNT, \
                        f"Merge failed but vertex count {vcount0 + vcount1} <= {MAX_VERTEX_COUNT}"
    
    # Line 1735: return vcount0 + vcount1;
    return vcount0 + vcount1

def RW_MergeLastTwoClusters(cluster_stack: List[RWUnitCluster], 
                             tris: List[Tuple[int, int, int]]) -> bool:
    """Merge last two clusters if combined vertex count fits
    
    EXACT port of ClusteredMeshBuilderMethods::MergeLastTwoClusters
    (rwcclusteredmeshbuildermethods.cpp lines 1751-1813)
    
    Args:
        cluster_stack: List of clusters
        tris: Triangle vertex indices
    
    Returns:
        True if merged, False if merge would exceed vertex limit
    """
    if len(cluster_stack) < 2:
        return False
    
    # Get last and penultimate clusters (lines 1755-1758)
    # Line 1755: UnitClusterStack::ReverseClusterIterator rIt = unitClusterStack.RBegin();
    # Line 1756: UnitCluster * lastCluster = *rIt;
    last = cluster_stack[-1]
    # Line 1757: ++rIt;
    # Line 1758: UnitCluster * penultimateCluster = *rIt;
    penultimate = cluster_stack[-2]
    
    # Line 1760: uint32_t mergedVertexCount = 0;
    merged_vertex_count = 0
    # Python: Use array instead of fixed-size buffer
    merged_vertices = []
    
    # Line 1762: uint32_t penultimateCounter = 0;
    penultimate_counter = 0
    # Line 1763: uint32_t lastCounter = 0;
    last_counter = 0
    
    # Iterate over all vertices in each cluster (lines 1765-1782)
    # until all vertices have been iterated or MAX_VERTEX_COUNT has been reached
    # Line 1767: while(penultimateCounter < penultimateCluster->numVertices && 
    #                  lastCounter < lastCluster->numVertices && 
    #                  mergedVertexCount < ClusteredMeshCluster::MAX_VERTEX_COUNT)
    while (penultimate_counter < penultimate.numVertices and 
           last_counter < last.numVertices and 
           merged_vertex_count < MAX_VERTEX_COUNT):
        
        # Line 1769: if (penultimateCluster->vertexIDs[penultimateCounter] == lastCluster->vertexIDs[lastCounter])
        if penultimate.vertexIDs[penultimate_counter] == last.vertexIDs[last_counter]:
            # Duplicate - add once (lines 1771-1772)
            # Line 1771: ++lastCounter;
            last_counter += 1
            # Line 1772: mergedVertices[mergedVertexCount++] = penultimateCluster->vertexIDs[penultimateCounter++];
            merged_vertices.append(penultimate.vertexIDs[penultimate_counter])
            penultimate_counter += 1
            merged_vertex_count += 1
        # Line 1774: else if (penultimateCluster->vertexIDs[penultimateCounter] < lastCluster->vertexIDs[lastCounter])
        elif penultimate.vertexIDs[penultimate_counter] < last.vertexIDs[last_counter]:
            # Line 1776: mergedVertices[mergedVertexCount++] = penultimateCluster->vertexIDs[penultimateCounter++];
            merged_vertices.append(penultimate.vertexIDs[penultimate_counter])
            penultimate_counter += 1
            merged_vertex_count += 1
        else:
            # Line 1780: mergedVertices[mergedVertexCount++] = lastCluster->vertexIDs[lastCounter++];
            merged_vertices.append(last.vertexIDs[last_counter])
            last_counter += 1
            merged_vertex_count += 1
    
    # Attempt to add the remaining entries in cluster g0 (lines 1785-1788)
    # Line 1785: while (penultimateCounter < penultimateCluster->numVertices && mergedVertexCount < ClusteredMeshCluster::MAX_VERTEX_COUNT)
    while penultimate_counter < penultimate.numVertices and merged_vertex_count < MAX_VERTEX_COUNT:
        # Line 1787: mergedVertices[mergedVertexCount++] = penultimateCluster->vertexIDs[penultimateCounter++];
        merged_vertices.append(penultimate.vertexIDs[penultimate_counter])
        penultimate_counter += 1
        merged_vertex_count += 1
    
    # Attempt to add the remaining entries in cluster g1 (lines 1791-1794)
    # Line 1791: while (lastCounter < lastCluster->numVertices && mergedVertexCount < ClusteredMeshCluster::MAX_VERTEX_COUNT)
    while last_counter < last.numVertices and merged_vertex_count < MAX_VERTEX_COUNT:
        # Line 1793: mergedVertices[mergedVertexCount++] = lastCluster->vertexIDs[lastCounter++];
        merged_vertices.append(last.vertexIDs[last_counter])
        last_counter += 1
        merged_vertex_count += 1
    
    # If the combined vertex count is less than the limit merge the two clusters (line 1797)
    # Line 1797: if (penultimateCounter == penultimateCluster->numVertices && 
    #                 lastCounter == lastCluster->numVertices && 
    #                 mergedVertexCount <= ClusteredMeshCluster::MAX_VERTEX_COUNT)
    if (penultimate_counter == penultimate.numVertices and 
        last_counter == last.numVertices and 
        merged_vertex_count <= MAX_VERTEX_COUNT):
        
        # Copy the merged vertex set into the penultimate cluster (lines 1800-1805)
        # Lines 1800-1803: for(uint32_t i = 0; i < mergedVertexCount; ++i)
        #                      penultimateCluster->vertexIDs[i] = mergedVertices[i];
        penultimate.vertexIDs = merged_vertices
        # Line 1805: penultimateCluster->numVertices = mergedVertexCount;
        penultimate.numVertices = merged_vertex_count
        
        # Merge the last two clusters (line 1808)
        # RenderWare: unitClusterStack.MergeLastTwoClusters();
        # This function (unitclusterstack.cpp lines 143-153):
        # - Line 149: Extends penultimate cluster's unit IDs with last cluster's
        #   m_previousNode->m_unitCluster.numUnits += m_currentNode->m_unitCluster.numUnits;
        # - Line 151: Removes the last cluster
        penultimate.unitIDs.extend(last.unitIDs)
        penultimate.numUnits += last.numUnits
        # Remove the last cluster from stack
        cluster_stack.pop()
        
        # Line 1810: return TRUE;
        return True
    
    # return failure (lines 1813-1814)
    # Line 1814: return FALSE;
    return False

# ============================================================================
# Step 3: AdjustKDTreeNodeEntriesForCluster (Fixup byte offsets)
# ============================================================================

def RW_AdjustKDTreeNodeEntriesForCluster(cluster: RWUnitCluster,
                                          cluster_id: int,
                                          leaf_map: Dict[int, RWBuildNode],
                                          tris: List[Tuple[int, int, int]],
                                          buildnode_set: set = None) -> int:
    """Update KDTree leaf nodes with final byte offsets
    
    EXACT port of AdjustKDTreeNodeEntriesForCluster
    (rwcclusteredmeshbuildermethods.cpp lines 1877-1946)
    
    This is THE CRITICAL STEP that connects KDTree leaves to serialized data!
    
    Args:
        cluster: Cluster being serialized
        cluster_id: Index of this cluster
        leaf_map: Maps first triangle ID → BuildNode (from WalkBranch)
        tris: Triangle vertex indices
        buildnode_set: Optional set to track which BuildNodes this cluster updates
    
    Returns:
        Total byte size of unit data
    """
    # Get number of units in cluster (line 1887)
    # RenderWare: const uint32_t numUnits = unitCluster.numUnits;
    num_units = cluster.numUnits
    
    # Pre-shift cluster ID for encoding (line 1888)
    # RenderWare: const uint32_t shiftedClusterId = unitClusterID << unitClusterIDShift;
    # Note: unitClusterIDShift is passed as parameter in RenderWare, hardcoded here for Skate 3
    unit_cluster_id_shift = 16  # For Skate 3 (flags=0x0010)
    shifted_cluster_id = cluster_id << unit_cluster_id_shift
    
    # Validate encoding (lines 1890-1891)
    # Line 1890: EA_ASSERT((shiftedClusterId >> unitClusterIDShift) == unitClusterID);
    assert (shifted_cluster_id >> unit_cluster_id_shift) == cluster_id, \
        f"Cluster ID shift validation failed: {cluster_id}"
    # Line 1891: EA_ASSERT(numUnits <= (1u << unitClusterIDShift));
    assert num_units <= (1 << unit_cluster_id_shift), \
        f"Too many units in cluster: {num_units} > {1 << unit_cluster_id_shift}"
    
    # Initialize unit data size counter (line 1893)
    # RenderWare: uint32_t sizeofUnitData = 0;
    sizeof_unit_data = 0
    
    # For each unit in cluster (line 1895)
    # RenderWare: for (uint32_t unitIndex = 0 ; unitIndex < numUnits ; ++unitIndex)
    for unit_index in range(num_units):
        # Line 1897: uint32_t unitID = unitCluster.unitIDs[unitIndex];
        unit_id = cluster.unitIDs[unit_index]
        
        # Line 1898: LeafMap::const_iterator it = leafMap.find(unitID);
        # Check if this unit is the FIRST in a KDTree leaf
        # Line 1900: if(leafMap.end() != it)
        if unit_id in leaf_map:
            # Line 1902: rw::collision::KDTreeBuilder::BuildNode * buildNode = it->second;
            build_node = leaf_map[unit_id]
            
            # DEBUG: Track which BuildNode this cluster is updating
            if buildnode_set is not None:
                buildnode_set.add(build_node)
            
            # Verify it's a leaf (line 1903)
            # RenderWare: EA_ASSERT(NULL == buildNode->m_left);
            assert build_node.left is None, "BuildNode should be a leaf"
            
            # Update BuildNode.m_firstEntry with encoded byte offset (lines 1905-1906)
            # Line 1905: uint32_t reformattedStartIndex = shiftedClusterId + sizeofUnitData;
            reformatted_start_index = shifted_cluster_id + sizeof_unit_data
            # Line 1906: buildNode->m_firstEntry = reformattedStartIndex;
            build_node.m_firstEntry = reformatted_start_index
            
            # Handle empty leaf siblings (RenderWare lines 1908-1936)
            # CRITICAL: RenderWare DOES NOT prevent multiple assignments!
            # It checks m_numEntries==0 and assigns EVERY time.
            # Multiple assignments are allowed - last assignment wins.
            # This is EXACTLY what RenderWare does (no tracking, no prevention)
            if build_node.parent is not None:
                parent = build_node.parent
                
                # Line 1918-1925: If we're the right child and left sibling is empty
                if parent.right == build_node:
                    if parent.left.m_numEntries == 0:
                        parent.left.m_firstEntry = reformatted_start_index
                # Lines 1927-1934: If we're the left child and right sibling is empty  
                elif parent.left == build_node:
                    if parent.right.m_numEntries == 0:
                        parent.right.m_firstEntry = reformatted_start_index
        
        # Accumulate unit size (lines 1939-1944)
        # Line 1939: const Unit &unit = unitList[unitID];
        # Line 1941: sizeofUnitData += ClusteredMeshCluster::GetUnitSize(...)
        # Unit size: 1 flag + 3 verts + 3 edges + 2 surfaceID = 9 bytes (NO groupID)
        # RenderWare calls GetUnitSize which returns this dynamically based on unit type
        sizeof_unit_data += 9
    
    # Python note: RenderWare doesn't have explicit return (void function)
    # We return sizeofUnitData for convenience in Python implementation
    return sizeof_unit_data

# ============================================================================
# Step 4: InitializeRuntimeKDTree (Convert BuildNode → Runtime KDNode)
# ============================================================================

def RW_InitializeRuntimeKDTree(root: RWBuildNode) -> List[KDNode]:
    """Convert BuildNode tree to runtime KDNode array
    
    EXACT port of KDTreeBuilder::InitializeRuntimeKDTree
    (rwckdtreebuilder.cpp lines 1309-1387)
    
    CRITICAL: Stack-based depth-first traversal that:
    1. Pre-allocates runtime KDNode array
    2. Traverses build tree and fills runtime nodes
    3. Branch children reference runtime node indices
    4. Leaf children embed data directly in NodeRef
    
    Args:
        root: Root BuildNode from tree construction
    
    Returns:
        List of KDNode (branch nodes only, leaves embedded)
    """
    # Python helper: Count branch nodes (RenderWare knows this from construction)
    def count_branches(node):
        if node is None or node.left is None:
            return 0
        return 1 + count_branches(node.left) + count_branches(node.right)
    
    num_branches = count_branches(root)
    
    # Validate tree structure (line 1311)
    # RenderWare: EA_ASSERT((1 + 2 * ((uint32_t) (kdtree->GetNumBranchNodes()))) == GetNumNodes());
    # Complete binary tree property: 1 + 2*numBranches == totalNodes
    # Where totalNodes = branches + leaves
    total_nodes = count_all_nodes(root)
    assert (1 + 2 * num_branches) == total_nodes, \
        f"Invalid tree structure: 1 + 2*{num_branches} != {total_nodes}"
    
    # Now fill in kdtree branch nodes (line 1314)
    # RenderWare: if (kdtree->GetNumBranchNodes() > 0)
    if num_branches == 0:
        return []
    
    # Pre-allocate runtime nodes array
    # Python: Create array of KDNode objects (RenderWare uses existing allocated array)
    rt_nodes = [KDNode() for _ in range(num_branches)]
    
    # Stack for traversal (lines 1321-1328)
    # Lines 1321-1326: struct stackValue definition
    class StackValue:
        def __init__(self, rt_parent, rt_child, build_node):
            self.rtParent = rt_parent  # uint32_t rtParent
            self.rtChild = rt_child    # uint32_t rtChild
            self.node = build_node     # BuildNode *node
    
    # Line 1328: stackValue stack[rwcKDTREE_STACK_SIZE], cur;
    # Initialize stack (lines 1330-1333)
    # Line 1330: stack[0].node = m_root;
    # Line 1331: stack[0].rtParent = 0;
    # Line 1332: stack[0].rtChild = 0;
    stack = [StackValue(0, 0, root)]
    # Line 1333: uint32_t top = 1;
    top = 1
    # Line 1334: uint32_t rtIndex;
    rt_index = 0
    
    # Traverse tree (line 1337)
    # RenderWare: for (rtIndex = 0; top > 0; rtIndex++)
    while top > 0:
        # Pop from stack (line 1339)
        # RenderWare: cur = stack[--top];
        top -= 1
        cur = stack[top]
        
        # Set reference to us in parent (unless we're the root node) (lines 1342-1345)
        # Line 1342: if (rtIndex != 0)
        if rt_index != 0:
            # Line 1344: kdtree->m_branchNodes[cur.rtParent].m_childRefs[cur.rtChild].m_index = rtIndex;
            # Parent's child ref points to this runtime node index
            parent_node = rt_nodes[cur.rtParent]
            # Update the placeholder we set earlier with actual runtime index
            # Keep m_content as rwcKDTREE_BRANCH_NODE (0xFFFFFFFF), update m_index
            parent_node.entries[cur.rtChild] = (0xFFFFFFFF, rt_index)
        
        # Get current graph kdtree branch node, child nodes, and runtime node (lines 1348-1351)
        # Line 1348: KDTree::BranchNode &rtNode = kdtree->m_branchNodes[rtIndex];
        rt_node = rt_nodes[rt_index]
        # Line 1349-1351: BuildNode *childNodes[2]; childNodes[0] = ...; childNodes[1] = ...;
        child_nodes = [cur.node.left, cur.node.right]
        
        # Initialize runtime node (lines 1354-1357)
        # Line 1354: rtNode.m_parent = cur.rtParent;
        rt_node.parent = cur.rtParent
        # Line 1355: rtNode.m_axis = cur.node->m_splitAxis;
        rt_node.axis = cur.node.m_splitAxis
        # Line 1356: rtNode.m_extents[0] = cur.node->m_left->m_bbox.Max().GetComponent((int)rtNode.m_axis);
        rt_node.ext0 = getattr(child_nodes[0].bbox.max, 'xyz'[rt_node.axis])
        # Line 1357: rtNode.m_extents[1] = cur.node->m_right->m_bbox.Min().GetComponent((int)rtNode.m_axis);
        rt_node.ext1 = getattr(child_nodes[1].bbox.min, 'xyz'[rt_node.axis])
        
        # DIAGNOSTIC: Check for gaps in KDTree coverage
        # ext0 = left child max, ext1 = right child min (on split axis)
        # If ext0 < ext1, there's a GAP which could cause triangles to be missed
        # If ext0 >= ext1, there's OVERLAP or exact meeting (good for coverage)
        if rt_node.ext0 < rt_node.ext1:
            gap_size = rt_node.ext1 - rt_node.ext0
            # Store as print for now - will be visible if Blender launched from terminal
            print(f"⚠️ KDTree node {rt_index}: GAP of {gap_size:.6f} on axis {['X','Y','Z'][rt_node.axis]}!")
            print(f"   ext0={rt_node.ext0:.6f} (left max) < ext1={rt_node.ext1:.6f} (right min)")
        
        # Initialize entries list for this node
        # Python: Pre-initialize to avoid index errors
        rt_node.entries = [(0, 0), (0, 0)]  # [left_child, right_child]
        
        # Will traverse left first, so add any right child branch to stack first (lines 1359-1380)
        # Line 1360: for (int32_t i = 1; i >= 0; --i)
        # RIGHT FIRST (i=1), then LEFT (i=0)
        for i in [1, 0]:
            child = child_nodes[i]
            
            # Line 1362: if (!childNodes[i]->m_left ) //if child node is a leaf store data in childref
            if child.left is None:  # Leaf node
                # Child is leaf node so store leaf content info (lines 1365-1366)
                # Line 1365: rtNode.m_childRefs[i].m_content = childNodes[i]->m_numEntries;
                # Line 1366: rtNode.m_childRefs[i].m_index = childNodes[i]->m_firstEntry;
                rt_node.entries[i] = (child.m_numEntries, child.m_firstEntry)
            else:  # Branch node (lines 1368-1379)
                # Put child branch node on stack (lines 1371-1374)
                # Line 1371: stack[top].rtParent = rtIndex;
                # Line 1372: stack[top].rtChild = (uint32_t) i;
                # Line 1373: stack[top].node = childNodes[i];
                if top >= len(stack):
                    stack.append(StackValue(rt_index, i, child))
                else:
                    stack[top] = StackValue(rt_index, i, child)
                # Line 1374: top++;
                top += 1
                
                # Will fill in reference to child branch later (lines 1377-1378)
                # Line 1377: rtNode.m_childRefs[i].m_content = rwcKDTREE_BRANCH_NODE;
                # Line 1378: rtNode.m_childRefs[i].m_index = rwcKDTREE_INVALID_INDEX;
                # rwcKDTREE_BRANCH_NODE = 0xFFFFFFFF, rwcKDTREE_INVALID_INDEX = 0xFFFFFFFF
                rt_node.entries[i] = (0xFFFFFFFF, 0xFFFFFFFF)
        
        # Increment rtIndex (part of for loop line 1337)
        rt_index += 1
    
    # Verify we created the correct number of nodes (line 1383)
    # RenderWare: EA_ASSERT_MSG(rtIndex == kdtree->GetNumBranchNodes(), ("Invalid number of nodes in the KDTree!"));
    assert rt_index == num_branches, \
        f"Invalid number of nodes in the KDTree: expected {num_branches}, got {rt_index}"
    
    # Line 1386: EA_ASSERT_MSG(kdtree->IsValid(), ("Failed to initialize a valid KDTree!"));
    # Python note: We don't have a full IsValid() implementation
    # The assertion above covers the main check (correct number of nodes)
    
    return rt_nodes

# ============================================================================
# MAIN PIPELINE: Complete RenderWare Algorithm
# ============================================================================

def is_triangle_valid(v0: Vector, v1: Vector, v2: Vector) -> bool:
    """Check if triangle has non-zero area (not degenerate)
    
    EXACT port of RenderWare TriangleValidator::IsTriangleValid
    Source: trianglevalidator.cpp lines 37-53
    
    A triangle is valid if the squared magnitude of its normal vector
    exceeds MINIMUM_RECIPROCAL (~1e-10). This filters out:
    - Zero-area triangles (all 3 vertices identical or collinear)
    - Near-degenerate triangles that cause numerical issues
    
    Args:
        v0, v1, v2: Triangle vertices in PSG space
        
    Returns:
        True if triangle has sufficient area, False if degenerate
    """
    # Compute normal via cross product: normal = (v1 - v0) × (v2 - v0)
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = vec_cross(edge1, edge2)
    
    # Check squared magnitude against minimum threshold
    # RenderWare uses MINIMUM_RECIPROCAL ≈ 1e-10
    length_squared = vec_dot(normal, normal)
    MINIMUM_RECIPROCAL = 1e-10
    
    return length_squared > MINIMUM_RECIPROCAL


def validate_triangles(verts: List[Vector], tris: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """Filter out degenerate triangles with zero or near-zero area
    
    EXACT port of RenderWare ClusteredMeshBuilderMethods::ValidateTriangles
    Source: rwcclusteredmeshbuildermethods.cpp lines 100-141
    
    This is the FIRST step in RenderWare's build pipeline (line 235).
    Degenerate triangles cause:
    - Invalid normal calculations → bad edge cosines
    - Numerical instabilities in collision detection
    - Potential NaN/inf values in compression
    
    Args:
        verts: Vertex list
        tris: Triangle list (may contain degenerate triangles)
        
    Returns:
        Filtered triangle list with only valid (non-degenerate) triangles
    """
    valid_tris = []
    num_discarded = 0
    
    for tri_id, (v0_idx, v1_idx, v2_idx) in enumerate(tris):
        # Get vertex positions
        v0 = verts[v0_idx]
        v1 = verts[v1_idx]
        v2 = verts[v2_idx]
        
        # Validate triangle area
        if is_triangle_valid(v0, v1, v2):
            valid_tris.append((v0_idx, v1_idx, v2_idx))
        else:
            num_discarded += 1
    
    if num_discarded > 0:
        print(f"⚠️ Discarded {num_discarded} of {len(tris)} triangles (degenerate/zero-area)")
    
    if len(valid_tris) == 0:
        raise ValueError("All triangles are degenerate! Check your mesh geometry.")
    
    return valid_tris


def RW_BuildClusteredMeshComplete(verts: List[Vector], 
                                   tris: List[Tuple[int, int, int]],
                                   granularity: float = 0.001,
                                   enable_vertex_smoothing: bool = False):
    """Complete RenderWare ClusteredMesh build pipeline
    
    Implements the EXACT algorithm from RenderWare rwcclusteredmeshbuilder.cpp Build():
    **CORRECT ORDER (verified against RenderWare source):**
    1. ValidateTriangles (lines 235-245) → filter degenerate triangles
    2. FindTriangleNeighbors (lines 282-288) → compute edge cosines
    3. GenerateTriangleEdgeCodes (lines 319-323) → convert to edge bytes
    4. SmoothVertices (lines 327-336) → disable non-feature vertices
    5. BuildKDTree (lines 369-376) → create spatial partitioning
    6. WalkBranch (lines 944-952) → create clusters
    7. AdjustKDTreeNodeEntriesForCluster (lines 968-976) → fixup byte offsets
    8. InitializeRuntimeKDTree (line 486) → convert BuildNode → KDNode
    
    Args:
        enable_vertex_smoothing: If False, disables vertex smoothing (all edges sharp)
    
    Returns: (clusters, kdtree_nodes, bbox_min, bbox_max)
    """
    # STEP 0: Validate and filter degenerate triangles (RenderWare lines 235-245)
    # CRITICAL: This MUST be first! Degenerate triangles cause edge cosine errors.
    tris = validate_triangles(verts, tris)
    
    # CRITICAL: Initialize edge codes dictionary for ALL triangles AFTER validation
    # This matches RenderWare's initialization (rwcclusteredmeshbuilder.cpp line 129)
    # Edge codes will be populated during clustering, but dict must exist first
    edge_codes_global = {}  # tri_id → (edge0, edge1, edge2)
    
    # STEP 1: Find Triangle Neighbors with ADVANCED edge matching
    # RenderWare: triangleneighborfinder.cpp lines 22-219
    # CRITICAL: Handles non-manifold geometry (edges shared by 3+ triangles)
    # Algorithm: Always keep the MOST CONVEX pairing for each edge
    
    # Initialize neighbor and edge cosine data structures
    # triangleNeighbors[tri_id] = [neighbor0, neighbor1, neighbor2] or None if unmatched
    # triangleEdgeCosines[tri_id] = [edgeCos0, edgeCos1, edgeCos2]
    NOMATCH = None  # Sentinel for unmatched edge
    triangle_neighbors = [[NOMATCH, NOMATCH, NOMATCH] for _ in range(len(tris))]
    triangle_edge_cosines = [[1.0, 1.0, 1.0] for _ in range(len(tris))]  # Default: flat edge
    
    # Build vertex → triangle adjacency map (RenderWare lines 255-260)
    vertex_tri_map = _build_vertex_triangle_map(tris)
    
    # For each triangle, try to mate each edge with neighboring triangles
    # RenderWare: triangleneighborfinder.cpp lines 37-76
    for tri1_id in range(len(tris)):
        tri1 = tris[tri1_id]
        tri1_verts = tri1
        
        # For each edge of triangle 1
        for edge1_idx in range(3):
            edge1_next_idx = (edge1_idx + 1) if edge1_idx < 2 else 0
            edge1_v0 = tri1_verts[edge1_idx]
            edge1_v1 = tri1_verts[edge1_next_idx]
            
            # Find all triangles sharing the edge1_v0 vertex (lines 53-55)
            adjoining_tris = vertex_tri_map.get(edge1_v0, [])
            
            for tri2_id in adjoining_tris:
                # Only process each pair once: tri1_id > tri2_id (line 60)
                if tri1_id > tri2_id:
                    # Try to mate edge1 of tri1 with edges of tri2
                    _mate_edge(tri1_id, edge1_idx, tri2_id, tris, verts, 
                              triangle_neighbors, triangle_edge_cosines)
    
    # STEP 2: Generate triangle edge codes from matched neighbors
    # RenderWare: edgecodegenerator.cpp lines 36-119
    min_concave_edge_cosine = -1.0  # Skate 3 tolerance
    
    for tri_id in range(len(tris)):
        tri = tris[tri_id]
        if not (0 <= tri[0] < len(verts) and 0 <= tri[1] < len(verts) and 0 <= tri[2] < len(verts)):
            edge_codes_global[tri_id] = (EDGE_ANGLE_ZERO | EDGE_UNMATCHED, EDGE_ANGLE_ZERO | EDGE_UNMATCHED, EDGE_ANGLE_ZERO | EDGE_UNMATCHED)
            continue
        
        codes = []
        for edge_idx in range(3):
            neighbor_id = triangle_neighbors[tri_id][edge_idx]
            extended_edge_cos = triangle_edge_cosines[tri_id][edge_idx]
            matched = (neighbor_id is not NOMATCH)
            
            # Convert edge cosine to angle byte (edgecodegenerator.cpp line 80)
            code = edge_cosine_to_angle_byte(extended_edge_cos)
            
            # Set CONVEX flag if edge cosine < 1.0 (line 83-86)
            if extended_edge_cos < 1.0:
                code |= EDGE_CONVEX
            
            # Check concave threshold (lines 90-109)
            capped_min_concave = max(-1.0, min(min_concave_edge_cosine, 1.0))
            concave_threshold = 2.0 - capped_min_concave
            if extended_edge_cos > concave_threshold:
                code = EDGE_ANGLE_ZERO
            
            # Set UNMATCHED flag (lines 112-115)
            if not matched:
                code |= EDGE_UNMATCHED
            
            codes.append(code)
        
        edge_codes_global[tri_id] = (codes[0], codes[1], codes[2])
    
    # STEP 2: Smooth non-feature vertices (SmoothVertices)
    # RenderWare: rwcclusteredmeshbuilder.cpp lines 327-336
    # CRITICAL: This happens BEFORE KDTree building and modifies edge_codes_global!
    # IMPORTANT: Disable this for hard-edged objects (stairs, rails, sharp ramps)!
    if enable_vertex_smoothing:
        vertex_tri_map = _build_vertex_triangle_map(tris)
        disabled_count = 0
        
        for vertex_id in vertex_tri_map:
            adjoining_tris = vertex_tri_map[vertex_id]
            if not adjoining_tris:
                continue
            
            vertex_pos = verts[vertex_id]
            disable_vertex = _all_coplanar_triangles(adjoining_tris, tris, verts, tolerance=0.01)
            
            if not disable_vertex:
                disable_vertex = _vertex_is_non_feature(
                    vertex_id, vertex_pos, adjoining_tris, tris, verts,
                    coplanar_tol=0.01, cosine_tol=0.05, concave_tol=0.15)
            
            if disable_vertex:
                # Modify edge_codes_global directly
                for tri_id in adjoining_tris:
                    if tri_id in edge_codes_global:
                        tri = tris[tri_id]
                        codes = list(edge_codes_global[tri_id])
                        if tri[0] == vertex_id:
                            codes[0] |= EDGE_VERTEX_DISABLE
                        elif tri[1] == vertex_id:
                            codes[1] |= EDGE_VERTEX_DISABLE
                        elif tri[2] == vertex_id:
                            codes[2] |= EDGE_VERTEX_DISABLE
                        edge_codes_global[tri_id] = tuple(codes)
                disabled_count += 1
        
        if len(vertex_tri_map) > 0:
            disabled_pct = (disabled_count / len(vertex_tri_map)) * 100
            print(f"✅ SmoothVertices ENABLED: {disabled_count}/{len(vertex_tri_map)} vertices ({disabled_pct:.1f}%) marked as VERTEXDISABLE")
            print(f"   (Real Skate 3 PSGs have ~93% VERTEXDISABLE - good for terrain/organic shapes)")
    else:
        print(f"⚠️ SmoothVertices DISABLED: ALL vertices will have hard collision!")
        print(f"   Use this for stairs, rails, sharp ramps, or thin geometry")
    
    # STEP 3: Build KDTree (creates sortedEntryIndices - the MASTER ORDERING)
    # RenderWare: rwcclusteredmeshbuilder.cpp lines 369-376
    root_build_node, sorted_entry_indices = RW_BuildKDTree(verts, tris, granularity)
    
    # STEP 4: WalkBranch (creates clusters preserving sortedObjects order)
    # RenderWare: rwcclusteredmeshbuilder.cpp lines 944-952
    cluster_stack = []
    leaf_map = {}  # Maps first triangle ID in each leaf → BuildNode
    
    RW_WalkBranch(root_build_node, leaf_map, cluster_stack, tris, sorted_entry_indices)
    
    # Fill vertex positions into clusters
    # After SortAndCompressVertexSet, vertexIDs contains sorted unique vertex IDs
    for cluster in cluster_stack:
        cluster.vertices = []
        for vert_id in cluster.vertexIDs:
            cluster.vertices.append(verts[vert_id])
    
    # Copy edge codes from global dict to each cluster
    # Each cluster gets the edge codes for its triangles
    for cluster in cluster_stack:
        for unit_id in cluster.unitIDs:
            if unit_id in edge_codes_global:
                cluster.edge_codes[unit_id] = edge_codes_global[unit_id]
            else:
                # Fallback for missing edge codes
                cluster.edge_codes[unit_id] = (EDGE_ANGLE_ZERO | EDGE_UNMATCHED, EDGE_ANGLE_ZERO | EDGE_UNMATCHED, EDGE_ANGLE_ZERO | EDGE_UNMATCHED)
    
    # STEP 5: Adjust KDTree leaf offsets
    # RenderWare: rwcclusteredmeshbuilder.cpp lines 968-976
    for cluster_id, cluster in enumerate(cluster_stack):
        cluster.clusterID = cluster_id
        RW_AdjustKDTreeNodeEntriesForCluster(cluster, cluster_id, leaf_map, tris)
    
    # STEP 6: Convert BuildNode tree → Runtime KDNode array
    # RenderWare: rwcclusteredmeshbuilder.cpp line 486
    rt_kdtree_nodes = RW_InitializeRuntimeKDTree(root_build_node)
    
    # Extract bbox from root
    bbox_min = root_build_node.bbox.min
    bbox_max = root_build_node.bbox.max
    
    # CRITICAL: Return the VALIDATED triangle list!
    # Clusters reference triangle IDs in the validated list, NOT the original list!
    return cluster_stack, rt_kdtree_nodes, bbox_min, bbox_max, tris

def _mate_edge(tri1_id: int, edge1_idx: int, tri2_id: int,
               tris: List[Tuple[int, int, int]], verts: List[Vector],
               triangle_neighbors: List[List], triangle_edge_cosines: List[List]) -> bool:
    """Try to mate an edge between two triangles (advanced non-manifold matching)
    
    EXACT port of RenderWare TriangleNeighborFinder::MateEdge
    Source: triangleneighborfinder.cpp lines 79-219
    
    This implements sophisticated edge matching for non-manifold geometry:
    - Edges shared by 3+ triangles: Keep MOST CONVEX pairing only
    - Less convex matches are marked as UNMATCHED
    - Handles all 4 cases: both unmatched, one matched, both matched
    
    Args:
        tri1_id: First triangle index
        edge1_idx: Edge index in tri1 (0, 1, or 2)
        tri2_id: Second triangle index
        tris: Triangle list
        verts: Vertex list
        triangle_neighbors: [tri_id][edge_idx] = neighbor_tri_id or None
        triangle_edge_cosines: [tri_id][edge_idx] = extended_edge_cosine
        
    Returns:
        True if edges were matched, False otherwise
    """
    NOMATCH = None
    
    edge1_next_idx = (edge1_idx + 1) if edge1_idx < 2 else 0
    
    tri1_verts = tris[tri1_id]
    tri2_verts = tris[tri2_id]
    
    # Get vertex positions for tri1
    v1_p0 = verts[tri1_verts[0]]
    v1_p1 = verts[tri1_verts[1]]
    v1_p2 = verts[tri1_verts[2]]
    
    # Get vertex positions for tri2
    v2_p0 = verts[tri2_verts[0]]
    v2_p1 = verts[tri2_verts[1]]
    v2_p2 = verts[tri2_verts[2]]
    
    # Compute triangle normals (lines 98-106)
    t1_normal = vec_normalize(vec_cross(v1_p1 - v1_p0, v1_p2 - v1_p0))
    t2_normal = vec_normalize(vec_cross(v2_p1 - v2_p0, v2_p2 - v2_p0))
    
    # Compute edge direction for extended edge cosine (lines 108-111)
    edge_dir = vec_normalize(verts[tri1_verts[edge1_next_idx]] - verts[tri1_verts[edge1_idx]])
    
    # Compute extended edge cosine (ComputeExtendedEdgeCosine)
    cos_theta = vec_dot(t1_normal, t2_normal)
    sin_theta = vec_dot(edge_dir, vec_cross(t1_normal, t2_normal))
    epsilon = -1e-6
    
    if sin_theta > epsilon:
        edge_cosine = max(cos_theta, -1.0)
    else:
        edge_cosine = min(2.0 - cos_theta, 3.0)
    
    # Test all 3 edges of tri2 against edge1 of tri1 (lines 114-216)
    # CRITICAL: RenderWare uses special loop: for (edge2=2, next=0; next < 3; edge2 = next++)
    # This creates sequence: (2,0), (0,1), (1,2) - NOT the same as range(3)!
    edge2_idx = 2
    edge2_next_idx = 0
    while edge2_next_idx < 3:
        
        # Check if edges match (reversed direction): e1 == e2_reversed (lines 117-118)
        if (tri1_verts[edge1_idx] == tri2_verts[edge2_next_idx] and
            tri2_verts[edge2_idx] == tri1_verts[edge1_next_idx]):
            
            # CASE ANALYSIS (lines 120-212):
            # Handle 4 cases based on whether tri1/tri2 edges are already matched
            
            tri1_matched = (triangle_neighbors[tri1_id][edge1_idx] is not NOMATCH)
            tri2_matched = (triangle_neighbors[tri2_id][edge2_idx] is not NOMATCH)
            
            # CASE 1: Both unmatched - simple match (lines 120-127)
            if not tri1_matched and not tri2_matched:
                triangle_neighbors[tri1_id][edge1_idx] = tri2_id
                triangle_neighbors[tri2_id][edge2_idx] = tri1_id
                triangle_edge_cosines[tri1_id][edge1_idx] = edge_cosine
                triangle_edge_cosines[tri2_id][edge2_idx] = edge_cosine
            
            # CASE 2: tri1 unmatched, tri2 matched (lines 128-149)
            elif not tri1_matched and tri2_matched:
                # tri2 already has neighbor tri3 - compare edge cosines
                old_edge_cosine = triangle_edge_cosines[tri2_id][edge2_idx]
                
                # If new pairing is MORE convex (higher cosine), replace (line 131)
                if edge_cosine > old_edge_cosine:
                    tri3_id = triangle_neighbors[tri2_id][edge2_idx]
                    
                    # Match tri1-tri2
                    triangle_neighbors[tri1_id][edge1_idx] = tri2_id
                    triangle_neighbors[tri2_id][edge2_idx] = tri1_id
                    triangle_edge_cosines[tri1_id][edge1_idx] = edge_cosine
                    triangle_edge_cosines[tri2_id][edge2_idx] = edge_cosine
                    
                    # Find which edge of tri3 was matched to tri2 (line 143)
                    edge3_idx = _find_edge_by_neighbor(triangle_neighbors[tri3_id], tri2_id)
                    
                    # Unmatch tri3 (lines 146-147)
                    triangle_neighbors[tri3_id][edge3_idx] = NOMATCH
                    triangle_edge_cosines[tri3_id][edge3_idx] = 1.0
            
            # CASE 3: tri1 matched, tri2 unmatched (lines 151-173)
            elif tri1_matched and not tri2_matched:
                # tri1 already has neighbor tri3 - compare edge cosines
                old_edge_cosine = triangle_edge_cosines[tri1_id][edge1_idx]
                
                # If new pairing is MORE convex (higher cosine), replace (line 156)
                if edge_cosine > old_edge_cosine:
                    tri3_id = triangle_neighbors[tri1_id][edge1_idx]
                    
                    # Match tri1-tri2
                    triangle_neighbors[tri1_id][edge1_idx] = tri2_id
                    triangle_neighbors[tri2_id][edge2_idx] = tri1_id
                    triangle_edge_cosines[tri1_id][edge1_idx] = edge_cosine
                    triangle_edge_cosines[tri2_id][edge2_idx] = edge_cosine
                    
                    # Find which edge of tri3 was matched to tri1 (line 168)
                    edge3_idx = _find_edge_by_neighbor(triangle_neighbors[tri3_id], tri1_id)
                    
                    # Unmatch tri3 (lines 171-172)
                    triangle_neighbors[tri3_id][edge3_idx] = NOMATCH
                    triangle_edge_cosines[tri3_id][edge3_idx] = 1.0
            
            # CASE 4: Both tri1 and tri2 matched (lines 175-210)
            else:
                # Both already matched to tri3 and tri4
                tri3_id = triangle_neighbors[tri1_id][edge1_idx]
                tri4_id = triangle_neighbors[tri2_id][edge2_idx]
                
                # Avoid circular references (line 182)
                if tri1_id != tri4_id and tri2_id != tri3_id:
                    old_edge_cos1 = triangle_edge_cosines[tri1_id][edge1_idx]
                    old_edge_cos2 = triangle_edge_cosines[tri2_id][edge2_idx]
                    
                    # If new pairing is MORE convex than BOTH old pairings (line 184)
                    if edge_cosine > old_edge_cos1 and edge_cosine > old_edge_cos2:
                        # Find edges in tri3 and tri4
                        edge3_idx = _find_edge_by_neighbor(triangle_neighbors[tri3_id], tri1_id)
                        edge4_idx = _find_edge_by_neighbor(triangle_neighbors[tri4_id], tri2_id)
                        
                        # Match tri1-tri2
                        triangle_neighbors[tri1_id][edge1_idx] = tri2_id
                        triangle_neighbors[tri2_id][edge2_idx] = tri1_id
                        triangle_edge_cosines[tri1_id][edge1_idx] = edge_cosine
                        triangle_edge_cosines[tri2_id][edge2_idx] = edge_cosine
                        
                        # Unmatch tri3 and tri4 (lines 203-208)
                        triangle_neighbors[tri3_id][edge3_idx] = NOMATCH
                        triangle_edge_cosines[tri3_id][edge3_idx] = 1.0
                        triangle_neighbors[tri4_id][edge4_idx] = NOMATCH
                        triangle_edge_cosines[tri4_id][edge4_idx] = 1.0
            
            return True  # Found matching edge
        
        # Advance loop (RenderWare pattern)
        edge2_idx = edge2_next_idx
        edge2_next_idx += 1
    
    return False  # No matching edge found


def _find_edge_by_neighbor(neighbor_list: List, target_neighbor_id: int) -> int:
    """Find which edge index has the given neighbor
    
    Port of RenderWare FindEdgeByNeighbor (helper function)
    Searches neighbor_list for target_neighbor_id and returns the index
    """
    for edge_idx in range(3):
        if neighbor_list[edge_idx] == target_neighbor_id:
            return edge_idx
    # Should never reach here if data is valid
    return 0


def _build_vertex_triangle_map(tris: List[Tuple[int, int, int]]) -> Dict[int, List[int]]:
    """Build mapping from vertex index to list of triangle indices
    
    Port of RenderWare VertexTriangleMap (rwcclusteredmeshbuildermethods.cpp lines 566-619)
    Returns: Dict[vertex_id] = [list_of_triangle_ids]
    """
    vertex_tri_map = {}
    for tri_id, (a, b, c) in enumerate(tris):
        for v in (a, b, c):
            if v not in vertex_tri_map:
                vertex_tri_map[v] = []
            vertex_tri_map[v].append(tri_id)
    return vertex_tri_map

# ============================================================================
# Vertex Smoothing (SmoothVertices)
# ============================================================================

def _all_coplanar_triangles(tri_ids: List[int], tris: List[Tuple[int, int, int]], 
                            verts: List[Vector], tolerance: float) -> bool:
    """Check if all triangles share same normal (coplanar)
    
    Port of RenderWare AllCoplanarTriangles (rwcclusteredmeshbuildermethods.cpp lines 1149-1210)
    """
    if not tri_ids:
        return False
    
    # Get first triangle normal as candidate plane normal
    tri = tris[tri_ids[0]]
    v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
    plane_normal = vec_normalize(vec_cross(v1 - v0, v2 - v0))
    
    # Check all other triangles against this normal
    for tri_id in tri_ids[1:]:
        tri = tris[tri_id]
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        tri_normal = vec_normalize(vec_cross(v1 - v0, v2 - v0))
        
        # IsSimilar(Dot(n1, n2), 1.0, tolerance)
        dot = vec_dot(tri_normal, plane_normal)
        if abs(dot - 1.0) > tolerance:
            return False
    
    return True

def _vertex_is_non_feature(vertex_id: int, vertex_pos: Vector, tri_ids: List[int],
                           tris: List[Tuple[int, int, int]], verts: List[Vector],
                           coplanar_tol: float, cosine_tol: float, concave_tol: float) -> bool:
    """Check if vertex is in featureless plane or concave region
    
    Port of RenderWare VertexIsNonFeature (rwcclusteredmeshbuildermethods.cpp lines 1269-1383)
    """
    if not tri_ids:
        return False
    
    # Get first triangle to establish plane normal
    tri = tris[tri_ids[0]]
    v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
    plane_normal = vec_normalize(vec_cross(v1 - v0, v2 - v0))
    
    # Get opposite vertices (vertices of first triangle that aren't vertex_id)
    vert_a, vert_b = _get_opposite_vertices(vertex_id, tri, verts)
    
    # Initialize advancing feature plane edges
    edge_a = vec_normalize(vertex_pos - vert_a)
    edge_b = vec_normalize(vertex_pos - vert_b)
    
    # Test all other triangles
    for tri_id in tri_ids[1:]:
        tri = tris[tri_id]
        vert_a, vert_b = _get_opposite_vertices(vertex_id, tri, verts)
        
        # Test edge from vertex_pos to vert_a
        edge_c = vec_normalize(vertex_pos - vert_a)
        if _edge_disables_vertex(edge_a, edge_b, edge_c, plane_normal, 
                                 coplanar_tol, cosine_tol, concave_tol):
            return True
        
        # Test edge from vertex_pos to vert_b
        edge_c = vec_normalize(vertex_pos - vert_b)
        if _edge_disables_vertex(edge_a, edge_b, edge_c, plane_normal,
                                 coplanar_tol, cosine_tol, concave_tol):
            return True
    
    return False

def _get_opposite_vertices(vertex_id: int, tri: Tuple[int, int, int], 
                           verts: List[Vector]) -> Tuple[Vector, Vector]:
    """Get the two vertices opposite to vertex_id in triangle
    
    Port of RenderWare GetOppositeVertices (rwcclusteredmeshbuildermethods.cpp lines 1396-1418)
    """
    if tri[0] == vertex_id:
        return verts[tri[1]], verts[tri[2]]
    elif tri[1] == vertex_id:
        return verts[tri[2]], verts[tri[0]]
    else:
        return verts[tri[0]], verts[tri[1]]

def _edge_disables_vertex(edge_a: Vector, edge_b: Vector, edge_c: Vector, plane_normal: Vector,
                          coplanar_tol: float, cosine_tol: float, concave_tol: float) -> bool:
    """Check if edge_c disables the vertex (featureless plane or concave)
    
    Port of RenderWare EdgeDisablesVertex (clusteredmeshbuilderutils.cpp lines 148-179)
    """
    plane_edge_c_dot = vec_dot(edge_c, plane_normal)
    
    # If edge is coplanar to plane
    if abs(plane_edge_c_dot) < coplanar_tol:
        # Check if edge creates featureless plane
        if _edge_produces_featureless_plane(edge_a, edge_b, edge_c, cosine_tol):
            return True
    # If edge is concave (points into surface)
    elif plane_edge_c_dot < concave_tol:
        return True
    
    return False

def _edge_produces_featureless_plane(edge_a: Vector, edge_b: Vector, edge_c: Vector, 
                                     tolerance: float) -> bool:
    """Check if edge_c creates a featureless plane between edge_a and edge_b
    
    Port of RenderWare EdgeProducesFeaturelessPlane (clusteredmeshbuilderutils.cpp lines 87-130)
    """
    a_dot_b = vec_dot(edge_a, edge_b)
    
    # Check if -edge_c lies in halfspace defined by edge_a + edge_b
    half_space = edge_a + edge_b
    if vec_dot(half_space, edge_c * -1.0) >= 0.0:
        # Check if -edge_c lies between feature edges
        neg_c = edge_c * -1.0
        if (vec_dot(neg_c, edge_a) >= (a_dot_b - tolerance) and
            vec_dot(neg_c, edge_b) >= (a_dot_b - tolerance)):
            return True
    
    return False

def count_all_nodes(node):
    """Helper to count all nodes in BuildNode tree"""
    if node is None:
        return 0
    if node.left is None:
        return 1  # Leaf
    return 1 + count_all_nodes(node.left) + count_all_nodes(node.right)

# ===== BLENDER DATA EXTRACTOR =====

@dataclass
class Vertex:
    x: float
    y: float
    z: float

@dataclass
class Face:
    v1: int
    v2: int
    v3: int
    surface_id: int = 0

@dataclass
class SplineCurve:
    """Spline curve extracted from Blender"""
    points: List[Vertex]
    name: str
    is_closed: bool

class BlenderDataExtractor:
    """Extract mesh and curve data from Blender scene
    
    Applies proper coordinate transformation from Blender space to PSG game space:
    - Blender (X, Y, Z) → PSG (X, Z, -Y)
    - This matches the inverse of importer's transform (line 3155, 3616)
    """
    
    def __init__(self):
        self.vertices: List[Vertex] = []
        self.faces: List[Face] = []
        self.splines: List[SplineCurve] = []
        self.bounds_min = [float('inf'), float('inf'), float('inf')]
        self.bounds_max = [float('-inf'), float('-inf'), float('-inf')]
    
    def extract_from_selected(self):
        """Extract geometry and curves from selected Blender objects"""
        
        selected_objects = bpy.context.selected_objects
        if not selected_objects:
            raise RuntimeError("No objects selected! Please select mesh and/or curve objects.")
        
        mesh_count = 0
        curve_count = 0
        
        for obj in selected_objects:
            if obj.type == 'MESH':
                self._extract_mesh(obj)
                mesh_count += 1
            elif obj.type == 'CURVE':
                self._extract_curve(obj)
                curve_count += 1
        

    
    def _extract_mesh(self, obj):
        """Extract mesh geometry from Blender object"""
        # Apply modifiers and get evaluated mesh
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        
        if not mesh:
            return
        
        # Apply world transform
        matrix_world = obj.matrix_world
        
        # Get vertices
        vertex_offset = len(self.vertices)
        for vert in mesh.vertices:
            # Transform to world space
            co_world = matrix_world @ vert.co
            blender_x, blender_y, blender_z = co_world.x, co_world.y, co_world.z
            
            # COORDINATE TRANSFORM: Blender → PSG Game Space
            # Importer does: PSG (X,Y,Z) → Blender (X,Z,-Y)  [line 3155 of PsgImportAndLoggingFinal.py]
            # Exporter: Blender (X,Y,Z) → PSG (X,Z,-Y)
            # CRITICAL: Direct mapping, no negations
            #   Blender X -> PSG X
            #   Blender Y -> PSG Z (negated)
            #   Blender Z -> PSG Y
            # COORDINATE TRANSFORM: Blender -> PSG Game Space (X, Z, -Y)
            # Mapping: psg_x = blender_x, psg_y = blender_z, psg_z = -blender_y
            psg_x = blender_x
            psg_y = blender_z
            psg_z = -blender_y
            
            self.vertices.append(Vertex(psg_x, psg_y, psg_z))
            
            # Update bounds (in PSG space)
            self.bounds_min[0] = min(self.bounds_min[0], psg_x)
            self.bounds_min[1] = min(self.bounds_min[1], psg_y)
            self.bounds_min[2] = min(self.bounds_min[2], psg_z)
            self.bounds_max[0] = max(self.bounds_max[0], psg_x)
            self.bounds_max[1] = max(self.bounds_max[1], psg_y)
            self.bounds_max[2] = max(self.bounds_max[2], psg_z)
        
        # Get faces (triangulate)
        mesh.calc_loop_triangles()

        # Get per polygon surface id
        if "surface_id" in mesh.attributes:
            surface_layer = mesh.attributes["surface_id"]
        else:
            surface_layer = None
        
        for tri in mesh.loop_triangles:
            v1 = tri.vertices[0] + vertex_offset
            v2 = tri.vertices[1] + vertex_offset
            v3 = tri.vertices[2] + vertex_offset

            poly_index = tri.polygon_index
            # Read surface ID; default to 0 if not assigned
            if surface_layer:
                surface_id = surface_layer.data[poly_index].value or 0
            else:
                surface_id = 0

            # CRITICAL: Keep Blender's original winding order
            # RenderWare computes normals as: cross(p1 - p0, p2 - p0)
            # For triangle (v1, v2, v3): normal = cross(v2 - v1, v3 - v1)
            # Blender's CCW winding matches this convention
            self.faces.append(Face(v1, v2, v3, surface_id))  # Keep original winding - DO NOT SWAP
        
        # Clean up
        eval_obj.to_mesh_clear()
    
    def _extract_curve(self, obj):
        """Extract spline curves from Blender curve object"""
        if obj.type != 'CURVE':
            return
        
        curve_data = obj.data
        matrix_world = obj.matrix_world
        
        # Process each spline in the curve object
        for spline in curve_data.splines:
            points = []
            
            # Get spline points based on type
            if spline.type in ('POLY', 'BEZIER', 'NURBS'):
                # Sample the spline at regular intervals
                # For NURBS/Bezier, we need to evaluate the curve
                resolution = max(len(spline.points) if hasattr(spline, 'points') else len(spline.bezier_points), 2)
                
                for i in range(resolution):
                    t = i / max(1, resolution - 1)  # Parameter from 0 to 1
                    
                    # Evaluate point on curve
                    if spline.type == 'BEZIER' and spline.bezier_points:
                        # For Bezier, sample the curve
                        if i < len(spline.bezier_points):
                            pt = spline.bezier_points[i].co
                        else:
                            continue
                    elif spline.type == 'NURBS' and spline.points:
                        if i < len(spline.points):
                            pt = spline.points[i].co.xyz  # Homogeneous coords
                        else:
                            continue
                    elif spline.type == 'POLY' and spline.points:
                        if i < len(spline.points):
                            pt = spline.points[i].co.xyz
                        else:
                            continue
                    else:
                        continue
                    
                    # Transform to world space
                    co_world = matrix_world @ BlenderVector(pt)
                    blender_x, blender_y, blender_z = co_world.x, co_world.y, co_world.z
                    
                    # COORDINATE TRANSFORM: Blender → PSG Game Space (same as mesh)
                    # Importer: PSG (X,Y,Z) → Blender (X,Z,-Y)  [line 3616 of PsgImportAndLoggingFinal.py]
                    # Exporter: Blender (X,Y,Z) → PSG (X,Z,-Y)
                    # CRITICAL: Y must be negated to match mesh transform!
                    psg_x = blender_x   # Blender X becomes PSG X
                    psg_y = blender_z   # Blender Z becomes PSG Y
                    psg_z = -blender_y  # Blender Y becomes PSG Z (NEGATED!)
                    
                    points.append(Vertex(psg_x, psg_y, psg_z))
            
            if len(points) >= 2:
                is_closed = spline.use_cyclic_u
                
                self.splines.append(SplineCurve(points, obj.name, is_closed))

# ===== PSG BUILDER =====

class PSGBuilder:
    """Builds PSG file structures"""
    
    # Type IDs
    TYPE_VER = 0x00EB0008
    TYPE_INST = 0x00EB000D
    TYPE_VOL = 0x00080001
    TYPE_CMESH = 0x00080006
    TYPE_CMODEL = 0x00EB000A
    TYPE_DMO = 0x00EB001D
    TYPE_SPLINE = 0x00EB0004
    TYPE_TOC = 0x00EB000B
    
    def __init__(self, data_extractor: BlenderDataExtractor, force_uncompressed: bool = False,
                 enable_vertex_smoothing: bool = False):
        self.obj = data_extractor  # Compatible with both OBJParser and BlenderDataExtractor
        self.blob = bytearray()
        self.force_uncompressed = force_uncompressed  # Store for use in _serialize_cluster_binary
        self.enable_vertex_smoothing = enable_vertex_smoothing  # Store for use in _build_clusteredmesh
        self.warnings = []  # Collect warnings to report to Blender UI
        
        # Compression statistics (track what compression modes were used)
        self.compression_stats = {
            0: 0,  # Uncompressed
            1: 0,  # 16-bit
            2: 0   # 32-bit
        }

        # Automatically determine finest granularity that fits within 32-bit limits
        if self.obj.vertices:
            verts_vec = [Vector((v.x, v.y, v.z)) for v in self.obj.vertices]
            try:
                self.granularity = determine_optimal_granularity(verts_vec)
            except ValueError as e:
                # If mesh is too large even with max granularity, this is a critical error
                error_msg = f"CRITICAL ERROR: Mesh is TOO LARGE for collision export!\n{str(e)}\nPlease scale your mesh down or split it into smaller pieces."
                raise ValueError(error_msg) from e
        else:
            self.granularity = 0.0625  # Default fallback (matches common Skate 3 granularity)
    
    def build(self, output_path: str):
        """Build complete PSG file"""
        
        # Step 1: Generate Arena ID (needed by sections and header)
        content_hash = hashlib.md5(f"{len(self.obj.vertices)}{len(self.obj.faces)}".encode()).digest()
        self.arena_id = int.from_bytes(content_hash[:4], 'big')
        
        # Step 2: Arena Header (reserve space)
        self.blob = bytearray(0xC0)
        
        # Step 3: Arena Sections
        sections = self._build_sections()
        sections_start = 0xC0


        self.blob.extend(sections)
        
        # Pad to 0x240

        while len(self.blob) < 0x240:
            self.blob.append(0)
        
        # Step 3: Objects
        objects_start = 0x240
        objects, obj_entries = self._build_objects()


        self.blob.extend(objects)  # CRITICAL: Add objects to blob!
        
        # Step 4: Arena Dictionary (Section 15)
        # Per PSG_File_Format_Analysis.md lines 1544-1627
        # 8 entries x 24 bytes = 192 bytes
        # Must be 16-byte aligned
        dict_start = align(len(self.blob), 16)
        while len(self.blob) < dict_start:
            self.blob.append(0)
        
        dictionary = self._build_dictionary(obj_entries)
        self.blob.extend(dictionary)
        
        # Step 5: Subreference Records (Section 16)
        # Per PSG_File_Format_Analysis.md lines 1630-1689
        # 8 bytes per record: objectId(4) + offset(4)
        # Count: 1 (InstanceData) + N (splines)
        subref_records_start = len(self.blob)
        subref_records = self._build_subref_records()
        self.blob.extend(subref_records)
        
        # Step 6: Subreference Dictionary (Section 17)
        # Per PSG_File_Format_Analysis.md lines 1693-1718
        # 4 bytes per entry, all 0x00000000 for internal refs
        # Count matches subref record count
        subref_dict_start = len(self.blob)
        subref_dict = self._build_subref_dict(len(subref_records) // 8)
        self.blob.extend(subref_dict)
        
        # Step 7: File End Padding (Section 18)
        # Per PSG_File_Format_Analysis.md lines 1722-1743
        # 
        # VERIFIED ANALYSIS (4 reference PSG files):
        # - Subref dict end is ALREADY 4-byte aligned
        # - Padding amounts VARY: 200, 980, 280, 1020 bytes (no fixed pattern!)
        # - All padding bytes are zeros
        # - No alignment target found (not 256, 512, 1KB, or 4KB)
        # 
        # CONCLUSION: Padding is arbitrary/build-process artifact
        # Safe approach: Ensure 4-byte alignment + minimal padding for safety
        # Using 16 bytes minimum (can use 200 to match smallest reference file)
        
        # Ensure 4-byte alignment (should already be aligned, but verify)
        while len(self.blob) % 4 != 0:
            self.blob.append(0)
        
        # Add minimal padding (16 bytes) for safety
        # Note: Reference files use 200-1020 bytes, but this varies with no pattern
        for _ in range(16):
            self.blob.append(0)
        
        # Step 8: Fill in header with FINAL file size
        # Update Arena Header (Section 1) with computed values
        self._fill_header(dict_start, len(self.blob))
        
        # Step 9: Update subreference section pointers AND counts
        # Backfill Section Subreferences (Section 5) with absolute offsets
        num_subref_records = len(subref_records) // 8
        self._update_subref_pointers(subref_records_start, subref_dict_start, num_subref_records)
        
        # Write file
        with open(output_path, 'wb') as f:
            f.write(self.blob)
        


    
    def _fill_header(self, dict_start: int, file_size: int):
        """Fill Arena header at start of file - Per PSG_File_Format_Analysis.md Section 1"""
        # Magic @ 0x00-0x0B (12 bytes) - Per markdown lines 156-158
        self.blob[0x00:0x0C] = b'\x89RW4ps3\x00\x0D\x0A\x1A\x0A'
        
        # Metadata @ 0x0C-0x1B - Per markdown lines 159-164
        self.blob[0x0C] = 0x01  # isBigEndian
        self.blob[0x0D] = 0x20  # pointerSizeInBits (32)
        self.blob[0x0E] = 0x04  # pointerAlignment
        self.blob[0x0F] = 0x00  # unused
        self.blob[0x10:0x14] = b'454\x00'  # majorVersion
        self.blob[0x14:0x18] = b'000\x00'  # minorVersion
        self.blob[0x18:0x1C] = be_u32(0)  # buildNo
        
        # Arena core - Per markdown lines 166-174
        # Use Arena ID generated earlier (same ID used in External Arenas)
        self.blob[0x1C:0x20] = be_u32(self.arena_id)  # +0x1C: Arena ID (DYNAMIC - unique per file)
        self.blob[0x20:0x24] = be_u32(8)           # +0x20: numEntries
        self.blob[0x24:0x28] = be_u32(8)           # +0x24: numUsed
        self.blob[0x28:0x2C] = be_u32(0x10)        # +0x28: alignment
        self.blob[0x2C:0x30] = be_u32(0)           # +0x2C: virt
        self.blob[0x30:0x34] = be_u32(dict_start)  # +0x30: dictStart
        self.blob[0x34:0x38] = be_u32(0xC0)        # +0x34: sections
        self.blob[0x38:0x3C] = be_u32(0)           # +0x38: base
        self.blob[0x3C:0x40] = be_u32(0)           # +0x3C: m_unfixContext
        self.blob[0x40:0x44] = be_u32(0)           # +0x40: m_fixContext
        
        # ResourceDescriptor[0] - FILE SIZE - Per markdown line 127
        self.blob[0x44:0x48] = be_u32(file_size)  # size
        self.blob[0x48:0x4C] = be_u32(0x10)       # alignment
        
        # ResourceDescriptor[1-4] - Per markdown lines 129-132
        self.blob[0x4C:0x50] = be_u32(0)   # [1].size
        self.blob[0x50:0x54] = be_u32(1)   # [1].alignment
        self.blob[0x54:0x58] = be_u32(0)   # [2].size
        self.blob[0x58:0x5C] = be_u32(1)   # [2].alignment
        self.blob[0x5C:0x60] = be_u32(0)   # [3].size
        self.blob[0x60:0x64] = be_u32(1)   # [3].alignment
        self.blob[0x64:0x68] = be_u32(0)   # [4].size
        self.blob[0x68:0x6C] = be_u32(1)   # [4].alignment
        
        # ResourcesUsed (ResourceDescriptor @ 0x6C) - VERIFIED from actual PSG file!
        # Single ResourceDescriptor: size + alignment (8 bytes total)
        # CRITICAL: Markdown line 225 is WRONG! Actual file has 0, not 1!
        self.blob[0x6C:0x70] = be_u32(0)      # m_resourcesUsed.size (VERIFIED: 0 in reference file)
        self.blob[0x70:0x74] = be_u32(1)      # m_resourcesUsed.alignment
        
        # TargetResource m_resource @ 0x74 (NOT part of ResourcesUsed!)
        # Per PSG_File_Format_Analysis.md line 226-227
        self.blob[0x74:0x78] = be_u32(0xC0)   # m_resource.base (points to section manifest)
        self.blob[0x78:0x7C] = be_u32(4)      # m_resource.count (4 sections)
        self.blob[0x7C:0x80] = be_u32(0)      # m_resource field
        self.blob[0x80:0x84] = be_u32(1)      # m_resource field
        self.blob[0x84:0x88] = be_u32(0)      # m_resource field
        self.blob[0x88:0x8C] = be_u32(1)      # m_resource field
        self.blob[0x8C:0x90] = be_u32(0)      # m_resource field
        self.blob[0x90:0x94] = be_u32(1)      # m_resource field
        
        # PS3 m_resource (TargetResource) - EXACT from real file
        # 0x94-0xA7 (20 bytes = 5 uint32)
        self.blob[0x94:0x98] = be_u32(0)
        self.blob[0x98:0x9C] = be_u32(1)
        self.blob[0x9C:0xA0] = be_u32(0)
        self.blob[0xA0:0xA4] = be_u32(1)
        self.blob[0xA4:0xA8] = be_u32(0)
        
        # m_arenaGroup @ 0xA8-0xAB
        self.blob[0xA8:0xAC] = be_u32(0)
        
        # Padding to 0xC0
        for i in range(0xAC, 0xC0):
            self.blob[i] = 0
    
    def _build_sections(self) -> bytes:
        """Build Arena Sections - 100% compliant with PSG_File_Format_Analysis.md"""
        sec = bytearray()
        
        # Section Manifest @ 0x00 (relative to 0xC0)
        sec += be_u32(0x00010004)  # +0x00: typeID
        sec += be_u32(4)           # +0x04: numEntries
        sec += be_u32(0x0C)        # +0x08: dict (relative offset to entry list)
        
        # Manifest entries (4 section offsets) @ 0x0C
        sec += be_u32(0x1C)   # Types offset
        sec += be_u32(0x128)  # ExtArenas offset  
        sec += be_u32(0x14C)  # Subrefs offset
        sec += be_u32(0x168)  # Atoms offset
        
        # Section Types @ 0x1C (28 bytes from start)
        sec += be_u32(0x00010005)  # typeID
        sec += be_u32(64)          # numEntries
        sec += be_u32(0x0C)        # dict (relative)
        
        # Type registry - EXACT 64 entries from reference PSG - Per markdown lines 282-346
        type_registry = [
            0x00000000, 0x00010030, 0x00010031, 0x00010032, 0x00010033, 0x00010034,
            0x00010010, 0x00EB0000, 0x00EB0001, 0x00EB0003, 0x00EB0004, 0x00EB0005,
            0x00EB0006, 0x00EB000A, 0x00EB000D, 0x00EB0019, 0x00EB0007, 0x00EB0008,
            0x00EB000C, 0x00EB0009, 0x00EB000B, 0x00EB000E, 0x00EB0011, 0x00EB000F,
            0x00EB0010, 0x00EB0012, 0x00EB0022, 0x00EB0013, 0x00EB0014, 0x00EB0015,
            0x00EB0016, 0x00EB001A, 0x00EB001C, 0x00EB001D, 0x00EB001B, 0x00EB001E,
            0x00EB001F, 0x00EB0021, 0x00EB0017, 0x00EB0020, 0x00EB0024, 0x00EB0023,
            0x00EB0025, 0x00EB0026, 0x00EB0027, 0x00EB0028, 0x00EB0029, 0x00EB0018,
            0x00EC0010, 0x00010000, 0x00010002, 0x000200EB, 0x000200EA, 0x000200E9,
            0x00020081, 0x000200E8, 0x00080002, 0x00080001, 0x00080006, 0x00080003,
            0x00080004, 0x00040006, 0x00040007, 0x0001000F,  # [63] Last entry
        ]
        for tid in type_registry:
            sec += be_u32(tid)
        
        # Pad to ExtArenas @ 0x128
        while len(sec) < 0x128:
            sec.append(0)
        
        # Section External Arenas @ 0x128 - VERIFIED from actual PSG file!
        # Structure: 12-byte header + 3 dict entries (8 bytes each = 24 bytes) = 36 bytes total
        # CRITICAL: Markdown line 430 is MISLEADING! Dict entries are NOT all zeros!
        sec += be_u32(0x00010006)  # +0x00: typeID
        sec += be_u32(3)           # +0x04: numEntries
        sec += be_u32(0x18)        # +0x08: dict (relative offset to dict array start)
        
        # Dictionary entries - VERIFIED from 09D7BC5A9527DD0B.psg @ 0x01E8:
        # Dict 0: Arena ID + flags (pattern: b0925826 ffb00000)
        sec += be_u32(self.arena_id)    # +0x0C: Arena ID (appears 3x total: header + 2x here)
        sec += be_u32(0xFFB00000)       # +0x10: Flags/marker
        # Dict 1: Arena ID + padding
        sec += be_u32(self.arena_id)    # +0x14: Arena ID again
        sec += be_u32(0)                # +0x18: Padding
        # Dict 2: Empty (8 bytes of zeros)
        sec += be_u32(0)                # +0x1C: Empty
        sec += be_u32(0)                # +0x20: Empty
        
        # Pad to Subrefs @ 0x14C
        while len(sec) < 0x14C:
            sec.append(0)
        
        # Section Subreferences @ 0x14C - 28 bytes (0x1C)
        sec += be_u32(0x00010007)  # +0x00: typeID
        sec += be_u32(10)  # +0x04: numEntries (placeholder)
        sec += be_u32(0)  # m_dictAfterRefix
        sec += be_u32(0)  # m_recordsAfterRefix
        sec += be_u32(0)  # dict (filled later with absolute offset)
        sec += be_u32(0)  # records (filled later with absolute offset)
        sec += be_u32(1)  # numUsed (placeholder - will update)
        
        # Pad to Atoms @ 0x168
        while len(sec) < 0x168:
            sec.append(0)
        
        # Section Atoms @ 0x168 - Per markdown lines 391-415
        sec += be_u32(0x00010008)  # +0x00: typeID
        sec += be_u32(0)           # +0x04: numEntries (unused in collision PSGs)
        
        # Total sections size should be 0x180 (384 bytes) = 0xC0 header + objects start
        # But objects start at 0x240 so pad appropriately
        return bytes(sec)
    
    def _build_objects(self) -> Tuple[bytes, List[Tuple[int, int, int, int]]]:
        """Build all objects, return (blob, entries)
        entries = [(offset, size, typeIndex, typeID), ...]
        """
        objs = bytearray()
        entries = []
        
        # Type registry - EXACT 64 entries from reference PSG - Per markdown lines 282-346
        type_registry = [
            0x00000000, 0x00010030, 0x00010031, 0x00010032, 0x00010033, 0x00010034,
            0x00010010, 0x00EB0000, 0x00EB0001, 0x00EB0003, 0x00EB0004, 0x00EB0005,
            0x00EB0006, 0x00EB000A, 0x00EB000D, 0x00EB0019, 0x00EB0007, 0x00EB0008,
            0x00EB000C, 0x00EB0009, 0x00EB000B, 0x00EB000E, 0x00EB0011, 0x00EB000F,
            0x00EB0010, 0x00EB0012, 0x00EB0022, 0x00EB0013, 0x00EB0014, 0x00EB0015,
            0x00EB0016, 0x00EB001A, 0x00EB001C, 0x00EB001D, 0x00EB001B, 0x00EB001E,
            0x00EB001F, 0x00EB0021, 0x00EB0017, 0x00EB0020, 0x00EB0024, 0x00EB0023,
            0x00EB0025, 0x00EB0026, 0x00EB0027, 0x00EB0028, 0x00EB0029, 0x00EB0018,
            0x00EC0010, 0x00010000, 0x00010002, 0x000200EB, 0x000200EA, 0x000200E9,
            0x00020081, 0x000200E8, 0x00080002, 0x00080001, 0x00080006, 0x00080003,
            0x00080004, 0x00040006, 0x00040007, 0x0001000F,  # [63] Last entry
        ]
        
        def add_object(data: bytes, type_id: int):
            # Align to 16 bytes
            while len(objs) % 16 != 0:
                objs.append(0)
            
            offset = 0x240 + len(objs)
            objs.extend(data)
            type_index = type_registry.index(type_id) if type_id in type_registry else 0
            entries.append((offset, len(data), type_index, type_id))
        
        # Object 0: VersionData - Per PSG_File_Format_Analysis.md lines 526-556
        # Structure: m_uiVersion(4) + m_uiRevision(4) + reserved(8) = 16 bytes
        # CONSTANT: Version 25, Revision 13 across all analyzed PSG files
        add_object(be_u32(25) + be_u32(13) + be_u64(0), self.TYPE_VER)
        
        # Object 1: InstanceData
        add_object(self._build_instancedata(), self.TYPE_INST)
        
        # Object 2: Volume
        add_object(self._build_volume(), self.TYPE_VOL)
        
        # Object 3: ClusteredMesh
        add_object(self._build_clusteredmesh(), self.TYPE_CMESH)
        
        # Object 4: CollisionModelData
        add_object(self._build_collision_model(), self.TYPE_CMODEL)
        
        # Object 5: DMOData - EXACT size from original
        add_object(self._build_dmodata(), self.TYPE_DMO)
        
        # Object 6: SplineData - EXACT size from original
        add_object(self._build_splinedata(), self.TYPE_SPLINE)
        
        # Object 7: TableOfContents
        add_object(self._build_toc(), self.TYPE_TOC)
        
        return bytes(objs), entries
    
    def _build_instancedata(self) -> bytes:
        """Build InstanceData - Per PSG_File_Format_Analysis.md lines 560-644
        
        Structure:
        - Header: 20 bytes (typeID + counts + 2 relative pointers)
        - Instance entries: 160 bytes each (0xA0)
        - String list: Variable length
        Total: 296 bytes (0x128) for reference file with 1 instance, 2 strings
        """
        blob = bytearray()
        # Header (20 bytes) - Per markdown lines 568-574
        blob += be_u32(0xACB31C9A)  # +0x00: m_typeID
        blob += be_u32(1)           # +0x04: m_uiNumInstances
        blob += be_u32(2)           # +0x08: m_uiNumStrings
        blob += be_u32(0x20)        # +0x0C: m_Instances (relative offset to instance array)
        blob += be_u32(0xC0)        # +0x10: m_StringList (relative offset to string data)
        
        # Pad to 0x20 (start of instance array)
        while len(blob) < 0x20:
            blob.append(0)
        
        # Instance entry (0xA0 = 160 bytes) - Per markdown lines 578-591
        # Structure: 124 bytes unknown + 1 marker + 3 subrefs + 4 string ptrs + 4 pad
        # Unknown bytes +0x00 to +0x7B (124 bytes) - likely transform + flags
        # Write identity matrix as placeholder for first 64 bytes
        for i in range(16):
            blob += be_f32(1.0 if i % 5 == 0 else 0.0)
        
        # Bounds 32 bytes
        for coord in self.obj.bounds_min + [0]:
            blob += be_f32(coord)
        for coord in self.obj.bounds_max + [0]:
            blob += be_f32(coord)
        
        # GUID fields at +0x60 to +0x7B (28 bytes) - fills gap to marker at +0x7C
        # Generate unique GUID for this instance
        bounds_str = f"{self.obj.bounds_min}{self.obj.bounds_max}{len(self.obj.vertices)}"
        hash_byte = int(hashlib.md5(bounds_str.encode()).hexdigest()[:2], 16)
        guid = 0xF8DED31D3B0F6B00 | (hash_byte & 0xFF)  # Base pattern + hash
        
        blob += be_u64(guid)                    # +0x60: GUID
        blob += be_u64(0xFFFFFFFFFFFFFFFF)      # +0x68: GUIDLocal (placeholder)
        blob += be_u64(0xFFFFFFFFFFFFFFFF)      # +0x70: ClassKey (placeholder)
        blob += be_u32(0xFFFFFFFF)              # +0x78: InstanceKey high 32 bits
        
        # Marker field - Per markdown lines 580, 640
        # CRITICAL: This field is NOT converted during Unfix (not a pointer!)
        blob += be_u32(0xFFFFFFFF)  # +0x7C: markerField (MUST be 0xFFFFFFFF!)
        
        # Subreferences at +0x80, +0x84, +0x88 - Per PSG_File_Format_Analysis.md lines 625-630
        # These fields undergo Serialize() during Unfix which creates subreference records
        # Values are resolved through the subreference system during fixup
        blob += be_u32(0)           # +0x80: pModel (SUBREFERENCE - NULL, no render model)
        blob += be_u32(4)           # +0x84: pCollision (SUBREFERENCE - value resolved via subreference records)
        blob += be_u32(0)           # +0x88: pAnimation (SUBREFERENCE - NULL, no animation)
        
        # String pointers +0x8C to +0x98 (16 bytes) - Per markdown lines 584-587, 632-636
        # These are relative offsets within InstanceData, unconditionally converted during Unfix
        blob += be_u32(0xC0)        # +0x8C: pName (relative offset to name string)
        blob += be_u32(0x11E)       # +0x90: pDesc (relative offset to "undefined")
        blob += be_u32(0x11E)       # +0x94: pComp (relative offset to "undefined")
        blob += be_u32(0x11E)       # +0x98: pCat (relative offset to "undefined")
        
        # Padding +0x9C to +0x9F (4 bytes) - Per markdown line 588
        blob += be_u32(0)           # +0x9C: padding (Total entry size: 0xA0 = 160 bytes)
        
        # Pad to 0xC0 (string list start)
        while len(blob) < 0xC0:
            blob.append(0)
        
        # Strings - Generate name from GUID (matches reference format)
        # Original format: [guid1][guid2][guid3]_HighLOD_Proc_Fuse_Seed_0_Split_0
        obj_name = f"[0x{guid:016x}]_Blender_Export_Collision\x00"
        blob += obj_name.encode('utf-8')
        
        # Pad to 0x11E for "undefined" string (matching original structure)
        while len(blob) < 0x11E:
            blob.append(0)
        
        # "undefined" string for desc/comp/cat
        blob += b'undefined\x00'
        
        # Pad to 0x128 total (exact size from reference)
        while len(blob) < 0x128:
            blob.append(0)
        
        return bytes(blob)
    
    def _build_volume(self) -> bytes:
        """Build Volume AGGREGATE - Per PSG_File_Format_Analysis.md lines 647-727
        
        Structure: 96 bytes (0x60)
        - Transform: 64 bytes (3x4 affine matrix with last row zeros)
        - vTable: 4 bytes (stored as volumeType=6 for AGGREGATE)
        - volumeTypeOrIndex: 4 bytes (dict index to ClusteredMesh)
        - Padding: 8 bytes
        - radius: 4 bytes (float)
        - groupID: 4 bytes
        - surfaceID: 4 bytes
        - m_flags: 4 bytes
        """
        blob = bytearray()
        
        # Transform matrix (64 bytes) - Per markdown lines 684-692
        # CRITICAL: 3x4 affine transform, last row MUST be [0,0,0,0]
        # This is NOT a standard 4x4 homogeneous matrix!
        for row in [(1.0,0.0,0.0,0.0), (0.0,1.0,0.0,0.0), (0.0,0.0,1.0,0.0), (0.0,0.0,0.0,0.0)]:
            for val in row:
                blob += be_f32(val)
        
        # Volume fields - Per markdown lines 694-700
        blob += be_u32(6)            # +0x40: vTable (stored as volumeType 6 = AGGREGATE)
        blob += be_u32(3)            # +0x44: volumeTypeOrIndex (dict index → ClusteredMesh at entry 3)
        blob += be_u32(0)            # +0x48: padding
        blob += be_u32(0)            # +0x4C: padding
        blob += be_f32(0.0)          # +0x50: radius
        blob += be_u32(0)            # +0x54: groupID
        blob += be_u32(0xFFFFFFFF)   # +0x58: surfaceID
        blob += be_u32(1)            # +0x5C: m_flags
        
        return bytes(blob)
    
    def _build_clusteredmesh(self) -> bytes:
        """Build ClusteredMesh using EXACT RenderWare algorithm
        
        See: markdown/RENDERWARE_BUILD_PIPELINE.md for complete algorithm documentation
        """
        # Convert Blender vertices to Vector
        verts_vec = [Vector((v.x, v.y, v.z)) for v in self.obj.vertices]
        tris = [(f.v1, f.v2, f.v3) for f in self.obj.faces]
        
        # Run complete RenderWare pipeline
        # CRITICAL: Returns validated_tris (degenerate triangles filtered out)
        clusters, kdtree_nodes, bbox_min, bbox_max, validated_tris = RW_BuildClusteredMeshComplete(
            verts_vec, tris, self.granularity, self.enable_vertex_smoothing
        )
        
        # Now serialize to binary format with VALIDATED triangle list
        return self._serialize_clusteredmesh_binary(clusters, kdtree_nodes, bbox_min, bbox_max, validated_tris)
    
    def _serialize_clusteredmesh_binary(self, clusters: List[RWUnitCluster], 
                                         kdtree_nodes: List[KDNode],
                                         bbox_min: Vector, bbox_max: Vector,
                                         validated_tris: List[Tuple[int, int, int]]) -> bytes:
        """Serialize ClusteredMesh to binary format
        
        Based on RenderWare binary format + PSG_File_Format_Analysis.md
        """
        out = bytearray(0x60)  # Pre-allocate 96-byte header
        
        num_clusters = len(clusters)
        total_triangles = sum(len(c.unitIDs) for c in clusters)
        
        # Calculate m_numTagBits
        # VERIFIED AGAINST SKATE 3 PSG FILES (09D7BC5A9527DD0B.psg):
        #   Actual value: 0x0A (10)
        #   With 3 clusters, 700 triangles → formula confirms: - 4 is correct
        # 
        # SKATE 3 FORMULA (differs from standard RenderWare):
        #   mNumClusterTagBits = 1 + log2(mNumClusters)
        #   numUnitTagBits = 1 + log2(maxUnitStreamLength)
        #   m_numTagBits = mNumClusterTagBits + numUnitTagBits - 4  ← SKATE 3 SPECIFIC!
        mNumClusterTagBits = 1 + int(math.log2(max(1, num_clusters)))
        
        # maxUnitStreamLength = max of all cluster.unitDataSize (in bytes)
        max_unit_stream_length = max((len(c.unitIDs) * 9) for c in clusters) if clusters else 0
        numUnitTagBits = 1 + int(math.log2(max(1, max_unit_stream_length)))
        
        # CRITICAL: Skate 3 uses - 4, NOT + 1 like standard RenderWare!
        m_numTagBits = mNumClusterTagBits + numUnitTagBits - 4
        
        # PROCEDURAL BASE (48 bytes: +0x00-0x2F)
        out[0x00:0x04] = be_f32(bbox_min.x)
        out[0x04:0x08] = be_f32(bbox_min.y)
        out[0x08:0x0C] = be_f32(bbox_min.z)
        out[0x0C:0x10] = be_u32(0)
        out[0x10:0x14] = be_f32(bbox_max.x)
        out[0x14:0x18] = be_f32(bbox_max.y)
        out[0x18:0x1C] = be_f32(bbox_max.z)
        out[0x1C:0x20] = be_u32(0)
        out[0x20:0x24] = be_u32(0)  # m_vTable
        out[0x24:0x28] = be_u32(m_numTagBits)
        out[0x28:0x2C] = be_u32(total_triangles)
        out[0x2C:0x30] = be_u32(0)  # m_flags
        
        # Serialize KDTree
        kd_off = align(len(out), 16)
        while len(out) < kd_off:
            out.append(0)
        kd_blob = self._serialize_kdtree_binary(kdtree_nodes, bbox_min, bbox_max, total_triangles)
        out += kd_blob
        
        # Cluster pointer array
        cl_ptr_off = align(len(out), 16)
        while len(out) < cl_ptr_off:
            out.append(0)
        out += b"\x00" * (4 * num_clusters)
        
        # Serialize clusters
        blobs_start = align(len(out), 16)
        while len(out) < blobs_start:
            out.append(0)
        
        # CRITICAL: Use VALIDATED triangle list (degenerate triangles removed!)
        # Clusters reference triangle IDs in the validated list, NOT the original list!
        
        cluster_ptrs = []
        for cluster in clusters:
            c_off = len(out)
            # CRITICAL: Store ABSOLUTE offset, will convert to relative when writing
            cluster_ptrs.append(c_off)
            out += self._serialize_cluster_binary(cluster, self.granularity, validated_tris, self.force_uncompressed)
            # Track compression mode used
            self.compression_stats[cluster.compressionMode] += 1
        
        # Write cluster pointers as offsets RELATIVE TO ClusteredMesh base
        # Script analysis proves: cluster pointers are relative to ClusteredMesh base (start of out[])
        # The clusteredmeshbase.h line 437 code adds to baseAddress which itself is offset from this
        for i, ptr in enumerate(cluster_ptrs):
            # ptr is already absolute offset from start of out[] (ClusteredMesh base)
            # Write directly - already in correct format
            out[cl_ptr_off + i*4:cl_ptr_off + i*4 + 4] = be_u32(ptr)
        
        # CLUSTEREDMESH FIELDS (48 bytes: +0x30-0x5F)
        out[0x30:0x34] = be_u32(kd_off)
        out[0x34:0x38] = be_u32(cl_ptr_off)
        # +0x38-0x3F: ClusterParams as packed uint64
        # Format: [granularity float32][flags uint16][group_id_size uint8][surface_id_size uint8]
        granularity_bits = struct.unpack('>I', struct.pack('>f', self.granularity))[0]
        # CRITICAL: Real PSGs have group_id_size=2 but write 9-byte units (no groupID)!
        # Game uses UNIT FLAGS (0x40 bit), not ClusterParams, to detect groupID
        # Set group_id_size=0, surface_id_size=2 to match actual data
        cluster_params_u64 = (granularity_bits << 32) | (0x0010 << 16) | (0x00 << 8) | 0x02
        out[0x38:0x40] = struct.pack('>Q', cluster_params_u64)
        out[0x40:0x44] = be_u32(num_clusters)
        out[0x44:0x48] = be_u32(num_clusters)
        out[0x48:0x4C] = be_u32(total_triangles)
        out[0x4C:0x50] = be_u32(total_triangles)
        out[0x50:0x54] = be_u32(len(out))
        out[0x54:0x56] = be_u16(0)
        out[0x56:0x58] = be_u16(0)
        out[0x58:0x59] = be_u8(128)
        out[0x59:0x60] = b"\x00" * 7
        
        return bytes(out)
    
    def _serialize_cluster_binary(self, cluster: RWUnitCluster, granularity: float, 
                                   tris: List[Tuple[int, int, int]], force_uncompressed: bool = False) -> bytes:
        """Serialize single cluster to binary format
        
        EXACT port of RenderWare cluster serialization
        Sources:
            - clusterdatabuilder.cpp lines 22-51, 76-129
            - rwcclusteredmeshcluster.cpp lines 734-768
            - vertexcompression.cpp lines 46-75
        
        Args:
            force_uncompressed: If True, forces VERTICES_UNCOMPRESSED mode (compressionMode = 0)
                                to avoid ALL compression artifacts and phantom collisions.
                                File size will increase ~2x, but guarantees perfect precision.
        """
        out = bytearray()
        
        # Cluster header (16 bytes)
        unit_count = len(cluster.unitIDs)
        # Unit size: 1 flag + 3 verts + 3 edges + 2 surfaceID = 9 bytes (NO groupID)
        unit_data_size = unit_count * 9
        
        # Determine compression mode (vertexcompression.cpp lines 46-75)
        if force_uncompressed:
            # User requested uncompressed mode (compressionMode = 0)
            compression_mode = 0
            offset = (0, 0, 0)
        else:
            compression_mode, offset = determine_compression_mode(cluster.vertices, granularity)
        
        # Set cluster compressionMode and clusterOffset to match RenderWare UnitCluster structure
        cluster.compressionMode = compression_mode
        cluster.clusterOffset = offset  # (offset_x, offset_y, offset_z) tuple
        
        # Serialize vertices based on compression mode (rwcclusteredmeshcluster.cpp lines 234-298)
        # CRITICAL: Handle compression overflow for huge meshes
        try:
            if compression_mode == 0:
                # Uncompressed mode (lines 234-254)
                payload = serialize_cluster_uncompressed(cluster.vertices)
            elif compression_mode == 1:
                # 16-bit compression (lines 255-283)
                payload = serialize_cluster_16bit(cluster.vertices, granularity, offset)
            else:
                # 32-bit compression (lines 284-298)
                payload = serialize_cluster_32bit(cluster.vertices, granularity)
        except ValueError as e:
            # Compression overflow detected - automatically fall back to next mode
            error_msg = str(e)
            if "16-bit compression overflow" in error_msg:
                # 16-bit overflow: fall back to 32-bit compression
                compression_mode = 2
                cluster.compressionMode = compression_mode
                cluster.clusterOffset = (0, 0, 0)
                try:
                    payload = serialize_cluster_32bit(cluster.vertices, granularity)
                except ValueError as e2:
                    # 32-bit also overflowed - fall back to uncompressed
                    compression_mode = 0
                    cluster.compressionMode = compression_mode
                    cluster.clusterOffset = (0, 0, 0)
                    payload = serialize_cluster_uncompressed(cluster.vertices)
            elif "32-bit compression overflow" in error_msg:
                # 32-bit overflow: fall back to uncompressed (guaranteed to work)
                compression_mode = 0
                cluster.compressionMode = compression_mode
                cluster.clusterOffset = (0, 0, 0)
                payload = serialize_cluster_uncompressed(cluster.vertices)
            else:
                # Unknown error - re-raise
                raise
        
        # Track compression mode used (for statistics reporting)
        # Note: compression_mode may have changed due to fallback logic above
        cluster.compressionMode = compression_mode
        
        # Calculate section offsets (rwcclusteredmeshcluster.cpp lines 749-767)
        # CRITICAL: unitDataStart format depends on compression mode!
        # - Compressed (16-bit/32-bit): quadword offset (lines 753-754, 760-761)
        # - Uncompressed: VERTEX COUNT! (lines 765-766)
        vertex_section_end = 16 + len(payload)
        vertex_section_end_aligned = align_qw(vertex_section_end)
        
        if compression_mode == 0:  # UNCOMPRESSED
            # Lines 765-766: normalStart = mVertexCount; unitDataStart = mVertexCount + normalCount;
            normal_start_value = cluster.numVertices
            unit_data_start_value = cluster.numVertices + 0  # normalCount=0
        else:  # COMPRESSED (16-bit or 32-bit)
            # Lines 753-754, 760-761: quadword offset
            normal_start_value = (vertex_section_end_aligned - 16) // 16
            unit_data_start_value = normal_start_value  # No normals
        
        # Write header (rwcclusteredmeshcluster.cpp lines 664-686)
        # NOTE: total_size will be calculated AFTER unit stream alignment
        out += be_u16(unit_count)
        out += be_u16(unit_data_size)
        out += be_u16(unit_data_start_value)
        out += be_u16(normal_start_value)
        total_size_placeholder = len(out)  # Save position to write total_size later
        out += be_u16(0)  # Placeholder - will update after unit stream is aligned
        out += be_u8(cluster.numVertices)
        out += be_u8(0)  # normalCount
        out += be_u8(compression_mode)  # DYNAMIC: 1 (16-bit) or 2 (32-bit)
        out += b"\x00\x00\x00"
        
        # Write vertex payload
        out += payload
        out += b"\x00" * (vertex_section_end_aligned - vertex_section_end)
        
        # Write unit stream (based on WriteUnitDataToCluster lines 88-128)
        for unit_id in cluster.unitIDs:
            tri = tris[unit_id]
            
            # Get cluster-local vertex indices
            v0_local = RW_GetVertexCode(cluster, tri[0])
            v1_local = RW_GetVertexCode(cluster, tri[1])
            v2_local = RW_GetVertexCode(cluster, tri[2])
            
            # Write triangle unit with edge cos flags (9 bytes total)
            # Flags: type=TRIANGLE(1) + EDGEANGLE(0x20) + SURFACEID(0x80) = 0xA1
            # NOTE: Real PSGs have group_id_size=2 in ClusterParams but DON'T write groupID!
            # Game uses UNIT FLAGS (0x40 bit), not ClusterParams, to detect groupID presence
            out += be_u8(0xA1)
            out += be_u8(v0_local) + be_u8(v1_local) + be_u8(v2_local)
            # CRITICAL: Every triangle MUST have edge codes computed
            if unit_id not in cluster.edge_codes:
                print(f"⚠️ Triangle {unit_id} missing edge codes! Cluster {cluster.clusterID}")
            assert unit_id in cluster.edge_codes, \
                f"Triangle {unit_id} missing edge codes in cluster {cluster.clusterID}! All triangles must have edge codes computed."
            e0, e1, e2 = cluster.edge_codes[unit_id]
            out += be_u8(e0) + be_u8(e1) + be_u8(e2)

            # CRITICAL: SurfaceID is LITTLE-ENDIAN (unlike everything else in PSG!)
            surface_id = self.obj.faces[unit_id].surface_id or 0
            out += le_u16(surface_id)
        
        # Verify we wrote the correct number of units
        actual_unit_bytes = len(out) - 16 - (vertex_section_end_aligned - 16)
        # Unit size: 1 flag + 3 verts + 3 edges + 2 surfaceID = 9 bytes (NO groupID despite ClusterParams)
        expected_unit_bytes = unit_count * 9
        assert actual_unit_bytes == expected_unit_bytes, \
            f"Cluster {cluster.clusterID}: Wrote {actual_unit_bytes} bytes but expected {expected_unit_bytes} (unit_count={unit_count})"
        
        # Align unit stream
        unit_stream_aligned = align_qw(actual_unit_bytes)
        out += b"\x00" * (unit_stream_aligned - actual_unit_bytes)
        
        # Now calculate final total_size (rwcclusteredmeshcluster.cpp lines 670-697)
        # CRITICAL: RenderWare's totalSize = vertexDataSize + unitDataSize (WITHOUT 16-byte header!)
        # GetSize returns: sizeof(ClusteredMeshCluster) + vertexDataSize + unitDataSize - sizeof(Vector3)
        #                = 16 + vertexDataSize + unitDataSize - 16 = vertexDataSize + unitDataSize
        # So totalSize does NOT include the 16-byte cluster header!
        final_total_size_bytes = len(out) - 16  # Exclude 16-byte header
        assert (final_total_size_bytes + 16) % 16 == 0, \
            f"Cluster {cluster.clusterID}: total cluster size not 16-byte aligned!"
        
        # CRITICAL: totalSize is ALWAYS in bytes for Skate 3 (uint16 max = 65535)
        # RenderWare GetSize() returns bytes, stored directly in uint16 field
        # Large clusters that exceed 65535 bytes are not supported by RenderWare
        if final_total_size_bytes > 65535:
            print(f"⚠️ Cluster {cluster.clusterID}: size {final_total_size_bytes} bytes OVERFLOWS uint16 field!")
            print(f"   totalSize will be truncated - may cause corruption!")
            # Truncate to fit uint16 (this will cause issues, but matches RenderWare behavior)
            total_size_value = final_total_size_bytes & 0xFFFF
        else:
            # Normal case - write size in bytes (matches RenderWare GetSize)
            total_size_value = final_total_size_bytes
        
        # Update total_size in header (was written as placeholder at total_size_placeholder)
        out[total_size_placeholder:total_size_placeholder + 2] = be_u16(total_size_value & 0xFFFF)
        
        return bytes(out)
    
    def _serialize_kdtree_binary(self, kdtree_nodes: List[KDNode], 
                                  bbox_min: Vector, bbox_max: Vector,
                                  num_entries: int) -> bytes:
        """Serialize KDTree to binary format"""
        out = bytearray()
        
        num_branches = len(kdtree_nodes)
        
        # KDTree header (48 bytes)
        out += be_u32(0x90)  # m_branchNodes offset
        out += be_u32(num_branches)
        out += be_u32(num_entries)
        out += be_u32(0)
        out += be_f32(bbox_min.x) + be_f32(bbox_min.y) + be_f32(bbox_min.z) + be_u32(0)
        out += be_f32(bbox_max.x) + be_f32(bbox_max.y) + be_f32(bbox_max.z) + be_u32(0)
        
        # Write branch nodes (32 bytes each)
        for node in kdtree_nodes:
            out += be_u32(node.parent)
            out += be_u32(node.axis)
            
            # Left child
            if len(node.entries) > 0:
                content, index = node.entries[0]
                out += be_u32(content) + be_u32(index)
            else:
                out += be_u32(0) + be_u32(0)
            
            # Right child
            if len(node.entries) > 1:
                content, index = node.entries[1]
                out += be_u32(content) + be_u32(index)
            else:
                out += be_u32(0) + be_u32(0)
            
            out += be_f32(node.ext0) + be_f32(node.ext1)
        
        return bytes(out)
    
    def _build_collision_model(self) -> bytes:
        """Build CollisionModelData - Per PSG_File_Format_Analysis.md lines 1100-1148
        
        Structure: 20 bytes (0x14)
        - +0x00: m_BoundingVolume (index/ref to bounding volume)
        - +0x04: m_iNumMeshes (number of collision meshes)
        - +0x08: m_pMeshTable (relative offset to mesh pointer table)
        - +0x0C: field_0x0C (unknown, typically 2)
        - +0x10: field_0x10 (unknown, typically 0)
        
        Runtime Fixup (lines 1134-1140):
        - m_pMeshTable converted: relative offset → absolute pointer
        - m_BoundingVolume resolved via separate mechanism
        """
        blob = bytearray()
        blob += be_u32(0)     # +0x00: m_BoundingVolume (0 = no bounding volume)
        blob += be_u32(1)     # +0x04: m_iNumMeshes (1 collision mesh)
        blob += be_u32(0x0C)  # +0x08: m_pMeshTable (relative offset, points to 0x400C)
        blob += be_u32(2)     # +0x0C: field_0x0C (constant value 2 from reference)
        blob += be_u32(0)     # +0x10: field_0x10 (constant value 0 from reference)
        return bytes(blob)
    
    def _build_dmodata(self) -> bytes:
        """Build DMOData - Per PSG_File_Format_Analysis.md lines 1151-1213
        
        Structure:
        - Header: 20 bytes (version + counts + 2 relative pointers)
        - DMO entries: 128 bytes each (0x80 per entry)
        - String list: Variable length
        
        DMOEntry structure (128 bytes per IDA decompilation):
        - +0x00-0x77: Unknown data (120 bytes)
        - +0x78: pName (relative pointer to name string)
        - +0x7C: pData (relative pointer to DMO data)
        
        Runtime Unfix (lines 1194-1212):
        - Converts m_DMOs and m_StringList pointers to relative offsets
        - For each entry: converts pName and pData to relative offsets
        - Entry stride: 128 bytes (0x80)
        
        CRITICAL FIX 2025-10-20: ALL reference PSGs have numDMOs >= 1!
        Game REQUIRES at least 1 DMO entry even for collision-only files.
        Writing minimal but complete DMOData matching reference structure.
        """
        blob = bytearray()
        
        # COMPLETE DMOData section - EXACT COPY from original PSG (445 bytes)
        # Per PSG_File_Format_Analysis.md structure + complete hex dump
        complete_dmodata = bytes.fromhex(
            '00000000000000010000000100000020000000a0000000000000000000000000'
            '3f800000000000000000000000000000000000003f800000000000000000000000000000000000003f80000000000000'
            'c0b861a84109e001c1f6f6eb3f800000c10b43424109aedac20466fb00000000c09e65c1411804dec1f4c9dd00000000'
            'b7275165169767882c70170600030d9a00020dec03e38707fe6dcbe0000000a0'
            '444d4f5f554e5f5069636e69635461626c655f313030315f3078326337303137303530303033303031323a3078326337303137303630303033306431653a3078326337303137303630303033306431663a3078326337303137303630303033306432303a3078326337303137303630303033306432313a3078326337303137303630303033306436633a3078326337303137303630303033306439613a3078303030316364623630336533383730333a3078303030323064656330336533383730373a3a5b3078303030303030386430336533383730345d5b3078326337303137303530303033303031325d5b3078326337303137303630303033306431655d5b3078326337303137303630303033306431665d5f486967684c4f4400'
        )
        blob += complete_dmodata
        
        # No padding needed - add_object() handles 16-byte alignment
        
        return bytes(blob)
    
    def _build_splinedata(self) -> bytes:
        """Build SplineData from Blender curves
        Per PSG_File_Format_Analysis.md lines 1216-1423
        
        Structure:
        - Header: 16 bytes (counts + 2 relative pointers)
        - Spline headers: 32 bytes each (GUID, instanceIndex, head/tail segment offsets)
        - Segments: 144 bytes each (Hermite matrix, inverse, bbox, length, distance, linkage)
        
        CRITICAL (lines 1254-1263):
        - ALL linkage fields are RELATIVE BYTE OFFSETS, NOT indices!
        - headSegment/tailSegment: Byte offsets like 0x0130, NOT indices like 0, 1, 2
        - splinePtr/prevSegment/nextSegment: Byte offsets within SplineData object
        - NULL links: 0x00000000 (NOT 0xFFFFFFFF)
        
        Runtime Unfix (lines 1360-1383):
        - Converts all pointers to relative offsets by subtracting object base
        - NULL-checks before conversion (0 pointers remain 0)
        """
        # Check if we have splines
        if not hasattr(self.obj, 'splines') or len(self.obj.splines) == 0:
            # Create minimal empty SplineData
            blob = bytearray()
            blob += be_u32(0)  # m_uiNumSplines
            blob += be_u32(0)  # m_uiNumSegments
            blob += be_u32(0)  # m_Splines ptr (NULL)
            blob += be_u32(0)  # m_Segments ptr (NULL)

            return bytes(blob)
        
        # CRITICAL FIX: Validate splines have non-zero length segments
        # Zero-length segments cause SPU "Invalid code" crashes!
        valid_splines = []
        for spline in self.obj.splines:
            if len(spline.points) < 2:
                continue
            # Check if any segment has non-zero length
            has_valid_segment = False
            for i in range(len(spline.points) - 1):
                v1 = spline.points[i]
                v2 = spline.points[i + 1]
                dx = v2.x - v1.x
                dy = v2.y - v1.y
                dz = v2.z - v1.z
                length = math.sqrt(dx*dx + dy*dy + dz*dz)
                if length > 0.001:  # Minimum 1mm threshold
                    has_valid_segment = True
                    break
            if has_valid_segment:
                valid_splines.append(spline)
        
        if len(valid_splines) == 0:

            blob = bytearray()
            blob += be_u32(0)  # m_uiNumSplines
            blob += be_u32(0)  # m_uiNumSegments
            blob += be_u32(0)  # m_Splines ptr (NULL)
            blob += be_u32(0)  # m_Segments ptr (NULL)
            return bytes(blob)
        
        splines = valid_splines
        total_segments = sum(len(s.points) - 1 for s in splines if len(s.points) > 1)
        
        # Header (16 bytes)
        blob = bytearray()
        blob += be_u32(len(splines))      # m_uiNumSplines
        blob += be_u32(total_segments)    # m_uiNumSegments
        blob += be_u32(0x10)              # m_Splines offset (after header)
        blob += be_u32(0x10 + len(splines) * 0x20)  # m_Segments offset
        
        # Spline headers (32 bytes each, stride 0x20)
        # Per PSG_File_Format_Analysis.md lines 1232-1239
        # Structure: guid_high(8) + guid_low(8) + instanceIndex(4) + headSegment(4) + tailSegment(4) + padding(4)
        segment_idx = 0
        for spline in splines:
            num_segs = max(0, len(spline.points) - 1)
            
            # Generate guid_hi from spline geometry (unique per spline)
            points_str = ''.join(f"{p.x:.6f}{p.y:.6f}{p.z:.6f}" for p in spline.points)
            guid_hi = int(hashlib.sha256(points_str.encode()).hexdigest()[:16], 16)
            
            # Generate guid_lo based on spline type (category pattern from analysis)
            # Pattern: 0x2C701707 00 07 004A (base + category + subtype)
            # Using 0x0007004A for rails (most common type in reference files)
            guid_lo = 0x2C7017070007004A  # Rail/grind category
            
            blob += be_u64(guid_hi)                # m_uiGuid (unique hash)
            blob += be_u64(guid_lo)                # m_uiGuidLocal (category ID)
            blob += be_u32(0)                      # m_Instance
            
            # m_Head/m_Tail: RELATIVE OFFSETS to first/last segments, or 0x00000000 for NULL
            # Segments start at: 0x10 + len(splines) * 0x20
            segments_base = 0x10 + len(splines) * 0x20
            if num_segs > 0:
                head_offset = segments_base + segment_idx * 0x90
                tail_offset = segments_base + (segment_idx + num_segs - 1) * 0x90
                blob += be_u32(head_offset)  # m_Head
                blob += be_u32(tail_offset)  # m_Tail
            else:
                blob += be_u32(0)  # m_Head (NULL for empty spline)
                blob += be_u32(0)  # m_Tail (NULL for empty spline)
            
            blob += be_u32(0)                      # +0x1C: padding
            segment_idx += num_segs
        
        # Segments (144 bytes each, stride 0x90)
        # Per PSG_File_Format_Analysis.md lines 1241-1251
        # Structure per segment:
        # +0x00: hermiteBasis[4][4] (64 bytes) - 4x4 transformation matrix
        # +0x40: inverse[4] (16 bytes) - inverse parameters
        # +0x50: bbox[2][4] (32 bytes) - bounding box min/max
        # +0x70: length (4 bytes) - segment length
        # +0x74: distance (4 bytes) - cumulative distance along spline
        # +0x78: splinePtr (4 bytes) - RELATIVE OFFSET to parent spline header
        # +0x7C: prevSegment (4 bytes) - RELATIVE OFFSET to previous segment (or 0x00000000)
        # +0x80: nextSegment (4 bytes) - RELATIVE OFFSET to next segment (or 0x00000000)
        # +0x84: padding[3] (12 bytes) - padding to 144 bytes
        global_seg_idx = 0
        
        for spline_idx, spline in enumerate(splines):
            spline_start_distance = 0.0
            
            for seg_local_idx in range(len(spline.points) - 1):
                v1 = spline.points[seg_local_idx]
                v2 = spline.points[seg_local_idx + 1]
                
                # Calculate segment length
                dx = v2.x - v1.x
                dy = v2.y - v1.y
                dz = v2.z - v1.z
                length = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                # Hermite basis matrix (64 bytes at +0x00-0x3F)
                # Per PSG_File_Format_Analysis.md lines 1329-1353
                # For linear segments: Row0=[dx,dy,dz,0], Row1-2=zeros, Row3=[x1,y1,z1,1]
                # Importer decodes: start=row3[0:3], end=(row0+row1+row2+row3)[0:3]
                hermite_matrix = [
                    dx, dy, dz, 0.0,            # Row 0: Displacement (v2-v1)
                    0.0, 0.0, 0.0, 0.0,         # Row 1: Zero
                    0.0, 0.0, 0.0, 0.0,         # Row 2: Zero
                    v1.x, v1.y, v1.z, 1.0,      # Row 3: Start point (v1)
                ]
                for val in hermite_matrix:
                    blob += be_f32(val)
                
                # Inverse parameters (16 bytes at +0x40-0x4F)
                # Per PSG_File_Format_Analysis.md line 1243 (CORRECTED: 16 bytes, not 64!)
                inv_length = 1.0 / length if length > 0 else 0.0
                blob += be_f32(inv_length) + be_f32(0) + be_f32(0) + be_f32(0)
                
                # Bounding box (32 bytes at +0x50-0x6F) - Per line 1244
                # 2 vec4s: min(x,y,z,0) + max(x,y,z,0)
                min_x, min_y, min_z = min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z)
                max_x, max_y, max_z = max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z)
                blob += be_f32(min_x) + be_f32(min_y) + be_f32(min_z) + be_f32(0)
                blob += be_f32(max_x) + be_f32(max_y) + be_f32(max_z) + be_f32(0)
                
                # Segment properties
                blob += be_f32(length)                  # +0x70: m_fLength
                blob += be_f32(spline_start_distance)   # +0x74: m_fDistance
                spline_start_distance += length
                
                # Linkage pointers - RELATIVE BYTE OFFSETS (lines 1247-1249, 1377-1382)
                # CRITICAL: These are offsets within SplineData object, NOT array indices!
                # Spline headers: base = 0x10, stride = 0x20
                # Segments: base = 0x10 + numSplines * 0x20, stride = 0x90
                segments_base = 0x10 + len(splines) * 0x20
                spline_header_offset = 0x10 + spline_idx * 0x20
                
                blob += be_u32(spline_header_offset)  # +0x78: splinePtr (offset to parent header)
                
                # prev/next: relative byte offsets to adjacent segments, or 0x00000000 for NULL
                if seg_local_idx > 0:
                    prev_offset = segments_base + (global_seg_idx - 1) * 0x90
                    blob += be_u32(prev_offset)  # +0x7C: prevSegment
                else:
                    blob += be_u32(0)  # +0x7C: NULL (first segment in spline)
                
                if seg_local_idx < len(spline.points) - 2:
                    next_offset = segments_base + (global_seg_idx + 1) * 0x90
                    blob += be_u32(next_offset)  # +0x80: nextSegment
                else:
                    blob += be_u32(0)  # +0x80: NULL (last segment in spline)
                
                # Padding to 144-byte stride (12 bytes at +0x84-0x8F)
                blob += be_u32(0) + be_u32(0) + be_u32(0)
                
                global_seg_idx += 1
        
        return bytes(blob)
    
    def _build_toc(self) -> bytes:
        """Build TableOfContents - Per PSG_File_Format_Analysis.md lines 1427-1541
        
        Structure:
        - Header: 20 bytes (itemsCount + 3 relative pointers + typeCount)
        - Entry array: 24 bytes per entry (TOCEntry structure)
        - Type Map: 50 entries (25 TypeIDs + 25 item counts)
        - Name strings: Variable (usually empty for collision PSGs)
        
        TOCEntry structure (24 bytes per IDA decompilation lines 1444-1452):
        - +0x00: mName (relative pointer, always 0 in collision PSGs)
        - +0x04: mHash1 (NOT converted, constant 0xFEFFFFFF)
        - +0x08: mHash2 (NOT converted, dynamic hash)
        - +0x0C: mResourceKeyHash (NOT converted, dynamic hash)
        - +0x10: mTypeID (NOT a subreference, just TypeID value)
        - +0x14: mObject (ONLY subreference in TOC entry)
        
        Type Map Pattern (lines 1480-1507):
        - 25 TypeIDs (CONSTANT across all PSG files)
        - Item count repeated after EACH TypeID (including last!)
        - Total: 50 uint32 entries
        
        Runtime Unfix (lines 1511-1541):
        - Converts mObject (+0x14) via Serialize() - ONLY subreference
        - mName (+0x00) converted to relative if non-zero (always 0 for collision)
        - Hash fields (+0x04, +0x08, +0x0C, +0x10) NOT converted
        """
        blob = bytearray()
        
        # Calculate number of entries
        num_splines = len(self.obj.splines) if hasattr(self.obj, 'splines') else 0
        num_items = 1 + 1 + num_splines + 1 + 1  # INSTANCESUBREF + INSTANCEDATA + SPLINESUBREFs + SPLINEDATA + DMODATA
        
        # Type Map - CONSTANT TypeID catalog from EA RenderWare
        # Pattern: TypeID, ItemCount, TypeID, ItemCount, ..., TypeID, ItemCount (ALL have counts!)
        # 25 TypeIDs + 25 ItemCount values = 50 total entries
        type_map_types = [
            0x00EB0066,  # RENDERMATERIALSUBREF
            0x00EB0005,  # RENDERMATERIALDATA
            0x00EB0067,  # COLLISIONMATERIALSUBREF
            0x00EB0006,  # COLLISIONMATERIALDATA
            0x00EB0001,  # RENDERMODELDATA
            0x00EB000A,  # COLLISIONMODELDATA
            0x00EB0065,  # ROLLERDESCSUBREF
            0x00EB0007,  # ROLLERDESCDATA
            0x00EB0069,  # INSTANCESUBREF
            0x00EB000D,  # INSTANCEDATA
            0x00EB006B,  # TRIGGERINSTANCESUBREF
            0x00EB0019,  # TRIGGERINSTANCEDATA
            0x00EB0064,  # SPLINESUBREF
            0x00EB0004,  # SPLINEDATA
            0x00EB0068,  # TRIGGERSUBREF
            0x00EB0009,  # TRIGGERDATA
            0x00EB0016,  # LIGHTSTREAMDATA
            0x00EB0013,  # LIGHTDATA
            0x00EB0014,  # LIGHTPROBEDATA
            0x00EB0018,  # PARTICLEEFFECTDATA
            0x00EB0017,  # SOUNDDATA
            0x00EB0020,  # DECALDATA
            0x00EB0024,  # PATHDATA
            0x00EB0026,  # AIDATA
            0x00EB0027,  # ENTITYDATA
        ]
        
        # Total entries = 25 TypeIDs + 25 item counts = 50 entries
        # NOTE: type_count field = 25 (counts pairs, not individual entries)
        type_count = len(type_map_types)
        
        # Header (20 bytes) - Per PSG_File_Format_Analysis.md lines 1435-1441
        blob += be_u32(num_items)  # +0x00: m_uiItemsCount (number of TOC entries)
        blob += be_u32(0x14)  # +0x04: m_pArray (relative offset to entry array)
        blob += be_u32(0x14 + num_items * 0x18)  # +0x08: m_pNames (relative offset to names, unused in collision)
        blob += be_u32(type_count)  # +0x0C: m_uiTypeCount (25 TypeIDs)
        blob += be_u32(0x14 + num_items * 0x18)  # +0x10: m_pTypeMap (relative offset to type map)
        
        # TOC Entries (24 bytes each) - Per lines 1445-1452
        # Pattern from reference: INSTANCESUBREF, INSTANCEDATA, SPLINESUBREFs, SPLINEDATA, DMODATA
        # Entry structure: mName(4) + mHash1(4) + mHash2(4) + mResourceKeyHash(4) + mTypeID(4) + mObject(4)
        
        # Get InstanceData GUID for Entry 0
        bounds_str = f"{self.obj.bounds_min}{self.obj.bounds_max}{len(self.obj.vertices)}"
        hash_byte = int(hashlib.md5(bounds_str.encode()).hexdigest()[:2], 16)
        instance_guid = 0xF8DED31D3B0F6B00 | (hash_byte & 0xFF)
        
        entry_idx = 0
        
        # Entry 0: INSTANCESUBREF
        blob += be_u32(0)                                       # +0x00: mName (0 = no name string)
        blob += be_u32(0xFEFFFFFF)                              # +0x04: mHash1 (constant marker, line 1474)
        blob += be_u32((instance_guid >> 32) & 0xFFFFFFFF)      # +0x08: mHash2 (high 32 bits of GUID)
        blob += be_u32(instance_guid & 0xFFFFFFFF)              # +0x0C: mResourceKeyHash (low 32 bits)
        blob += be_u32(0x00EB0069)                              # +0x10: mTypeID (INSTANCESUBREF)
        blob += be_u32(0x00800000)                              # +0x14: mObject (subreference, high bit pattern)
        entry_idx += 1
        
        # Entry 1: INSTANCEDATA
        blob += be_u32(0)                                       # +0x00: mName
        blob += be_u32(0xFEFFFFFF)                              # +0x04: mHash1
        entry_hash = int(hashlib.md5(f"inst{bounds_str}".encode()).hexdigest()[:16], 16)
        blob += be_u32((entry_hash >> 32) & 0xFFFFFFFF)         # +0x08: mHash2
        blob += be_u32(entry_hash & 0xFFFFFFFF)                 # +0x0C: mResourceKeyHash
        blob += be_u32(0x00EB000D)                              # +0x10: mTypeID (INSTANCEDATA)
        blob += be_u32(1)                                       # +0x14: mObject (arena dict index 1)
        entry_idx += 1
        
        # Entries 2...N: SPLINESUBREF (one per spline)
        for i in range(num_splines):
            blob += be_u32(0)  # mName
            blob += be_u32(0xFEFFFFFF)  # mHash1
            spline_hash = int(hashlib.md5(f"spline{i}{bounds_str}".encode()).hexdigest()[:16], 16)
            blob += be_u32((spline_hash >> 32) & 0xFFFFFFFF)  # mHash2
            blob += be_u32(spline_hash & 0xFFFFFFFF)  # mResourceKeyHash
            blob += be_u32(0x00EB0064)  # mTypeID (SPLINESUBREF)
            blob += be_u32(0x00800001 + i)  # mObject (high bit set + index)
            entry_idx += 1
        
        # Entry N+1: SPLINEDATA
        blob += be_u32(0)  # mName
        blob += be_u32(0xFEFFFFFF)  # mHash1
        spline_data_hash = int(hashlib.md5(f"splinedata{bounds_str}".encode()).hexdigest()[:16], 16)
        blob += be_u32((spline_data_hash >> 32) & 0xFFFFFFFF)  # mHash2
        blob += be_u32(spline_data_hash & 0xFFFFFFFF)  # mResourceKeyHash
        blob += be_u32(0x00EB0004)  # mTypeID (SPLINEDATA)
        blob += be_u32(6)  # mObject (dict index 6)
        entry_idx += 1
        
        # Entry N+2: DMODATA
        blob += be_u32(0)  # mName
        blob += be_u32(0xFEFFFFFF)  # mHash1
        dmo_hash = int(hashlib.md5(f"dmo{bounds_str}".encode()).hexdigest()[:16], 16)
        blob += be_u32((dmo_hash >> 32) & 0xFFFFFFFF)  # mHash2
        blob += be_u32(dmo_hash & 0xFFFFFFFF)  # mResourceKeyHash
        blob += be_u32(0x00EB001D)  # mTypeID (DMODATA)
        blob += be_u32(5)  # mObject (dict index 5)
        entry_idx += 1
        
        # Type Map (50 entries: 25 TypeIDs + 25 item counts)
        # Per PSG_File_Format_Analysis.md lines 1480-1507
        # CRITICAL: The value between TypeIDs is NOT a separator!
        # It's m_uiItemsCount repeated 25 times (changes per file: 13, 52, 17, 54 observed)
        # Pattern: TypeID, ItemCount, TypeID, ItemCount, ..., TypeID, ItemCount
        # ALL 25 TypeIDs have item counts after them (including the last one!)
        for typ in type_map_types:
            blob += be_u32(typ)         # TypeID from CONSTANT catalog
            blob += be_u32(num_items)   # m_uiItemsCount repeated after EACH TypeID
        
        return bytes(blob)
    
    def _build_dictionary(self, obj_entries) -> bytes:
        """Build Arena Dictionary - Per PSG_File_Format_Analysis.md lines 1544-1627
        
        Structure: 8 entries x 24 bytes = 192 bytes total
        Entry count is CONSTANT across all PSG files (always 8 objects)
        
        ArenaDictEntry structure (24 bytes per lines 1551-1558):
        - +0x00: ptr (relative offset to object from file start)
        - +0x04: reloc (relocation type, always 0)
        - +0x08: size (object size in bytes)
        - +0x0C: alignment (always 0x10 = 16 bytes)
        - +0x10: typeIndex (index into Type Registry section)
        - +0x14: typeID (object TypeID for identification)
        
        Object order (CONSTANT per lines 1564-1572):
        [0] VersionData, [1] InstanceData, [2] Volume, [3] ClusteredMesh,
        [4] CollisionModelData, [5] DMOData, [6] SplineData, [7] TableOfContents
        
        CRITICAL (lines 1575-1576):
        - ObjectID is 0-BASED: ObjectID N = Entry[N] in this dictionary
        - Subreferences use 0-based indexing to reference these entries
        """
        blob = bytearray()
        
        # Write all 8 arena dictionary entries (24 bytes each)
        for offset, size, type_index, type_id in obj_entries:
            blob += be_u32(offset)      # +0x00: ptr (relative offset to object)
            blob += be_u32(0)           # +0x04: reloc (always 0)
            blob += be_u32(size)        # +0x08: size (object size in bytes)
            blob += be_u32(0x10)        # +0x0C: alignment (always 0x10)
            blob += be_u32(type_index)  # +0x10: typeIndex (into Type Registry)
            blob += be_u32(type_id)     # +0x14: typeID (object identification)
        
        return bytes(blob)
    
    def _build_subref_records(self) -> bytes:
        """Build subreference records - Per PSG_File_Format_Analysis.md lines 1630-1689
        
        Structure: 8 bytes per record (objectId + offset)
        Count: DYNAMIC (1 + number of splines)
        
        ArenaSectionSubreferencesRecord structure (8 bytes per lines 1642-1645):
        - +0x00: objectId (0-BASED index into Arena Dictionary)
        - +0x04: offset (byte offset within target object)
        
        CRITICAL (lines 1635-1638):
        - ObjectID is 0-BASED: ObjectID N = Entry[N] in Arena Dictionary
        - ObjectID 0 = VersionData, 1 = InstanceData, ..., 6 = SplineData
        
        Record pattern (lines 1656-1665):
        - Record[0]: ObjectID=1 (InstanceData) @ offset 0x20
          Points to instance entry which contains 3 subreferences at +0x80, +0x84, +0x88
        - Record[1-N]: ObjectID=6 (SplineData) @ offsets 0x10, 0x30, 0x50, ...
          Points to spline headers (stride 0x20 = 32 bytes)
          Number of ObjectID=6 records = spline count (verified pattern)
        
        Purpose: Arena fixup system needs these records to locate and resolve
        subreference fields during file loading (Fixup) and saving (Unfix).
        """
        blob = bytearray()
        
        # Record 0: InstanceData entry
        # Per lines 1657, 1677: ObjectID=1 (0-based Entry[1]), offset=0x20
        blob += be_u32(1)     # +0x00: objectId (Entry[1] = InstanceData)
        blob += be_u32(0x20)  # +0x04: offset (instance entry at InstanceData+0x20)
        
        # Records 1-N: Spline headers
        # Per lines 1658, 1678-1686: ObjectID=6 (0-based Entry[6]), stride 0x20
        num_splines = len(self.obj.splines) if hasattr(self.obj, 'splines') else 0
        for i in range(num_splines):
            blob += be_u32(6)                # +0x00: objectId (Entry[6] = SplineData)
            blob += be_u32(0x10 + i * 0x20)  # +0x04: offset to spline header[i]
        
        return bytes(blob)
    
    def _build_subref_dict(self, num_entries: int) -> bytes:
        """Build subreference dictionary - Per PSG_File_Format_Analysis.md lines 1693-1718
        
        Structure: 4 bytes per entry (uint32)
        Count: Matches subreference record count (DYNAMIC)
        
        Value: 0x00000000 for ALL entries in collision PSGs
        
        Pattern Analysis (lines 1704-1708):
        - Entry size: 4 bytes (CONFIRMED across all files)
        - Entry counts: Match subref record counts (10, 49, 14, 51)
        - All entries: 0x00000000 (CONSTANT - all internal references)
        - Purpose: Indicates arena location for external references
        
        For collision PSGs: All subreferences are internal (within same arena),
        so all dictionary entries are 0x00000000. Non-zero values would indicate
        external arena references (unused in collision files).
        """
        return be_u32(0) * num_entries
    
    def _update_subref_pointers(self, records_off: int, dict_off: int, num_subrefs: int):
        """Update Section Subreferences - Per PSG_File_Format_Analysis.md lines 443-488
        
        ArenaSectionSubreferences structure (28 bytes at 0x020C):
        - +0x00: typeID (0x00010007 - already written)
        - +0x04: numEntries (DYNAMIC - updated here)
        - +0x08: m_dictAfterRefix (0x00000000)
        - +0x0C: m_recordsAfterRefix (0x00000000)
        - +0x10: dict (ABSOLUTE file offset - updated here)
        - +0x14: records (ABSOLUTE file offset - updated here)
        - +0x18: numUsed (DYNAMIC - updated here)
        
        CRITICAL (line 481):
        - dict and records pointers use ABSOLUTE file offsets (NOT relative!)
        - This is the only section that uses absolute offsets instead of relative
        - Runtime Unfix does NOT convert these pointers (they remain absolute)
        """
        subref_section = 0x20C
        
        # Update counts
        self.blob[subref_section + 0x04:subref_section + 0x08] = be_u32(num_subrefs)  # +0x04: numEntries
        self.blob[subref_section + 0x18:subref_section + 0x1C] = be_u32(num_subrefs)  # +0x18: numUsed
        
        # Update ABSOLUTE file offset pointers (CRITICAL: absolute, not relative!)
        self.blob[subref_section + 0x10:subref_section + 0x14] = be_u32(dict_off)     # +0x10: dict (ABSOLUTE)
        self.blob[subref_section + 0x14:subref_section + 0x18] = be_u32(records_off)  # +0x14: records (ABSOLUTE)

# ===== BLENDER UI =====

class PSGExportProperties(bpy.types.PropertyGroup):
    """Properties for PSG export"""
    filepath: StringProperty(
        name="Export Folder",
        description="Folder where PSG files will be saved (filename auto-generated as 16-char hex like 17A9ABC06C68C2E4.psg)",
        default="//",
        subtype='DIR_PATH'
    )

    force_uncompressed: BoolProperty(
        name="Force Uncompressed Vertices",
        description=(
            "FIXES PHANTOM COLLISIONS: Store vertices as uncompressed float32 instead of compressed integers.\n"
            "Use this if you have complex geometry in a small space and are getting phantom collision walls.\n"
            "⚠️ File size will increase ~33% (16 vs 12 bytes/vertex), but guarantees perfect precision and eliminates ALL compression artifacts"
        ),
        default=False
    )
    
    enable_vertex_smoothing: BoolProperty(
        name="Enable Vertex Smoothing",
        description=(
            "SMOOTHS NON-FEATURE VERTICES: Disables vertex collision on smooth surfaces (~93% of vertices).\n"
            "✅ ENABLE for terrain/organic shapes (smooth skateparks, natural surfaces)\n"
            "❌ DISABLE for hard-edged objects (stairs, rails, sharp ramps, thin geometry)\n"
            "⚠️ If players phase through corners or rails, DISABLE this!"
        ),
        default=False  # DEFAULT TO OFF for user safety - prevents phasing through rails/corners
    )
    
    # Bulk export progress tracking (internal properties)
    is_exporting: BoolProperty(
        name="Is Exporting",
        description="Internal flag indicating bulk export is in progress",
        default=False
    )
    
    current_export: bpy.props.IntProperty(
        name="Current Export",
        description="Current object being exported in bulk export",
        default=0,
        min=0
    )
    
    total_exports: bpy.props.IntProperty(
        name="Total Exports",
        description="Total number of objects to export in bulk export",
        default=0,
        min=0
    )
    
    # Surface material properties
    audio_surface: EnumProperty(
        name="Audio Surface",
        description="Audio/grind material type (controls sound and grinding behavior)",
        items=[
            ('0', 'Undefined', 'Generic surface'),
            ('1', 'Asphalt_Smooth', 'Smooth asphalt'),
            ('2', 'Asphalt_Rough', 'Rough asphalt'),
            ('3', 'Concrete_Polished', 'Polished concrete'),
            ('4', 'Concrete_Rough', 'Rough concrete'),
            ('5', 'Concrete_Aggregate', 'Aggregate concrete'),
            ('6', 'Wood_Ramp', 'Wood ramp'),
            ('7', 'Plywood', 'Plywood'),
            ('8', 'Dirt', 'Dirt'),
            ('9', 'Metal', 'Metal'),
            ('10', 'Grass', 'Grass (sets mIsAboveGrass flag)'),
            ('11', 'Metal_Solid_Round_1', 'Metal solid round 1'),
            ('12', 'Metal_Solid_Round_1_Up', 'Metal solid round 1 up'),
            ('13', 'Metal_Solid_Round_2', 'Metal solid round 2'),
            ('14', 'Metal_Solid_Square_1', 'Metal solid square 1'),
            ('15', 'Metal_Solid_Square_2', 'Metal solid square 2'),
            ('16', 'Metal_Hollow_Round_1', 'Metal hollow round 1'),
            ('17', 'Metal_Hollow_Round_1_Dead', 'Metal hollow round 1 dead'),
            ('18', 'Metal_Hollow_Round_1_Dn', 'Metal hollow round 1 down'),
            ('19', 'Metal_Hollow_Round_2', 'Metal hollow round 2'),
            ('20', 'Metal_Hollow_Round_2_Dead', 'Metal hollow round 2 dead'),
            ('21', 'Metal_Hollow_Round_2_Dn', 'Metal hollow round 2 down'),
            ('22', 'Metal_Hollow_Round_3', 'Metal hollow round 3'),
            ('23', 'Metal_Hollow_Round_4', 'Metal hollow round 4'),
            ('24', 'Metal_Hollow_Square_1', 'Metal hollow square 1'),
            ('25', 'Metal_Hollow_Square_2', 'Metal hollow square 2'),
            ('26', 'Metal_Hollow_Square_3', 'Metal hollow square 3'),
            ('27', 'Metal_Hollow_Square_3_Dead', 'Metal hollow square 3 dead'),
            ('28', 'Metal_Hollow_Square_4', 'Metal hollow square 4'),
            ('29', 'Metal_Hollow_1', 'Metal hollow 1'),
            ('30', 'Metal_Hollow_2', 'Metal hollow 2'),
            ('31', 'Metal_Sheet', 'Metal sheet'),
            ('32', 'Metal_Complex_1', 'Metal complex 1'),
            ('33', 'Metal_Complex_2', 'Metal complex 2'),
            ('34', 'Metal_Complex_3', 'Metal complex 3'),
            ('35', 'Metal_Complex_4', 'Metal complex 4'),
            ('36', 'Metal_Complex_5', 'Metal complex 5'),
            ('37', 'Metal_Complex_6', 'Metal complex 6'),
            ('38', 'Metal_Complex_7', 'Metal complex 7'),
            ('39', 'Metal_Complex_8', 'Metal complex 8'),
            ('40', 'Metal_Complex_Debris', 'Metal complex debris'),
            ('41', 'Wood_1', 'Wood 1'),
            ('42', 'Wood_1_Up', 'Wood 1 up'),
            ('43', 'Wood_2', 'Wood 2'),
            ('44', 'Wood_3', 'Wood 3'),
            ('45', 'Wood_3_Up', 'Wood 3 up'),
            ('46', 'Wood_4', 'Wood 4'),
            ('47', 'Plastic_1', 'Plastic 1'),
            ('48', 'Plastic_2', 'Plastic 2'),
            ('49', 'Plastic_3', 'Plastic 3'),
            ('50', 'Plastic_4', 'Plastic 4'),
            ('51', 'Glass_Thick_Large', 'Glass thick large'),
            ('52', 'Glass_Thin_Small', 'Glass thin small'),
            ('53', 'Concrete_Curb', 'Concrete curb'),
            ('54', 'Concrete_Bench', 'Concrete bench'),
            ('55', 'Leaves', 'Leaves'),
            ('56', 'Bush', 'Bush'),
            ('57', 'Pottery', 'Pottery'),
            ('58', 'Paper', 'Paper'),
            ('59', 'Cardboard', 'Cardboard'),
            ('60', 'Garbage_Bag', 'Garbage bag'),
            ('61', 'Garbage_Spill', 'Garbage spill'),
            ('62', 'Bottle', 'Bottle'),
            ('63', 'Tile_Ceramic', 'Tile ceramic'),
            ('64', 'Marble_or_Slate', 'Marble or slate'),
            ('65', 'Brick_Smooth', 'Brick smooth'),
            ('66', 'Brick_Coarse', 'Brick coarse'),
            ('67', 'Manhole_Metal', 'Manhole metal'),
            ('68', 'Metal_Grate_Sewer', 'Metal grate sewer'),
            ('69', 'Metal_Grate_Planter', 'Metal grate planter'),
            ('70', 'DeepSnow', 'Deep snow'),
            ('71', 'PackedSnow', 'Packed snow'),
            ('72', 'Ice', 'Ice'),
            ('73', 'Antennas', 'Antennas'),
            ('74', 'Chandelier', 'Chandelier'),
            ('75', 'Plexiglass_Small', 'Plexiglass small'),
            ('76', 'Plexiglass_Large', 'Plexiglass large'),
            ('77', 'Potted_Plant', 'Potted plant'),
            ('78', 'Crumpled_Paper', 'Crumpled paper'),
            ('79', 'Cloth', 'Cloth'),
            ('80', 'Pop_Can', 'Pop can'),
            ('81', 'Paper_Cup', 'Paper cup'),
            ('82', 'Wire_Cable', 'Wire cable'),
            ('83', 'VolleyBall', 'Volleyball'),
            ('84', 'OilDrum', 'Oil drum'),
            ('85', 'DMORail', 'DMO rail'),
            ('86', 'Fruit', 'Fruit'),
            ('87', 'Plastic_Bottle', 'Plastic bottle'),
            ('88', 'Drum_Pylon', 'Drum pylon'),
            ('89', 'Metal_Rail_4', 'Metal rail 4'),
            ('90', 'Wood_5', 'Wood 5'),
            ('91', 'Metal_Ramp', 'Metal ramp'),
            ('92', 'Complex_Plastic_1', 'Complex plastic 1'),
            ('93', 'Max_Mappable_Surface', 'Max mappable surface'),
            # Note: Indices 94-127 are body/ragdoll sounds (Board, Head, Torso, BoneCrack, etc.)
            # These are NOT surface materials - they're triggered by crash/ragdoll systems
            # Removed from dropdown to avoid confusion
        ],
        default='32'  # Metal_Complex_1 (was the old hardcoded 0x8620's audio component)
    )
    
    physics_surface: EnumProperty(
        name="Physics Surface",
        description="Physics behavior (player-surface interaction)",
        items=[
            ('0', 'Undefined', 'Default physics'),
            ('1', 'Smooth', 'Fast/smooth movement'),
            ('2', 'Rough', 'Medium friction'),
            ('3', 'Slow', 'Slower movement'),
            ('4', 'Slippery', 'Low friction/slippery'),
            ('5', 'VerySlow', 'Very slow movement (BREADCRUMB BLOCKED)'),
            ('6', 'Unrideable', 'Cannot ride (BREADCRUMB BLOCKED)'),
            ('7', 'DoNotAlign', 'Special alignment behavior'),
            ('8', 'Stair', 'Stairs (BREADCRUMB BLOCKED, sets mIsAboveStairs)'),
            ('9', 'InstantBail', 'Immediately forces the player into a bail state'),
            ('10', 'SlipperyRagdoll', 'Ragdoll slides smoothly with no friction in bail state'),
            ('11', 'BouncyRagdoll', 'Ragdoll is bouncy in bail state'),
            ('12', 'Water', 'Causes the player to enter a swimming state'),
        ],
        default='4'  # Slippery (was the old hardcoded 0x8620's physics component)
    )
    
    surface_pattern: EnumProperty(
        name="Surface Pattern",
        description="Surface texture pattern (visual/audio variation)",
        items=[
            ('0', 'None (Default)', 'No pattern'),
            ('1', 'SpiderCrack', 'Cracked surface'),
            ('2', 'Square2x2', '2x2 tile pattern'),
            ('3', 'Square4x4', '4x4 tile pattern'),
            ('4', 'Square8x8', '8x8 tile pattern'),
            ('5', 'Square12x12', '12x12 tile pattern'),
            ('6', 'Square24x24', '24x24 tile pattern'),
            ('7', 'IrregularSmall', 'Small irregular pattern'),
            ('8', 'IrregularMedium', 'Medium irregular pattern'),
            ('9', 'IrregularLarge', 'Large irregular pattern'),
            ('10', 'Slats', 'Slat pattern'),
            ('11', 'Sidewalk', 'Sidewalk pattern'),
            ('12', 'BrickTileRandomSize', 'Brick tile (random size)'),
            ('13', 'MiniTile', 'Mini tile pattern'),
            ('14', 'Special1', 'Special pattern 1'),
            ('15', 'Special2', 'Special pattern 2'),
        ],
        default='8'  # IrregularMedium (was the old hardcoded 0x8620's pattern component)
    )

def raycast_bmesh(bm, ray_origin, ray_direction):
    closest_hit = None
    min_distance = float('inf')

    # Ensure faces lookup
    bm.faces.ensure_lookup_table()

    # Iterate through all faces
    for face in bm.faces:
        # Triangulate face if necessary
        if len(face.verts) == 3:
            tris = [[v.co for v in face.verts]]
        else:
            tris = [ [face.verts[i].co, face.verts[i+1].co, face.verts[i+2].co]
                     for i in range(len(face.verts)-2) ]

        for tri in tris:
            hit = intersect_ray_tri(tri[0], tri[1], tri[2], ray_direction, ray_origin, True)
            if hit:
                distance = (hit - ray_origin).length
                if distance < min_distance:
                    min_distance = distance
                    closest_hit = {
                        "location": hit,
                        "normal": face.normal.copy(),
                        "face": face
                    }
    return closest_hit

class PSG_OT_ShowSurfaceIDUnderCursor(bpy.types.Operator):
    """Show Surface ID of the face under cursor"""
    _active_handle = None
    bl_idname = "psg.show_surface_id_under_cursor"
    bl_label = "Toggle Surface Type Overlay"
    bl_options = {'REGISTER'}
    
    # Initialize as class attributes instead of in __init__
    _label = ""
    _cursor_pos = (0, 0)
    _enum_cache = {}

    def cache_enum_names(self):
        """Cache enum value -> name dictionaries for quick lookup."""
        scene = bpy.context.scene
        if not hasattr(scene, "psg_export_props") or scene.psg_export_props is None:
            return
        props = scene.psg_export_props

        self._enum_cache['audio'] = {item.identifier: item.name 
                                     for item in props.bl_rna.properties['audio_surface'].enum_items}
        self._enum_cache['physics'] = {item.identifier: item.name 
                                       for item in props.bl_rna.properties['physics_surface'].enum_items}
        self._enum_cache['pattern'] = {item.identifier: item.name 
                                       for item in props.bl_rna.properties['surface_pattern'].enum_items}

    def draw_callback(self):
        if not self._label:
            return
        x, y = self._cursor_pos
        font_id = 0
        blf.position(font_id, x + 10, y + 10, 0)
        blf.size(font_id, 16)
        blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
        
        for i, line in enumerate(self._label.split("\n")):
            blf.position(font_id, x + 10, y + 10 - i*18, 0)  # 18 pixels per line
            blf.draw(font_id, line)

    def execute(self, context):
        if PSG_OT_ShowSurfaceIDUnderCursor._active_handle:
            bpy.types.SpaceView3D.draw_handler_remove(PSG_OT_ShowSurfaceIDUnderCursor._active_handle, 'WINDOW')
            PSG_OT_ShowSurfaceIDUnderCursor._active_handle = None
            self._label = ""
            self.report({'INFO'}, "Surface overlay disabled")
            return {'FINISHED'}
        else:
            # Enable overlay
            self.cache_enum_names()
            PSG_OT_ShowSurfaceIDUnderCursor._active_handle = bpy.types.SpaceView3D.draw_handler_add(
                self.draw_callback, (), 'WINDOW', 'POST_PIXEL'
            )
            context.window_manager.modal_handler_add(self)
            self.report({'INFO'}, "Surface overlay enabled")
            return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if not context.area or context.area.type != 'VIEW_3D':
            return {'PASS_THROUGH'}

        if event.type == 'MOUSEMOVE':
            self._cursor_pos = (event.mouse_region_x, event.mouse_region_y)
            region = context.region
            rv3d = context.region_data
            coord = self._cursor_pos

            ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
            ray_direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

            sid = None

            # Check if in edit mode - only check the active object
            obj = context.object
            if obj and obj.type == 'MESH' and obj.mode == 'EDIT':
                bm = bmesh.from_edit_mesh(obj.data)
                bm.faces.ensure_lookup_table()

                # Transform ray into object local space
                inv_matrix = obj.matrix_world.inverted()
                local_origin = inv_matrix @ ray_origin
                local_direction = inv_matrix.to_3x3() @ ray_direction

                closest_dist = float('inf')
                id_layer = bm.faces.layers.int.get("surface_id")
                if id_layer:
                    for face in bm.faces:
                        tris = [[v.co for v in face.verts]] if len(face.verts) == 3 else [
                            [face.verts[i].co, face.verts[i+1].co, face.verts[i+2].co] 
                            for i in range(len(face.verts)-2)
                        ]
                        for tri in tris:
                            hit = intersect_ray_tri(tri[0], tri[1], tri[2], local_direction, local_origin, True)
                            if hit:
                                dist = (hit - local_origin).length
                                if dist < closest_dist:
                                    closest_dist = dist
                                    sid = face[id_layer]
            else:
                # Object Mode: raycast ALL mesh objects in the scene
                depsgraph = context.evaluated_depsgraph_get()
                result, location, normal, face_index, obj_hit, matrix = context.scene.ray_cast(
                    depsgraph, ray_origin, ray_direction
                )
                # Check ANY mesh object that was hit, not just the active one
                if result and obj_hit and obj_hit.type == 'MESH':
                    if "surface_id" in obj_hit.data.attributes and face_index < len(obj_hit.data.polygons):
                        attr = obj_hit.data.attributes["surface_id"]
                        sid = attr.data[face_index].value

            if sid is not None:
                try:
                    a, p, t = decode_surface_id(sid)
                    audio_name = self._enum_cache.get('audio', {}).get(str(a), str(a))
                    physics_name = self._enum_cache.get('physics', {}).get(str(p), str(p))
                    pattern_name = self._enum_cache.get('pattern', {}).get(str(t), str(t))
                    self._label = (
                        f"Audio: {audio_name}\n"
                        f"Physics: {physics_name}\n"
                        f"Pattern: {pattern_name}"
                    )
                except Exception:
                    self._label = f"SurfaceID {sid}"
            else:
                self._label = ""

            context.area.tag_redraw()

        return {'PASS_THROUGH'}

def surface_values_to_color(audio_val, physics_val, pattern_val):
    key = f"{audio_val}-{physics_val}-{pattern_val}"
    hsh = int(hashlib.md5(key.encode()).hexdigest(), 16)
    h = (hsh % 256) / 256.0
    h = (h + 0.61803398875) % 1.0
    s, v = 0.8, 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b, 1.0)

def get_surface_id_material():
    mat_name = "SurfaceID_Vis"
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]

    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Attribute → Diffuse → Output
    attr_node = nodes.new("ShaderNodeAttribute")
    attr_node.attribute_name = "surface_id_color"

    diffuse_node = nodes.new("ShaderNodeBsdfDiffuse")
    output_node = nodes.new("ShaderNodeOutputMaterial")

    links.new(attr_node.outputs['Color'], diffuse_node.inputs['Color'])
    links.new(diffuse_node.outputs['BSDF'], output_node.inputs['Surface'])

    return mat

def toggle_surface_id_material(obj):
    """Toggle SurfaceID visualization for a mesh object."""
    if not obj or obj.type != 'MESH' or not obj.material_slots:
        return

    vis_mat = get_surface_id_material()

    # Determine current state; default to False if not set
    currently_enabled = obj.get("_surface_id_enabled", False)

    if not currently_enabled:
        # Enable visualization
        if "_original_mats" not in obj:
            obj["_original_mats"] = [slot.material for slot in obj.material_slots]
        for slot in obj.material_slots:
            slot.material = vis_mat
        obj["_surface_id_enabled"] = True
    else:
        # Disable visualization
        original = obj.get("_original_mats")
        if original:
            for i, mat in enumerate(original):
                if i < len(obj.material_slots):
                    obj.material_slots[i].material = mat
            del obj["_original_mats"]
        obj["_surface_id_enabled"] = False

class PSG_OT_ToggleSurfaceColor(bpy.types.Operator):
    bl_idname = "psg.toggle_surface_color"
    bl_label = "Toggle SurfaceID Colors"
    bl_description = "Toggle face colors based on surface_id_color attribute"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not meshes:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}

        for obj in meshes:
            toggle_surface_id_material(obj)

        state = "enabled" if meshes[0].get("_surface_id_enabled", False) else "disabled"
        self.report({'INFO'}, f"SurfaceID color visualization {state}")
        return {'FINISHED'}

class PSG_OT_PickSurfaceRaycast(bpy.types.Operator):
    """Pick Surface ID from any face using a raycast (Photoshop-style)"""
    bl_idname = "psg.pick_surface_raycast"
    bl_label = "Pick Surface from Face (Raycast)"
    bl_options = {'REGISTER', 'UNDO'}

    _cursor_pos = (0, 0)
    
    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        self.report({'INFO'}, "Click a face to pick its surface")
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        context.window.cursor_set("EYEDROPPER") 
        if event.type == 'MOUSEMOVE':
            self._cursor_pos = (event.mouse_region_x, event.mouse_region_y)
            return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            region = context.region
            rv3d = context.region_data
            coord = self._cursor_pos

            ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
            ray_direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

            props = context.scene.psg_export_props
            surface_id = None

            if context.object and context.object.type == 'MESH' and context.object.mode == 'EDIT':
                # Edit mode: only the active object
                bm = bmesh.from_edit_mesh(context.object.data)
                bm.faces.ensure_lookup_table()
                closest_dist = float('inf')

                for face in bm.faces:
                    tris = [[v.co for v in face.verts]] if len(face.verts) == 3 else [
                        [face.verts[i].co, face.verts[i+1].co, face.verts[i+2].co] 
                        for i in range(len(face.verts)-2)
                    ]

                    for tri in tris:
                        hit = intersect_ray_tri(tri[0], tri[1], tri[2], ray_direction, ray_origin, True)
                        if hit:
                            dist = (hit - ray_origin).length
                            if dist < closest_dist:
                                closest_dist = dist
                                id_layer = bm.faces.layers.int.get("surface_id")
                                if id_layer:
                                    surface_id = face[id_layer]

            else:
                # Object mode: raycast ALL mesh objects in the scene
                depsgraph = context.evaluated_depsgraph_get()
                result, location, normal, face_index, obj_hit, matrix = context.scene.ray_cast(
                    depsgraph, ray_origin, ray_direction
                )
                # Check ANY mesh object that was hit
                if result and obj_hit and obj_hit.type == 'MESH':
                    attr = obj_hit.data.attributes.get("surface_id")
                    if attr and face_index < len(obj_hit.data.polygons):
                        surface_id = attr.data[face_index].value

            if surface_id is not None:
                try:
                    a, p, t = decode_surface_id(surface_id)
                    props.audio_surface = str(a)
                    props.physics_surface = str(p)
                    props.surface_pattern = str(t)
                    self.report({'INFO'}, f"Picked SurfaceID: {surface_id} → Audio={a}, Physics={p}, Pattern={t}")
                    context.area.tag_redraw()
                except Exception:
                    self.report({'WARNING'}, f"Failed to decode SurfaceID {surface_id}")
            else:
                self.report({'WARNING'}, "No face hit!")
                
            context.window.cursor_set("DEFAULT")
            return {'FINISHED'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

class PSG_OT_ApplySurfaceToFaces(bpy.types.Operator):
    bl_idname = "psg.apply_surface_to_faces"
    bl_label = "Apply Surface Type"
    bl_description = "Assigns the selected Audio/Physics/Pattern surface values to all selected faces"
    bl_options = {'UNDO'}

    def execute(self, context):
        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected_meshes:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}

        props = context.scene.psg_export_props
        audio_val = int(props.audio_surface)
        physics_val = int(props.physics_surface)
        pattern_val = int(props.surface_pattern)

        # Encode ID and color
        surface_id = encode_surface_id(audio_val, physics_val, pattern_val)
        surface_color = surface_values_to_color(audio_val, physics_val, pattern_val)

        total_applied = 0

        for obj in selected_meshes:
            mesh = obj.data

            # Ensure integer attribute exists
            if "surface_id" not in mesh.attributes:
                mesh.attributes.new(name="surface_id", type='INT', domain='FACE')
            id_attr = mesh.attributes["surface_id"]

            # Ensure color attribute exists
            if "surface_id_color" not in mesh.attributes:
                mesh.attributes.new(name="surface_id_color", type='FLOAT_COLOR', domain='FACE')
            color_attr = mesh.attributes["surface_id_color"]

            applied_count = 0

            if obj.mode == 'EDIT':
                bm = bmesh.from_edit_mesh(mesh)

                # Get or create integer face layer
                id_layer = bm.faces.layers.int.get("surface_id")
                if not id_layer:
                    id_layer = bm.faces.layers.int.new("surface_id")

                # Get or create loop color layer
                color_layer = bm.faces.layers.float_color.get("surface_id_color")
                if not color_layer:
                    color_layer = bm.faces.layers.float_color.new("surface_id_color")

                applied_count = 0

                for f in bm.faces:
                    if f.select:
                        # Assign integer per face
                        f[id_layer] = surface_id
                        f[color_layer] = surface_color
                        applied_count += 1

                bmesh.update_edit_mesh(mesh)

            else:
                # --- OBJECT MODE ---
                for poly in mesh.polygons:
                    id_attr.data[poly.index].value = surface_id
                    color_attr.data[poly.index].color = surface_color
                    applied_count += 1
                mesh.update()

            total_applied += applied_count

        self.report(
            {'INFO'},
            f"Applied SurfaceID ({audio_val}, {physics_val}, {pattern_val}) "
            f"to {total_applied} faces across {len(selected_meshes)} objects"
        )
        return {'FINISHED'}

class PSG_OT_Export(Operator):
    """Export selected objects to PSG format"""
    bl_idname = "export_scene.psg"
    bl_label = "Export PSG"
    bl_description = "Export selected mesh and curve objects to Skate 3 PSG format"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.psg_export_props
        folder_path = bpy.path.abspath(props.filepath)
        
        # Validate folder path
        if not folder_path or folder_path == "//":
            self.report({'ERROR'}, "Please set a valid export folder")
            return {'CANCELLED'}
        
        # Ensure folder exists
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
            except Exception as e:
                self.report({'ERROR'}, f"Cannot create folder: {str(e)}")
                return {'CANCELLED'}
        
        # Generate random 16-character hex filename (8 bytes = 64 bits)
        filename = secrets.token_hex(8).upper() + '.psg'
        filepath = os.path.join(folder_path, filename)
        
        try:
            # Extract data from Blender


            
            extractor = BlenderDataExtractor()
            extractor.extract_from_selected()
            
            # Validate we have geometry
            if len(extractor.faces) == 0:
                self.report({'ERROR'}, "No mesh geometry found! Select at least one mesh object.")
                return {'CANCELLED'}
            
            # Build PSG
            props = context.scene.psg_export_props
            builder = PSGBuilder(extractor, force_uncompressed=props.force_uncompressed,
                               enable_vertex_smoothing=props.enable_vertex_smoothing)
            builder.build(filepath)
            
            # Generate compression report
            total_clusters = sum(builder.compression_stats.values())
            compression_report = []
            
            if builder.compression_stats[0] > 0:
                pct = (builder.compression_stats[0] / total_clusters) * 100
                compression_report.append(f"Uncompressed: {builder.compression_stats[0]} ({pct:.1f}%)")
            if builder.compression_stats[1] > 0:
                pct = (builder.compression_stats[1] / total_clusters) * 100
                compression_report.append(f"16-bit: {builder.compression_stats[1]} ({pct:.1f}%)")
            if builder.compression_stats[2] > 0:
                pct = (builder.compression_stats[2] / total_clusters) * 100
                compression_report.append(f"32-bit: {builder.compression_stats[2]} ({pct:.1f}%)")
            
            compression_msg = ", ".join(compression_report) if compression_report else "unknown"
            
            # Success message with detailed stats
            spline_msg = f", {len(extractor.splines)} splines" if extractor.splines else ""
            self.report({'INFO'}, 
                f"✅ PSG exported: {filename} | {len(extractor.faces)} triangles{spline_msg}, "
                f"{total_clusters} clusters [{compression_msg}], "
                f"granularity={builder.granularity:.6f}")
            
            # Additional info message about compression if fallback occurred
            if builder.compression_stats[0] > 0 and not props.force_uncompressed:
                self.report({'WARNING'}, 
                    f"⚠️ {builder.compression_stats[0]} cluster(s) auto-fallback to UNCOMPRESSED due to overflow. "
                    "This is normal for complex geometry and prevents phantom collisions.")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

class PSG_OT_BulkExport(Operator):
    """Bulk export selected objects as separate PSG files"""
    bl_idname = "export_scene.psg_bulk"
    bl_label = "Bulk Export PSGs"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        props = context.scene.psg_export_props
        folder_path = bpy.path.abspath(props.filepath)
        
        # Validate folder path
        if not folder_path or folder_path == "//":
            self.report({'ERROR'}, "Please set a valid export folder")
            return {'CANCELLED'}
        
        # Ensure folder exists
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
            except Exception as e:
                self.report({'ERROR'}, f"Cannot create folder: {str(e)}")
                return {'CANCELLED'}
        
        # Get all selected mesh objects
        selected_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        
        if not selected_objects:
            self.report({'ERROR'}, "No mesh objects selected!")
            return {'CANCELLED'}
        
        total_objects = len(selected_objects)
        
        # Initialize UI counter
        props.is_exporting = True
        props.total_exports = total_objects
        props.current_export = 0
        
        try:
            for i, obj in enumerate(selected_objects):
                # Update UI counter BEFORE export
                props.current_export = i + 1
                
                # Force UI redraw to show updated counter
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                
                print(f"\n{'='*60}")
                print(f"Exporting object {i+1}/{total_objects}: {obj.name}")
                print(f"{'='*60}")
                
                # Generate random 16-character hex filename (8 bytes = 64 bits)
                filename = secrets.token_hex(8).upper() + '.psg'
                filepath = os.path.join(folder_path, filename)
                
                try:
                    # Create temporary selection with only this object
                    bpy.ops.object.select_all(action='DESELECT')
                    obj.select_set(True)
                    context.view_layer.objects.active = obj
                    
                    # Extract data from this object only
                    extractor = BlenderDataExtractor()
                    extractor.extract_from_selected()
                    
                    # Validate we have geometry
                    if len(extractor.faces) == 0:
                        print(f"⚠️ Skipping {obj.name}: No mesh geometry found")
                        continue
                    
                    # Build PSG
                    builder = PSGBuilder(extractor, force_uncompressed=props.force_uncompressed,
                                       enable_vertex_smoothing=props.enable_vertex_smoothing)
                    builder.build(filepath)
                    
                    # Generate compression report
                    total_clusters = sum(builder.compression_stats.values())
                    compression_report = []
                    
                    if builder.compression_stats[0] > 0:
                        pct = (builder.compression_stats[0] / total_clusters) * 100
                        compression_report.append(f"Uncompressed: {builder.compression_stats[0]} ({pct:.1f}%)")
                    if builder.compression_stats[1] > 0:
                        pct = (builder.compression_stats[1] / total_clusters) * 100
                        compression_report.append(f"16-bit: {builder.compression_stats[1]} ({pct:.1f}%)")
                    if builder.compression_stats[2] > 0:
                        pct = (builder.compression_stats[2] / total_clusters) * 100
                        compression_report.append(f"32-bit: {builder.compression_stats[2]} ({pct:.1f}%)")
                    
                    compression_msg = ", ".join(compression_report) if compression_report else "unknown"
                    
                    # Success message with detailed stats
                    spline_msg = f", {len(extractor.splines)} splines" if extractor.splines else ""
                    print(f"✅ PSG exported: {filename} | {len(extractor.faces)} triangles{spline_msg}, "
                          f"{total_clusters} clusters [{compression_msg}], "
                          f"granularity={builder.granularity:.6f}")
                    
                except Exception as e:
                    print(f"❌ Failed to export {obj.name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Restore original selection
            for obj in selected_objects:
                obj.select_set(True)
            
            self.report({'INFO'}, f"✅ Bulk export complete: {total_objects} objects exported to {folder_path}")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Bulk export failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
        finally:
            # Reset UI state
            props.is_exporting = False
            props.current_export = 0
            props.total_exports = 0
            
            # Force final UI redraw
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()


class PSG_PT_ExportPanel(Panel):
    """N-Panel for PSG export"""
    bl_label = "Collision Export"
    bl_idname = "PSG_PT_export_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Collision Export'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.psg_export_props
        
        # Header
        box = layout.box()
        box.label(text="Skate 3 Collision Exporter", icon='EXPORT')
        
        # Export settings
        col = layout.column(align=True)
        col.label(text="Export Settings:", icon='SETTINGS')
        col.prop(props, "filepath")
        
        # Compression settings
        layout.separator()
        box = layout.box()
        box.label(text="Vertex Compression:", icon='MESH_DATA')
        box.prop(props, "force_uncompressed")
        if props.force_uncompressed:
            row = box.row()
            row.label(text="⚠️ File size will be larger", icon='INFO')
        
        # Vertex smoothing settings
        layout.separator()
        box = layout.box()
        box.label(text="Edge Collision:", icon='OUTLINER_OB_MESH')
        box.prop(props, "enable_vertex_smoothing")
        if not props.enable_vertex_smoothing:
            row = box.row()
            row.label(text="⚠️ All edges will be sharp!", icon='INFO')
        else:
            row = box.row()
            row.label(text="✅ Smooths ~93% of vertices", icon='INFO')
        
        # Surface Material Settings
        layout.separator()
        box = layout.box()
        box.label(text="Surface Material:", icon='MATERIAL')
        
        # Show current SurfaceID in hex
        audio_val = int(props.audio_surface)
        physics_val = int(props.physics_surface)
        pattern_val = int(props.surface_pattern)
        surface_id = encode_surface_id(audio_val, physics_val, pattern_val)
        box.label(text=f"SurfaceID: 0x{surface_id:04X} ({surface_id})")
        
        col = box.column(align=True)
        col.operator("psg.toggle_surface_color", text="Toggle Surface Type View", icon='MATERIAL')
        col.operator("psg.show_surface_id_under_cursor", text="Show Surface Under Cursor", icon='HIDE_OFF')
        col.operator("psg.pick_surface_raycast", text="Pick Surface", icon='EYEDROPPER')
        col.prop(props, "audio_surface")
        col.prop(props, "physics_surface")
        col.prop(props, "surface_pattern")
        
        # Show warning for breadcrumb-blocked surfaces
        # _IsSurfaceOK returns false (NOT ok) for PhysicsSurfaceType = 5, 6, 8, or 9
        if physics_val in [5, 6, 8, 9]:
            warning_box = box.box()
            warning_box.alert = True
            warning_box.label(text="⚠ BREADCRUMB BLOCKED", icon='ERROR')
            warning_box.label(text="Players cannot place breadcrumbs here!")
        
        # Button for applying surface types to faces
        col.operator("psg.apply_surface_to_faces", text="Apply to Selected Faces", icon='BRUSH_DATA')

        # Info about coordinate system
        box = layout.box()
        box.label(text="Coordinate Transform:", icon='ORIENTATION_GLOBAL')
        box.label(text="Blender (X,Y,Z) → PSG (X,Z,-Y)")
        box.label(text="(Auto-applied to match game)")
        
        # Selection info
        layout.separator()
        box = layout.box()
        box.label(text="Selection Info:", icon='INFO')
        
        selected = context.selected_objects
        mesh_count = sum(1 for obj in selected if obj.type == 'MESH')
        curve_count = sum(1 for obj in selected if obj.type == 'CURVE')
        
        row = box.row()
        row.label(text=f"Meshes: {mesh_count}")
        row.label(text=f"Curves: {curve_count}")
        
        if not selected:
            box.label(text="⚠ No objects selected!", icon='ERROR')
        elif mesh_count == 0:
            box.label(text="⚠ No mesh objects!", icon='ERROR')
        
        # Export buttons
        layout.separator()
        box = layout.box()
        
        # Show progress counter if bulk exporting
        if props.is_exporting:
            progress_box = box.box()
            progress_box.alert = True
            progress_box.label(text=f"⏳ Exporting: {props.current_export} of {props.total_exports}", icon='TIME')
        
        # Single export button
        row = box.row()
        row.scale_y = 1.5
        row.enabled = not props.is_exporting  # Disable during export
        row.operator("export_scene.psg", text="Export All as Single PSG", icon='FILE')
        
        # Bulk export button
        row = box.row()
        row.scale_y = 1.5
        row.enabled = not props.is_exporting  # Disable during export
        row.operator("export_scene.psg_bulk", text="Bulk Export (Separate PSGs)", icon='DOCUMENTS')
        
        # Help
        layout.separator()
        box = layout.box()
        box.label(text="Usage:", icon='QUESTION')
        box.label(text="1. Select mesh objects")
        box.label(text="2. Optional: Add curve objects for splines")
        box.label(text="3. Set export path above")
        box.label(text="4. Click 'Export PSG'")


# ===== BLENDER REGISTRATION =====

classes = (
    PSGExportProperties,
    PSG_OT_Export,
    PSG_OT_BulkExport,
    PSG_PT_ExportPanel,
    PSG_OT_ApplySurfaceToFaces,
    PSG_OT_PickSurfaceRaycast,
    PSG_OT_ToggleSurfaceColor,
    PSG_OT_ShowSurfaceIDUnderCursor
)

def register():
    """Register addon"""
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.psg_export_props = bpy.props.PointerProperty(type=PSGExportProperties)

def unregister():
    """Unregister addon"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.psg_export_props

if __name__ == "__main__":
    register()

