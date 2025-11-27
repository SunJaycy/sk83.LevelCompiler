bl_info = {
    "name": "Skate – Generate Splines (Edges & Vertices)",
    "author": "Ethan + ChatGPT",
    "version": (1, 1, 0),
    "blender": (3, 6, 0),
    "location": "3D Viewport > N-Panel > Spline",
    "description": "In Edit Mode, creates splines from selected edges OR between selected vertices (straight line for 2 verts, bezier curve for 3+ verts).",
    "category": "Mesh",
}

import bpy
import bmesh
from mathutils import Vector

# ----------------------------
# Helpers
# ----------------------------
def _safe_normalize(v: Vector) -> Vector:
    l = v.length
    return v / l if l > 1e-12 else Vector((0.0, 0.0, 0.0))

def _edge_key(e):
    a, b = e.verts[0].index, e.verts[1].index
    return (a, b) if a < b else (b, a)

def _edge_dir(v0: Vector, v1: Vector) -> Vector:
    return _safe_normalize(v1 - v0)

def _avg_face_normal_for_edge(edge) -> Vector:
    """Average of adjacent face normals; if none, zero vector."""
    if not edge.link_faces:
        return Vector((0.0, 0.0, 0.0))
    acc = Vector((0.0, 0.0, 0.0))
    for f in edge.link_faces:
        acc += f.normal
    return _safe_normalize(acc)

def _project_normal_perp_to_edge(n: Vector, e_dir: Vector) -> Vector:
    """Remove any component along the edge; return a vector perpendicular to the edge."""
    # n_perp = n - (n·e)e
    n_perp = n - e_dir * n.dot(e_dir)
    return _safe_normalize(n_perp)

def _vertex_fallback_normal(v) -> Vector:
    # Use BMVert normal if available; else zero.
    try:
        return _safe_normalize(v.normal.copy())
    except Exception:
        return Vector((0.0, 0.0, 0.0))

def _compute_offset_dir_for_edge(edge) -> Vector:
    """
    Robust offset direction:
    1) average adjacent face normals
    2) project perpendicular to edge
    3) if degenerate, fall back to averaged vertex normals then project
    """
    v0 = edge.verts[0].co
    v1 = edge.verts[1].co
    e_dir = _edge_dir(v0, v1)

    n = _avg_face_normal_for_edge(edge)
    n_perp = _project_normal_perp_to_edge(n, e_dir)
    if n_perp.length > 0.0:
        return n_perp

    # Fallback: average vertex normals
    vn0 = _vertex_fallback_normal(edge.verts[0])
    vn1 = _vertex_fallback_normal(edge.verts[1])
    n2 = _safe_normalize(vn0 + vn1)
    n2_perp = _project_normal_perp_to_edge(n2, e_dir)
    if n2_perp.length > 0.0:
        return n2_perp

    # Final fallback: arbitrary perpendicular (try world Z crossed with edge)
    z = Vector((0.0, 0.0, 1.0))
    guess = _safe_normalize(z.cross(e_dir))
    if guess.length == 0.0:
        # edge is parallel to Z; use X axis cross
        guess = _safe_normalize(Vector((1.0, 0.0, 0.0)).cross(e_dir))
    return guess

def _make_poly_curve(name: str, world_pt0: Vector, world_pt1: Vector) -> bpy.types.Object:
    """Create a straight line poly curve between two points."""
    crv = bpy.data.curves.new(name=name, type='CURVE')
    crv.dimensions = '3D'
    sp = crv.splines.new(type='POLY')
    sp.points.add(1)  # total 2 points
    sp.points[0].co = (world_pt0.x, world_pt0.y, world_pt0.z, 1.0)
    sp.points[1].co = (world_pt1.x, world_pt1.y, world_pt1.z, 1.0)
    sp.use_cyclic_u = False
    obj = bpy.data.objects.new(name, crv)
    return obj

def _make_bezier_curve(name: str, world_points: list) -> bpy.types.Object:
    """Create a bezier curve through multiple control points."""
    crv = bpy.data.curves.new(name=name, type='CURVE')
    crv.dimensions = '3D'
    sp = crv.splines.new(type='BEZIER')
    
    # Add points (bezier splines need n-1 points for n control points)
    sp.bezier_points.add(len(world_points) - 1)
    
    for i, pt in enumerate(world_points):
        bp = sp.bezier_points[i]
        bp.co = (pt.x, pt.y, pt.z)
        bp.handle_left_type = 'AUTO'
        bp.handle_right_type = 'AUTO'
    
    sp.use_cyclic_u = False
    obj = bpy.data.objects.new(name, crv)
    return obj

def _make_vertex_spline(name: str, world_points: list) -> bpy.types.Object:
    """Create appropriate spline type based on number of points."""
    if len(world_points) == 2:
        return _make_poly_curve(name, world_points[0], world_points[1])
    else:
        return _make_bezier_curve(name, world_points)

def _get_vertices_in_spatial_order(obj, selected_verts):
    """
    Order vertices spatially to create a logical spline path.
    This is much more reliable than trying to guess selection order.
    """
    if len(selected_verts) <= 2:
        return selected_verts
    
    # Convert to world coordinates
    M = obj.matrix_world
    world_points = [(M @ v.co, v) for v in selected_verts]
    
    # Find the two vertices that are farthest apart (start and end)
    max_dist = 0
    start_idx = 0
    end_idx = 1
    
    for i in range(len(world_points)):
        for j in range(i + 1, len(world_points)):
            dist = (world_points[i][0] - world_points[j][0]).length
            if dist > max_dist:
                max_dist = dist
                start_idx = i
                end_idx = j
    
    # Start with the two farthest vertices
    ordered_verts = [world_points[start_idx][1], world_points[end_idx][1]]
    remaining_points = [world_points[i] for i in range(len(world_points)) if i != start_idx and i != end_idx]
    
    # Insert remaining vertices at the position that minimizes total path length
    while remaining_points:
        best_insert_idx = 0
        best_insert_point = remaining_points[0]
        min_total_length = float('inf')
        
        # Try inserting each remaining point at each possible position
        for point in remaining_points:
            for insert_idx in range(1, len(ordered_verts) + 1):
                # Calculate total path length if we insert this point here
                test_order = ordered_verts[:insert_idx] + [point[1]] + ordered_verts[insert_idx:]
                total_length = 0
                
                for i in range(len(test_order) - 1):
                    pos1 = M @ test_order[i].co
                    pos2 = M @ test_order[i + 1].co
                    total_length += (pos1 - pos2).length
                
                if total_length < min_total_length:
                    min_total_length = total_length
                    best_insert_idx = insert_idx
                    best_insert_point = point
        
        # Insert the best point at the best position
        ordered_verts.insert(best_insert_idx, best_insert_point[1])
        remaining_points.remove(best_insert_point)
    
    return ordered_verts

# ----------------------------
# Properties & UI
# ----------------------------
class SkateSimpleProps(bpy.types.PropertyGroup):
    offset: bpy.props.FloatProperty(
        name="Offset",
        description="Distance to lift the spline off the surface (scene units)",
        min=0.0, soft_max=0.5, default=0.01
    )
    spline_mode: bpy.props.EnumProperty(
        name="Spline Mode",
        description="Choose what to create splines from",
        items=[
            ('EDGES', 'From Edges', 'Create splines from selected edges (offset from surface)'),
            ('VERTICES', 'From Vertices', 'Create splines between selected vertices (straight line for 2, bezier curve for 3+)'),
        ],
        default='EDGES'
    )

class VIEW3D_PT_SkateSimplePanel(bpy.types.Panel):
    bl_label = "Spline"
    bl_idname = "VIEW3D_PT_skate_simple_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Spline"

    def draw(self, context):
        layout = self.layout
        props = context.scene.skate_simple_props
        col = layout.column(align=True)
        
        # Mode selector
        col.prop(props, "spline_mode", text="Mode")
        
        # Show offset only for edge mode
        if props.spline_mode == 'EDGES':
            col.prop(props, "offset", text="Offset")
            col.label(text="Select edges in Edit Mode", icon='INFO')
        else:
            col.label(text="Select 2+ vertices in Edit Mode", icon='INFO')
            col.label(text="2 verts = straight line", icon='CURVE_DATA')
            col.label(text="3+ verts = bezier curve", icon='CURVE_BEZCURVE')
            col.label(text="Spatial ordering (auto)", icon='INFO')
        
        col.operator("skate.generate_offset_splines", icon='CURVE_DATA', text="Generate Splines")

# ----------------------------
# Main Operator
# ----------------------------
class SKATE_OT_GenerateOffsetSplines(bpy.types.Operator):
    bl_idname = "skate.generate_offset_splines"
    bl_label = "Generate Splines"
    bl_description = "Create splines from selected edges OR between selected vertices (Edit Mode)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == 'MESH' and context.mode == 'EDIT_MESH'

    def execute(self, context):
        obj = context.active_object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.normal_update()  # ensure face/vert normals are valid
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()

        props = context.scene.skate_simple_props
        M = obj.matrix_world
        made = 0

        if props.spline_mode == 'EDGES':
            # Original edge-based spline generation
            d = float(props.offset)
            sel_edges = [e for e in bm.edges if e.select]
            
            if not sel_edges:
                self.report({'ERROR'}, "Select one or more edges in Edit Mode.")
                return {'CANCELLED'}

            for i, e in enumerate(sel_edges):
                v0 = e.verts[0].co
                v1 = e.verts[1].co
                n_off = _compute_offset_dir_for_edge(e)

                # offset both endpoints outward by d
                p0 = v0 + n_off * d
                p1 = v1 + n_off * d

                wp0 = M @ p0
                wp1 = M @ p1

                curve_obj = _make_poly_curve(f"Spline_from_{obj.name}_{i:04d}", wp0, wp1)
                context.collection.objects.link(curve_obj)

                # simple metadata for downstream use
                curve_obj["source_object"] = obj.name
                curve_obj["edge_index_a"] = e.verts[0].index
                curve_obj["edge_index_b"] = e.verts[1].index
                curve_obj["offset"] = d
                curve_obj["spline_type"] = "edge_offset"

                made += 1

        else:
            # New vertex-based spline generation
            sel_verts = [v for v in bm.verts if v.select]
            
            if len(sel_verts) < 2:
                self.report({'ERROR'}, "Select 2 or more vertices in Edit Mode.")
                return {'CANCELLED'}

            # CRITICAL: Get vertices in spatial order (much more reliable)
            ordered_verts = _get_vertices_in_spatial_order(obj, sel_verts)
            
            # Convert vertex coordinates to world space in correct order
            world_points = [M @ v.co for v in ordered_verts]
            
            # Create spline name based on number of vertices
            spline_type = "straight" if len(ordered_verts) == 2 else "bezier"
            curve_obj = _make_vertex_spline(f"Spline_{spline_type}_{obj.name}_{len(ordered_verts)}pts", world_points)
            context.collection.objects.link(curve_obj)

            # Metadata for downstream use
            curve_obj["source_object"] = obj.name
            curve_obj["vertex_indices"] = [v.index for v in ordered_verts]
            curve_obj["spline_type"] = spline_type
            curve_obj["point_count"] = len(ordered_verts)

            made = 1

        # No destructive mesh changes; refresh viewport anyway
        bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
        
        mode_text = "edge offset" if props.spline_mode == 'EDGES' else "vertex-based"
        self.report({'INFO'}, f"Created {made} {mode_text} spline(s).")
        return {'FINISHED'}

# ----------------------------
# Register
# ----------------------------
classes = (
    SkateSimpleProps,
    VIEW3D_PT_SkateSimplePanel,
    SKATE_OT_GenerateOffsetSplines,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.skate_simple_props = bpy.props.PointerProperty(type=SkateSimpleProps)

def unregister():
    del bpy.types.Scene.skate_simple_props
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
