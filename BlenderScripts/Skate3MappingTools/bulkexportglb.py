bl_info = {
    "name": "Unique Mesh Batch GLB Exporter (with Bounds JSON, Material Split & Safe Splitting)",
    "author": "Wissp + ChatGPT",
    "version": (1, 8),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Batch Export",
    "description": "Exports each unique mesh as GLB, splits by material and size, writes world-space bounds",
    "category": "Import-Export",
}

import bpy
import os
import json
import bmesh
from mathutils import Vector

MAX_INDEX_SUM = 65535 # V + E + F limit

# -------------------- Helpers --------------------
def combined_index_count(mesh):
    return len(mesh.vertices) + len(mesh.edges) + len(mesh.polygons)

def split_polygon_indices(mesh, limit=MAX_INDEX_SUM):
    poly_chunks = []
    current_chunk = []
    used_verts = set()
    used_edges = set()
    current_count = 0

    for poly in mesh.polygons:
        poly_verts = set(poly.vertices)
        poly_edges = set(poly.edge_keys)
        
        new_verts = len(poly_verts - used_verts)
        new_edges = len(poly_edges - used_edges)
        new_faces = 1

        added_count = new_verts + new_edges + new_faces

        if current_count + added_count > limit and current_chunk:
            poly_chunks.append(current_chunk)
            current_chunk = []
            used_verts = set()
            used_edges = set()
            current_count = 0

        current_chunk.append(poly.index)
        used_verts.update(poly_verts)
        used_edges.update(poly_edges)
        current_count += added_count

    if current_chunk:
        poly_chunks.append(current_chunk)

    return poly_chunks

def create_temp_chunk_object(source_obj, poly_indices, name):
    mesh = source_obj.data.copy()
    mesh.name = f"{name}_mesh"

    bm = bmesh.new()
    bm.from_mesh(mesh)

    keep = set(poly_indices)

    for face in list(bm.faces):
        if face.index not in keep:
            bm.faces.remove(face)

    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    bmesh.ops.delete(bm, geom=[e for e in bm.edges if not e.link_faces], context='EDGES')

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()

    temp_obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(temp_obj)

    temp_obj.data.update(calc_edges=True, calc_edges_loose=False)
    temp_obj.data.validate(clean_customdata=True)

    return temp_obj

def split_mesh_by_material(source_obj):
    """Split mesh into temporary objects per material."""
    mats = [i for i, m in enumerate(source_obj.data.materials) if m]
    if not mats:
        return [(None, source_obj)]
    temp_objects = []
    for mat_idx in mats:
        poly_indices = [p.index for p in source_obj.data.polygons if p.material_index == mat_idx]
        if poly_indices:
            temp_obj = create_temp_chunk_object(source_obj, poly_indices, f"{source_obj.name}_mat{mat_idx}")
            temp_objects.append((mat_idx, temp_obj))
    return temp_objects

# -------------------- Panel --------------------
class BATCHGLB_PT_panel(bpy.types.Panel):
    bl_label = "Batch GLB Exporter"
    bl_idname = "BATCHGLB_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Batch Export'

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "batch_glb_export_path")
        layout.prop(context.scene, "batch_glb_use_draco")
        layout.operator("export.unique_mesh_glb", text="Export Unique Meshes")

# -------------------- Operator --------------------
class EXPORT_OT_unique_mesh_glb(bpy.types.Operator):
    bl_idname = "export.unique_mesh_glb"
    bl_label = "Export Unique Mesh GLBs"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        export_dir = bpy.path.abspath(context.scene.batch_glb_export_path)
        os.makedirs(export_dir, exist_ok=True)

        for f in os.listdir(export_dir):
            if f.lower().endswith(".glb"):
                try:
                    os.remove(os.path.join(export_dir, f))
                except:
                    pass

        unique_meshes = set(o.data for o in bpy.data.objects if o.type == 'MESH' and o.visible_get() and o.data)
        bounds_data = {}
        exported = 0
        failed = 0
        total = len(unique_meshes)

        for mesh in unique_meshes:
            obj = next((o for o in bpy.data.objects if o.data == mesh and o.type == 'MESH'), None)
            if not obj:
                continue

            safe_name = bpy.path.clean_name(mesh.name)
            eval_obj = obj.evaluated_get(context.evaluated_depsgraph_get())
            temp_objects = []

            # Write bounds
            try:
                corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
                bounds_data[safe_name] = {
                    "min": [min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)],
                    "max": [max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)],
                }
            except Exception as e:
                self.report({'WARNING'}, f"Failed bounds for {mesh.name}: {e}")

            try:
                # Split by material first
                material_chunks = split_mesh_by_material(eval_obj)
                for mat_idx, mat_obj in material_chunks:
                    index_sum = combined_index_count(mat_obj.data)
                    if index_sum <= MAX_INDEX_SUM:
                        # Export single material object
                        bpy.ops.object.select_all(action='DESELECT')
                        mat_obj.select_set(True)
                        context.view_layer.objects.active = mat_obj
                        part_name = f"{safe_name}" if mat_idx is None else f"{safe_name}_mat{mat_idx}"
                        filepath = os.path.join(export_dir, f"{part_name}.glb")
                        bpy.ops.export_scene.gltf(
                            filepath=filepath,
                            export_format='GLB',
                            use_selection=True,
                            export_apply=True,
                            export_yup=True,
                            export_normals=True,
                            export_tangents=True,
                            export_texcoords=True,
                            export_materials='EXPORT',
                            export_draco_mesh_compression_enable=context.scene.batch_glb_use_draco
                        )
                        exported += 1
                        self.report({'INFO'}, f"Exported {part_name}.glb (V+E+F={index_sum})")

                        # Record bounds for this exported GLB
                        corners = [mat_obj.matrix_world @ Vector(corner) for corner in mat_obj.bound_box]
                        bounds_data[f"{part_name}"] = {
                            "min": [min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)],
                            "max": [max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)],
                        }
                    else:
                        # Split large mesh by polygons
                        chunks = split_polygon_indices(mat_obj.data, MAX_INDEX_SUM)
                        for i, poly_list in enumerate(chunks):
                            temp_obj = create_temp_chunk_object(mat_obj, poly_list, f"{mat_obj.name}_part{i}")
                            temp_objects.append((temp_obj.name, temp_obj.data.name))
                            bpy.ops.object.select_all(action='DESELECT')
                            temp_obj.select_set(True)
                            context.view_layer.objects.active = temp_obj
                            part_path = os.path.join(export_dir, f"{temp_obj.name}.glb")
                            bpy.ops.export_scene.gltf(
                                filepath=part_path,
                                export_format='GLB',
                                use_selection=True,
                                export_apply=True,
                                export_yup=True,
                                export_normals=True,
                                export_tangents=True,
                                export_texcoords=True,
                                export_materials='EXPORT',
                                export_draco_mesh_compression_enable=context.scene.batch_glb_use_draco
                            )
                            exported += 1
                            self.report({'INFO'}, f"Exported {temp_obj.name}.glb")

                            # Record bounds for this exported GLB
                            corners = [temp_obj.matrix_world @ Vector(corner) for corner in temp_obj.bound_box]
                            bounds_data[f"{temp_obj.name}.glb"] = {
                                "min": [min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)],
                                "max": [max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)],
                            }
            except Exception as e:
                failed += 1
                self.report({'ERROR'}, f"Failed exporting {safe_name}: {e}")

        # Cleanup temporary objects
        for obj_name, mesh_name in temp_objects:
            if obj_name in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
            if mesh_name in bpy.data.meshes:
                bpy.data.meshes.remove(bpy.data.meshes[mesh_name])

        # Also remove material-chunk objects safely
        for mat_idx, mat_obj in material_chunks:
            # Store mesh data reference before removing object
            mesh_data = mat_obj.data
            if mat_obj.name in bpy.data.objects and mat_obj.name != obj.name:
                bpy.data.objects.remove(mat_obj, do_unlink=True)
            if mesh_data.name in bpy.data.meshes:
                bpy.data.meshes.remove(mesh_data)

        # Write bounds JSON
        json_path = os.path.join(export_dir, "glb_bounds.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(bounds_data, f, indent=2)
            self.report({'INFO'}, f"Wrote {json_path}")
        except Exception as e:
            self.report({'ERROR'}, f"Could not write glb_bounds.json: {e}")

        self.report({'INFO'}, f"Done: {exported} exported, {failed} failed.")
        return {'FINISHED'}

# -------------------- Register --------------------
def register():
    bpy.utils.register_class(BATCHGLB_PT_panel)
    bpy.utils.register_class(EXPORT_OT_unique_mesh_glb)
    bpy.types.Scene.batch_glb_export_path = bpy.props.StringProperty(
        name="Export Folder",
        description="Folder where .glb files will be written",
        subtype='DIR_PATH'
    )
    bpy.types.Scene.batch_glb_use_draco = bpy.props.BoolProperty(
        name="Use Draco Compression",
        description="Enable Draco compression for GLB export",
        default=False
    )

def unregister():
    bpy.utils.unregister_class(BATCHGLB_PT_panel)
    bpy.utils.unregister_class(EXPORT_OT_unique_mesh_glb)
    del bpy.types.Scene.batch_glb_export_path
    del bpy.types.Scene.batch_glb_use_draco

if __name__ == "__main__":
    register()
