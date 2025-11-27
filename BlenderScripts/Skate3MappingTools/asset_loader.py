# skate_tools/asset_loader.py
import bpy
import os
from bpy.app.handlers import persistent

ASSET_REL_PATH = os.path.join("assets", "reference.glb")
_LOADED_FLAG = "_skate_reference_loaded"   # per-scene "already imported" flag
_CLEANED_FLAG = "_skate_startup_cleaned"   # per-scene "removed cube/light/camera" flag

def _addon_dir() -> str:
    return os.path.dirname(__file__)

def _asset_path() -> str:
    return os.path.join(_addon_dir(), ASSET_REL_PATH)

def _remove_default_starters(scene):
    """Remove default Cube/Light/Camera exactly once per .blend."""
    if scene.get(_CLEANED_FLAG):
        return
    names = {"Cube", "Light", "Camera"}
    found = [obj for obj in scene.objects if obj.name in names]
    if found:
        # Try operator path first
        try:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in found:
                try:
                    obj.select_set(True)
                except Exception:
                    pass
            bpy.ops.object.delete()
        except Exception:
            # Fallback unlink/remove if operators not available (e.g., background mode)
            for obj in found:
                for coll in list(obj.users_collection):
                    try:
                        coll.objects.unlink(obj)
                    except Exception:
                        pass
                try:
                    bpy.data.objects.remove(obj)
                except Exception:
                    pass
    scene[_CLEANED_FLAG] = True

def _import_reference_into_scene():
    """Import assets/reference.glb into the active scene without creating any collections."""
    scene = bpy.context.scene
    if scene is None:
        return

    # First-start cleanup
    _remove_default_starters(scene)

    # Avoid duplicate import within the same scene/session
    if scene.get(_LOADED_FLAG):
        return

    path = _asset_path()
    if not os.path.exists(path):
        print(f"[Skate Tools] Bundled GLB not found at {path}. "
              f"Place your file as assets/reference.glb")
        return

    try:
        bpy.ops.import_scene.gltf(filepath=path)
        imported = list(bpy.context.selected_objects)
        # Drop at origin
        for obj in imported:
            obj.location = (0.0, 0.0, 0.0)
        print(f"[Skate Tools] Imported bundled reference from {path}")
    except Exception as e:
        print(f"[Skate Tools] Failed to import reference.glb: {e}")

    scene[_LOADED_FLAG] = True

# --- Operator (re-usable from Add > Mesh) ---

class SKATE_OT_LoadBundledGLB(bpy.types.Operator):
    """Import the bundled reference.glb into the scene (no collections created)."""
    bl_idname = "skate.load_bundled_glb"
    bl_label = "Skate Reference"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        path = _asset_path()
        if not os.path.exists(path):
            self.report({'ERROR'}, f"Bundled GLB not found: {path}")
            return {'CANCELLED'}

        try:
            bpy.ops.import_scene.gltf(filepath=path)
            imported = list(context.selected_objects)
            for obj in imported:
                obj.location = (0.0, 0.0, 0.0)
            self.report({'INFO'}, f"Imported {len(imported)} objects from bundled GLB.")
        except Exception as e:
            self.report({'ERROR'}, f"Import failed: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}

# --- Add > Mesh menu hook ---

def _menu_add_reference(self, _context):
    # Shows up under Add > Mesh
    self.layout.operator(
        SKATE_OT_LoadBundledGLB.bl_idname,
        text="Skate Reference",
        icon='OUTLINER_OB_MESH'
    )

# --- Handlers ---

@persistent
def _load_on_file_open(_dummy):
    _import_reference_into_scene()

def _one_shot_timer():
    # Safety: run once after startup in case load_post didn't hit initial empty scene
    _import_reference_into_scene()
    return None

def register():
    bpy.utils.register_class(SKATE_OT_LoadBundledGLB)

    # On file load (including startup file)
    if _load_on_file_open not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_load_on_file_open)

    # One-shot timer in case Blender starts with a blank scene without triggering load_post yet
    try:
        bpy.app.timers.register(_one_shot_timer, first_interval=1.0)
    except Exception:
        pass

    # Add the operator to Add > Mesh
    bpy.types.VIEW3D_MT_mesh_add.append(_menu_add_reference)

def unregister():
    # Remove menu entry
    try:
        bpy.types.VIEW3D_MT_mesh_add.remove(_menu_add_reference)
    except Exception:
        pass

    # Remove the load handler
    try:
        if _load_on_file_open in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.remove(_load_on_file_open)
    except Exception:
        pass

    bpy.utils.unregister_class(SKATE_OT_LoadBundledGLB)
