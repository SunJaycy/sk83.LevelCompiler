
bl_info = {
    "name": "Skate Mapping Tools (One-Drop Bundle)",
    "author": "SunJay, Dumbad, Tuukkas, Wissp",
    "version": (1, 1, 0),
    "blender": (3, 6, 0),
    "location": "3D View > Sidebar > Skate",
    "description": "Loads all Skate Map tools together: Exporters, Spline Generator, and auto-loads assets/reference.glb on startup.",
    "category": "Import-Export",
}

import importlib, sys

SUBMODULES = [
    "bulkexportglb",
    "Collision_Export_Dumbad_Tuukkas",
    "SplineGeneratorv2",
    "asset_loader",
]

_loaded = {}

def _load(name):
    full = f"{__package__}.{name}"
    if full in sys.modules:
        _loaded[name] = importlib.reload(sys.modules[full])
    else:
        _loaded[name] = importlib.import_module(full)
    return _loaded[name]

def register():
    for name in SUBMODULES:
        mod = _load(name)
        if hasattr(mod, "register"):
            mod.register()

def unregister():
    for name in reversed(SUBMODULES):
        mod = _loaded.get(name)
        if mod and hasattr(mod, "unregister"):
            try:
                mod.unregister()
            except Exception:
                pass
