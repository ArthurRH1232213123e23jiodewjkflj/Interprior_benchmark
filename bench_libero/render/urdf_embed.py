"""Put a URDF's meshes into the page, without the ones nothing draws.

WHY NOT THE UPSTREAM FUNCTION DIRECTLY. `pose_viewer._embed_urdf_mesh_data`
base64s every `<mesh>` it finds, and a URDF names each mesh TWICE: once under
`<visual>` and once under `<collision>`, usually the same file. Measured on a
7-object LIBERO page: 79.7 MB total, of which 29.4 MB was collision -- and the
viewer never parsed a byte of it. urdf-loader's `parseCollision` defaults to
false and the page never sets it, so that payload was loaded by the browser,
held in memory, and used for nothing.

THIS DOES NOT TOUCH PHYSICS. The URDF on disk is unchanged; this only rewrites
the copy handed to the HTML. Isaac reads the same file down a separate path
(`scene_spawn.spawn` -> `_convert_urdf_to_usd` -> `_bake_usd`), where
`<collision>` is exactly what the collider is built from. Two readers, one file.

Collision elements are REMOVED rather than left with bare filenames. A bare
relative filename in a standalone page is what made the objects invisible in the
first place; leaving them behind, harmless only for as long as nobody turns
`parseCollision` on, would re-arm that trap. `include_collision=True` keeps and
embeds them, for when someone does want to see the colliders.
"""

from __future__ import annotations

import base64
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

# Mirrors the upstream map. `.obj` really is text/plain: the page's
# `detectMeshFormat` keys off exactly that to pick OBJLoader for a data URI.
MIME_BY_SUFFIX = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".obj": "text/plain",
    ".stl": "model/stl",
    ".dae": "model/vnd.collada+xml",
}


# Where converted meshes are kept. Outside the repo on purpose: it is a cache,
# not an asset, and it is keyed by the source file's content hash so a rebuilt
# mesh never returns a stale conversion.
GLB_CACHE = Path.home() / ".cache" / "bench_libero" / "glb"

# Formats worth converting. `.obj` is the expensive one: it is ASCII, so every
# coordinate is a decimal string. The LIBERO meshes are generated from mesh.npz
# and carry ONLY `v` and `f` -- no uv, no normals, no mtl -- so the conversion
# is lossless in the strict sense: there is nothing else in the file. Colour
# comes from the URDF `<material>` tag, which this never touches.
# Measured: basket.obj 66,390 faces, 7.39 MB -> 1.20 MB as glb, 6.2x, same
# geometry.
CONVERTIBLE = {".obj", ".stl", ".dae", ".ply"}


# THE EXPORTED MESH MUST CARRY A MATERIAL, OR IT RENDERS BLACK.
# These URDFs declare no <material> at all. Under the old .obj path that was
# harmless: three.js OBJLoader invents a plain white material for an .obj with no
# mtl. glTF does not work that way -- a primitive with no material gets the glTF
# DEFAULT material, whose metallicFactor is 1.0, and a fully metallic surface
# with no environment map to reflect is black. So the first glb conversion turned
# every object black while the geometry was perfectly correct.
#
# Metal-free, mid grey, matching what OBJLoader used to produce. Normals are
# still omitted: the source meshes have none either (`v` and `f` only), glTF
# requires flat normals be computed when NORMAL is absent, and that is exactly
# the flat look the .obj path had.
GLB_BASE_COLOR = (0.80, 0.80, 0.80, 1.0)

# Folded into the cache key. Bump it whenever the conversion's OUTPUT changes for
# unchanged input -- the key is otherwise the source file's hash, which would
# happily serve a pre-material black glb forever.
GLB_RECIPE = "v2-nonmetal"


def _as_glb(mesh_path: Path) -> tuple[bytes, str] | None:
    """Binary-glTF bytes for a mesh file, cached on disk. None if unavailable.

    Returns None rather than raising: a viewer page that loses a conversion
    should fall back to the original bytes, not fail the run that produced it.
    """
    try:
        import trimesh
        from trimesh.visual.material import PBRMaterial
    except Exception:
        return None

    digest = hashlib.sha1(
        mesh_path.read_bytes() + GLB_RECIPE.encode()).hexdigest()[:16]
    cached = GLB_CACHE / f"{digest}.glb"
    if cached.is_file():
        return cached.read_bytes(), "model/gltf-binary"
    try:
        mesh = trimesh.load(str(mesh_path), force="mesh")
        mesh.visual = trimesh.visual.TextureVisuals(
            material=PBRMaterial(
                baseColorFactor=GLB_BASE_COLOR,
                metallicFactor=0.0,
                roughnessFactor=0.75,
            )
        )
        blob = trimesh.exchange.gltf.export_glb(trimesh.Scene(mesh))
    except Exception:
        return None
    if not blob or len(blob) >= mesh_path.stat().st_size:
        return None                      # no win; keep the original
    GLB_CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(blob)
    return blob, "model/gltf-binary"


def embed_meshes(
    urdf_text: str,
    *,
    source_urdf_path: Path,
    include_collision: bool = False,
    binary_meshes: bool = True,
) -> str:
    """Inline mesh files as data URIs.

    Drops `<collision>` unless asked, and converts ASCII meshes to binary glTF
    unless asked not to. Both only affect the page.
    """

    try:
        root = ET.fromstring(urdf_text)
    except ET.ParseError:
        return urdf_text

    if not include_collision:
        # ElementTree has no parent pointer, so strip from each link.
        for parent in root.iter():
            for child in list(parent):
                if child.tag == "collision":
                    parent.remove(child)

    cache: dict[Path, str] = {}
    for mesh_elem in root.findall(".//mesh"):
        filename = mesh_elem.get("filename")
        if not filename or filename.startswith(("data:", "http://", "https://")):
            continue
        mesh_path = (source_urdf_path.parent / filename).resolve()
        if not mesh_path.exists():
            continue
        if mesh_path not in cache:
            payload = mime = None
            if binary_meshes and mesh_path.suffix.lower() in CONVERTIBLE:
                converted = _as_glb(mesh_path)
                if converted is not None:
                    payload, mime = converted
            if payload is None:
                payload = mesh_path.read_bytes()
                mime = MIME_BY_SUFFIX.get(
                    mesh_path.suffix.lower(), "application/octet-stream")
            encoded = base64.b64encode(payload).decode("ascii")
            cache[mesh_path] = f"data:{mime};base64,{encoded}"
        mesh_elem.set("filename", cache[mesh_path])
    return ET.tostring(root, encoding="unicode")
