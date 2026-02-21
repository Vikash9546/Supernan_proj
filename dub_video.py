#!/usr/bin/env python3
"""
dub_video.py
3-Stage Hindi dubbing pipeline with competition-winning architecture.

Stage 1 (NLP):   Whisper transcription → Sentence segmentation → Translation
Stage 2 (Audio): Per-segment TTS → Per-segment time stretch → Timeline reconstruction
Stage 3 (Video): Streaming face crop → Wav2Lip → CodeFormer → Alpha blend → Pipe encode

Models are loaded ONCE per stage and freed before the next stage begins.
Peak VRAM never exceeds 6 GB.
"""

import argparse
import os
import tempfile
import time
import logging
import torch

# Monkey patch torch.load for PyTorch 2.6 to fix XTTS WeightsUnpickler error
_original_load = torch.load
torch.load = lambda *a, **k: _original_load(*a, **{**k, 'weights_only': False})

from tqdm import tqdm

from src.utils.gpu_manager import StageContext
from src.utils.batch import VideoBatcher
from src.utils.stream_utils import FFmpegFrameReader, FFmpegFrameWriter, probe_video
from src.audio.transcribe import WhisperTranscriber
from src.audio.translate import IndicTranslator
from src.audio.tts import XTTSGenerator
from src.audio.align import AudioAligner
from src.audio.process_ref import VoiceReferenceProcessor
from src.video.face_tracker import FaceTracker
from src.video.lip_sync import LipSyncEngine
from src.video.enhance import FaceEnhancer
from src.video.sync_eval import SyncEvaluator
from src.video.compose import VideoComposer


def setup_logger(log_level: str):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Competition-winning Hindi dubbing pipeline")
    p.add_argument("--input", required=True, help="Path to source video")
    p.add_argument("--output", required=True, help="Path for dubbed video")
    p.add_argument("--whisper-model", default="small", help="Whisper model size")
    p.add_argument("--tts-checkpoint", default="tts_models/multilingual/multi-dataset/xtts_v2")
    p.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--use-llm-rephrase", action="store_true",
                   help="Enable Llama-3 8B rephrase for length-constrained translation")
    p.add_argument("--ema-alpha", type=float, default=0.15,
                   help="EMA smoothing factor for face tracking (0=smooth, 1=responsive)")
    p.add_argument("--lip-batch-size", type=int, default=64,
                   help="Batch size for lip-sync GPU inference")
    return p.parse_args()


def extract_full_audio(video_path: str, workdir: str) -> str:
    """Extract full audio track from video as 16kHz mono WAV."""
    audio_path = os.path.join(workdir, "full_audio.wav")
    os.system(
        f"ffmpeg -y -i \"{video_path}\" -vn -acodec pcm_s16le -ar 16000 -ac 1 \"{audio_path}\""
    )
    return audio_path


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: NLP — Transcribe, Segment, Translate
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage_nlp(args, audio_path: str) -> list:
    """
    Stage 1: Transcribe with word-level timestamps, segment into sentences,
    translate with syllable constraint. All NLP models loaded once here.
    """
    logger = logging.getLogger("stage.nlp")
    logger.info("═══ STAGE 1: NLP Pipeline ═══")

    with StageContext("nlp", device=args.device) as ctx:
        # 1a. Transcribe with word-level timestamps
        t0 = time.time()
        transcriber = ctx.load_model(
            "whisper",
            lambda: WhisperTranscriber(model=args.whisper_model, device=args.device).load()
        )
        sentences = transcriber.transcribe(audio_path)
        logger.info(f"Transcription: {len(sentences)} sentences in {time.time()-t0:.1f}s")

        # 1b. Translate with syllable constraint
        t0 = time.time()
        translator = ctx.load_model(
            "translator",
            lambda: IndicTranslator(device=args.device).load()
        )

        if args.use_llm_rephrase:
            translator.load_rephrase_llm()

        hindi_segs = translator.translate(sentences)
        logger.info(f"Translation: {len(hindi_segs)} segments in {time.time()-t0:.1f}s")

    # Models freed here — ~6GB VRAM reclaimed
    return hindi_segs


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Audio — Per-Segment TTS, Stretch, Reconstruct
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage_audio(args, hindi_segs: list, total_duration: float,
                    workdir: str, ref_audio: str) -> str:
    """
    Stage 2: Synthesize each segment individually, time-stretch to match
    original timing, reconstruct full audio with silence preservation.
    """
    logger = logging.getLogger("stage.audio")
    logger.info("═══ STAGE 2: Audio Pipeline ═══")

    with StageContext("audio", device=args.device) as ctx:
        # 2a. Per-segment TTS synthesis
        t0 = time.time()
        tts = ctx.load_model(
            "tts",
            lambda: XTTSGenerator(checkpoint=args.tts_checkpoint, device=args.device).load()
        )
        segment_audios = tts.synthesize(hindi_segs, workdir, speaker_wav=ref_audio)
        logger.info(f"TTS: {len(segment_audios)} segments in {time.time()-t0:.1f}s")

    # TTS model freed — ~2.5GB VRAM reclaimed

    # 2b. Per-segment time stretch and timeline reconstruction (CPU-only)
    t0 = time.time()
    aligner = AudioAligner(total_duration=total_duration)
    aligned_audio = os.path.join(workdir, "aligned_audio.wav")
    aligner.align_segments(segment_audios, aligned_audio)
    logger.info(f"Alignment: {time.time()-t0:.1f}s")

    import shutil
    shutil.copy(aligned_audio, "/Users/gourav/Desktop/Supernan_proj/DEBUG_aligned_audio.wav")
    
    return aligned_audio


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Video — Stream, Track, Crop, Sync, Enhance, Blend, Encode
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage_video(args, aligned_audio: str, workdir: str) -> str:
    """
    Stage 3: Streaming video pipeline.
    Reads frames via FFmpeg pipe → face track → crop → Wav2Lip → CodeFormer
    → alpha blend back to 1080p → pipe to FFmpeg encoder.
    
    Memory footprint is constant regardless of video length.
    """
    logger = logging.getLogger("stage.video")
    logger.info("═══ STAGE 3: Video Pipeline ═══")

    info = probe_video(args.input)
    fps = info["fps"]
    width = info["width"]
    height = info["height"]
    output_video = os.path.join(workdir, "video_only.mp4")

    with StageContext("video", device=args.device) as ctx:
        # Load all video models ONCE
        tracker = ctx.load_model(
            "face_tracker",
            lambda: FaceTracker(ema_alpha=args.ema_alpha).load()
        )
        lip_sync = ctx.load_model(
            "lip_sync",
            lambda: LipSyncEngine(device=args.device, batch_size=args.lip_batch_size).load()
        )
        enhancer = ctx.load_model(
            "enhancer",
            lambda: FaceEnhancer(device=args.device).load()
        )
        sync_eval = ctx.load_model(
            "sync_eval",
            lambda: SyncEvaluator(device=args.device).load()
        )

        # Open streaming reader and writer
        reader = FFmpegFrameReader(args.input)
        writer = FFmpegFrameWriter(output_video, fps, width, height)

        # Accumulate crops in batches for efficient GPU inference
        batch_crops = []
        batch_matrices = []
        batch_originals = []
        BATCH_SIZE = args.lip_batch_size

        t0 = time.time()
        frame_count = 0
        lse_scores = []

        for frame_idx, frame in reader:
            # 3a. Face detection with EMA smoothing
            bbox = tracker.track(frame)

            # 3b. Extract 256×256 face crop
            crop, M = tracker.extract_crop(frame, bbox)

            batch_crops.append(crop)
            batch_matrices.append(M)
            batch_originals.append(frame)

            # 3c. Process batch when full
            if len(batch_crops) >= BATCH_SIZE:
                score = _process_and_write_batch(
                    batch_crops, batch_matrices, batch_originals,
                    lip_sync, enhancer, sync_eval, tracker, aligned_audio, fps,
                    writer
                )
                if score > 0:
                    lse_scores.append(score)
                batch_crops = []
                batch_matrices = []
                batch_originals = []

            frame_count += 1
            if frame_count % 500 == 0:
                logger.info(f"Processed {frame_count} frames...")

        # Process remaining frames
        if batch_crops:
            score = _process_and_write_batch(
                batch_crops, batch_matrices, batch_originals,
                lip_sync, enhancer, sync_eval, tracker, aligned_audio, fps,
                writer
            )
            if score > 0:
                lse_scores.append(score)

        reader.close()
        writer.close()
        
        avg_lse = sum(lse_scores) / len(lse_scores) if lse_scores else 0.0
        logger.info(f"Video pipeline: {frame_count} frames in {time.time()-t0:.1f}s")
        logger.info(f"Final Lip-Sync Confidence (LSE-C): {avg_lse:.2f}")

    return output_video


def _process_and_write_batch(crops, matrices, originals,
                              lip_sync, enhancer, sync_eval, tracker,
                              audio_path, fps, writer):
    """Process a batch of crops through lip-sync → enhance → blend → write."""
    # Lip-sync on crops
    synced_crops = lip_sync.sync_crops(crops, audio_path, fps)

    # Evaluate sync confidence before enhancement
    score = sync_eval.evaluate(synced_crops, audio_path, fps)

    # Free memory before running CodeFormer
    del crops

    # Enhance crops with CodeFormer
    enhanced_crops = enhancer.enhance_crops(synced_crops)

    # Blend each crop back into its original full-res frame and write
    for orig_frame, crop, M in zip(originals, enhanced_crops, matrices):
        final_frame = tracker.blend_crop(orig_frame, crop, M, feather=0.1)
        writer.write(final_frame)
        
    return score


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)
    logger = logging.getLogger("dub_pipeline")
    logger.info("🎬 Starting dubbing pipeline (3-stage architecture)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    total_start = time.time()

    with tempfile.TemporaryDirectory() as workdir:
        # Probe video info
        info = probe_video(args.input)
        total_duration = info["duration"]
        logger.info(f"Input: {info['width']}x{info['height']} @ {info['fps']:.1f}fps, "
                     f"{total_duration:.1f}s duration")

        # Extract and clean reference audio (once)
        full_audio = extract_full_audio(args.input, workdir)
        ref_processor = VoiceReferenceProcessor(workdir)
        ref_audio = ref_processor.process(full_audio)

        # ── Stage 1: NLP ──
        hindi_segs = run_stage_nlp(args, full_audio)

        # ── Stage 2: Audio ──
        aligned_audio = run_stage_audio(
            args, hindi_segs, total_duration, workdir, ref_audio
        )

        # ── Stage 3: Video ──
        video_only = run_stage_video(args, aligned_audio, workdir)

        # ── Final Mux ──
        logger.info("Muxing final output...")
        mux_cmd = (
            f"ffmpeg -y -i {video_only} -i {aligned_audio} "
            f"-c:v copy -c:a aac -b:a 192k -shortest {args.output}"
        )
        os.system(mux_cmd)

    total_time = time.time() - total_start
    logger.info(f"✅ Pipeline complete in {total_time:.1f}s → {args.output}")


if __name__ == "__main__":
    main()
