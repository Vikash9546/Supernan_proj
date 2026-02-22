# Supernan AI Intern Submission
## High-Fidelity Video Dubbing Pipeline (English to Kannada/Hindi)

This modular Python pipeline ingests a source video, extracts and transcribes the audio, translates the transcript to Hindi while enforcing strict length constraints, synthesizes new voice-cloned audio matching the original speaker, and lip-syncs the final video with high-fidelity face restoration. 

### Output Files
* **[Final Submission Clip]**: A processed 15-second snippet of the successfully dubbed video, verifying the audio-visual sync and high-fidelity face tracking.
* **[Full Processed Video]**: The full 5-minute rendered 1080p video is present locally as `dubbed_kannada_output.mp4` (excluded from git due to size).

---

## 🏗️ Setup & Dependencies

### Prerequisites
* Python 3.9+ 
* FFmpeg (must be installed globally: `brew install ffmpeg` or `sudo apt install ffmpeg`)

### Installation 
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vikash9546/Supernan_proj.git
   cd Supernan_proj
   ```
2. **Create a virtual environment & install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   *Note: For macOS Apple Silicon (M1/M2/M3), PyTorch might need MPS fallback configurations, which are handled dynamically in `src/utils/gpu_manager.py`.*

### Running the Pipeline
```bash
python dub_video.py --video "path/to/source_video.mp4" --target_lang "hin_Deva"
```

---

## 🚀 Resourceful Architecture 

To achieve high-quality output on a budget of **₹0** without relying on expensive APIs like ElevenLabs or OpenAI, the entire pipeline is architected around powerful open-source models optimized for consumer hardware or free-tier T4/A10G cloud notebooks (Colab/Kaggle).

1. **Extraction & Transcription:** **Whisper (base/small)** extracts the audio. A custom sliding-window chunker handles edge cases in non-punctuated inputs (like the Kannada source video) to prevent 90-second run-on sentences, strictly managing audio boundaries.
2. **Translation:** We use **NLLB (No Language Left Behind)**. The translation is forced to `hin_Deva` tokens. Crucially, a *length-constrained algorithm* enforces syllable limits so the translated sentence doesn't dramatically exceed the original speaker's timeframe, bypassing the need for a costly LLM chain.
3. **Voice Synthesis (TTS):** **Coqui XTTS v2** provides extremely accurate voice cloning instantly. To bypass XTTS's strict 250-character limit and avoid dropping silent frames, the architecture chunks the text programmatically at exactly ~200 characters and smoothly stitches wave files in memory. Target audio duration matching is handled via *native TTS speed control* rather than destructive time-stretching (e.g. Librosa) to maintain high acoustic fidelity.
4. **Lip-Sync & Face Restoration:** **Wav2Lip** handles lip synchronization. To massively speed up compute on a free GPU layout:
    * **Optical Flow:** Dlib/MTCNN is heavy. Face detection is only executed on *keyframes*. Intermediate frames use Optical Flow tracking (Lucas-Kanade), speeding up facial tracking by >90%.
    * **CodeFormer:** High-def face restoration is applied directly to the 256x256 bounding box crops, rather than the raw 1080p frame, saving staggering amounts of VRAM and preventing Out-Of-Memory crashes on 16GB limit machines like free Colab instances.
    * **FFmpeg Piping:** Instead of writing thousands of frames to disk causing I/O bottlenecks, video streams route natively through `ffmpeg` pipes.

---

## 💰 Estimated Cost at Scale

If moved from a free Colab tier to a paid serverless GPU instance (e.g., RunPod/Lambda Labs with an NVIDIA A10G/RTX 4090 ($0.75 - $0.80/hour)):

* **Processing Time:** Thanks to optical flow caching and bounding-box level inference, the pipeline runs at roughly **1.5x - 2.0x real-time factor** on a modern GPU.
* **Cost Calculation:** 1 minute of video takes ~2 minutes to process. 
* **Cost per Minute of Video:** **~$0.026** per minute of video. ($1.56 per hour of video content).
* This is vastly cheaper than an ElevenLabs/HeyGen pipeline, which can easily scale into dollars per minute for TTS alone.

---

## 📈 The "Scale" Question (500 Hours Overnight)

To process 500 hours of video overnight (approx. 10 hours), we need to process 30,000 minutes of video in 600 minutes. Running sequentially at a 2.0x real-time factor requires **1,000 GPU hours**. 

**Scaling Architecture:**
1. **Decoupled Microservices:** Split the pipeline. CPU nodes are extremely cheap and can handle FFmpeg extraction, Whisper transcription, and NLLB translation. Only queue the TTS and Wav2Lip/CodeFormer stages to the expensive GPU workers.
2. **Message Queue:** Use **RabbitMQ** or **Celery** to distribute chunks of video (e.g., 5-minute segments) across a worker cluster.
3. **Dynamic Orchestration:** Use **Kubernetes (KEDA)** or **Ray** configured on AWS EKS or GCP GKE with spot instances. The cluster auto-scales 100+ spot-GPU instances (e.g., L4, T4, A10g) to concurrently churn through the video chunks.
4. **Object Storage Native:** The pipeline streams chunked frames and audio directly to/from an S3 bucket or high-IOPS shared file system (EFS/FSx) rather than saving full outputs locally to container volumes.

---

## 🛑 Known Limitations
* **Audio Mixing:** The pipeline currently overrides the entire original audio track with the dub. It does not intelligently isolate and preserve background music (BGM) or sound effects (SFX) which requires a vocal-isolation model (like UVR/Demucs).
* **Multi-Speaker Diarization:** Whisper currently lacks explicit diarization in this pipeline setting; if two subjects are speaking, XTTS clones the dominant voice condition. 

---

## 🛠️ What I'd Improve with More Time
* **UVX/Demucs Integration:** I'd insert an open-source vocal isolation step prior to translation to extract the background noise and cleanly overlay the new dubbed audio on top.
* **SyncNet Scoring Validation:** Build an autonomous loop that mathematically scores the lip-sync quality with SyncNet/LSE-C and automatically re-generates lower-scored frames.
* **Speaker Diarization:** I'd integrate Pyannote.audio to identify multiple speakers in a video and generate separate conditional voice embeddings for each cluster, dubbing multi-actor sequences flawlessly.
