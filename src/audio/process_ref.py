import os
import subprocess
import tempfile

class VoiceReferenceProcessor:
    def __init__(self, workdir: str):
        self.workdir = workdir

    def process(self, input_wav: str) -> str:
        """
        Trims silence and normalizes loudness of the speaker reference clip.
        Returns the path to the cleaned reference audio.
        """
        processed_wav = os.path.join(self.workdir, "ref_cleaned.wav")
        if os.path.exists(processed_wav):
            return processed_wav # Cache hit for this chunk/session

        # ffmpeg silenceremove and loudnorm filters
        cmd = [
            "ffmpeg", "-y", "-i", input_wav,
            "-af", "silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-50dB,loudnorm",
            processed_wav
        ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return processed_wav
