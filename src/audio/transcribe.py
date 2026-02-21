"""
transcribe.py
Word-level timestamped transcription with sentence segmentation.
Uses Whisper with word_timestamps=True.
"""

import whisper
import re
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: List[Word] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "words": [{"text": w.text, "start": w.start, "end": w.end} for w in self.words],
        }


# Sentence-ending punctuation regex
_SENTENCE_END = re.compile(r'[.!?]+\s*$')


def _group_words_into_sentences(words: List[Word]) -> List[Segment]:
    """
    Groups word-level timestamps into sentence-level segments.
    Splits at punctuation boundaries (. ! ?) so each segment
    represents a semantically complete sentence.
    """
    if not words:
        return []

    sentences = []
    current_words = []
    current_text_parts = []

    for word in words:
        current_words.append(word)
        current_text_parts.append(word.text.strip())

        # Check if this word ends a sentence
        if _SENTENCE_END.search(word.text):
            text = " ".join(current_text_parts)
            sentences.append(Segment(
                text=text,
                start=current_words[0].start,
                end=current_words[-1].end,
                words=list(current_words),
            ))
            current_words = []
            current_text_parts = []

    # Don't lose trailing words that don't end with punctuation
    if current_words:
        text = " ".join(current_text_parts)
        sentences.append(Segment(
            text=text,
            start=current_words[0].start,
            end=current_words[-1].end,
            words=list(current_words),
        ))

    return sentences


class WhisperTranscriber:
    """
    Transcribes audio with word-level timestamps, then groups into sentences.
    Model is created externally via StageContext.load_model() to enable reuse.
    """

    def __init__(self, model: str = "small", device: str = "cuda"):
        self.device = device
        self.model_size = model
        self._model = None

    def load(self):
        """Load the Whisper model into VRAM."""
        logger.info(f"Loading Whisper '{self.model_size}' on {self.device}")
        self._model = whisper.load_model(self.model_size, device=self.device)
        return self

    def transcribe(self, audio_path: str) -> List[Segment]:
        """
        Transcribe audio with word-level timestamps and return sentence segments.
        Each Segment contains: text, start, end, words[].
        """
        assert self._model is not None, "Call .load() first or use via StageContext"

        fp16 = self.device.startswith("cuda")
        logger.info(f"Transcribing {audio_path} with word timestamps...")

        result = self._model.transcribe(
            audio_path,
            fp16=fp16,
            word_timestamps=True,
        )

        # Extract word-level timestamps from all segments
        all_words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                all_words.append(Word(
                    text=w.get("word", ""),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                ))

        # Group into sentences
        sentences = _group_words_into_sentences(all_words)
        logger.info(f"Transcribed {len(all_words)} words → {len(sentences)} sentences")

        for i, seg in enumerate(sentences):
            logger.debug(f"  Sentence {i}: [{seg.start:.2f}-{seg.end:.2f}] {seg.text[:60]}...")

        return sentences
