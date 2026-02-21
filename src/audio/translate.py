"""
translate.py
Syllable-constrained translation with LLM rephrase fallback.
Guarantees no incomplete sentences — never truncates.
"""

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Maximum allowed syllable ratio (Hindi / English)
MAX_SYLLABLE_RATIO = 1.3


def count_syllables_en(text: str) -> int:
    """Approximate English syllable count using vowel-cluster heuristic."""
    text = text.lower().strip()
    if not text:
        return 0
    # Count vowel clusters
    count = len(re.findall(r'[aeiouy]+', text))
    # Minimum 1 syllable per word
    words = text.split()
    return max(count, len(words))


def count_syllables_hi(text: str) -> int:
    """
    Approximate Hindi/Devanagari syllable count.
    Count matras (vowel signs) + independent vowels + consonants without matras.
    """
    text = text.strip()
    if not text:
        return 0

    # Devanagari vowel signs (matras)
    matras = re.findall(r'[\u093e-\u094c\u0962\u0963]', text)
    # Independent vowels
    vowels = re.findall(r'[\u0904-\u0914]', text)
    # Consonants (may or may not have matras)
    consonants = re.findall(r'[\u0915-\u0939\u0958-\u095f]', text)
    # Consonants without matras form inherent 'a' vowel syllable
    consonants_without_matra = len(consonants) - len(matras)

    count = len(matras) + len(vowels) + max(0, consonants_without_matra)

    # Fallback for Romanized Hindi or mixed scripts
    if count == 0:
        count = count_syllables_en(text)

    return max(count, 1)


class IndicTranslator:
    """
    Translates English segments to Hindi with syllable-counting validation.
    Uses IndicTrans2 as primary, with optional LLM rephrase for length control.
    """

    def __init__(self, model_name: str = "ai4bharat/indictrans2-en-indic-1B",
                 device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._llm = None  # Optional rephrase model

    def load(self):
        """Load translation model into VRAM."""
        logger.info(f"Loading IndicTrans2 on {self.device} (FP16)")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.device.startswith("cuda"):
            self._model = self._model.half().to(self.device)
        return self

    def load_rephrase_llm(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        """
        Optionally load a quantized LLM for syllable-constrained rephrasing.
        Uses 4-bit quantization to stay within VRAM budget.
        """
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            from transformers import AutoModelForCausalLM
            logger.info(f"Loading rephrase LLM: {model_name} (4-bit)")
            self._llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
            )
            logger.info("Rephrase LLM loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load rephrase LLM: {e}, will skip rephrasing")
            self._llm = None

    def _translate_text(self, text: str) -> str:
        """Translate a single text string to Hindi."""
        inputs = self._tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=int(len(text.split()) * 2.0 + 20),
                num_beams=4,
            )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _rephrase_shorter(self, hindi_text: str, target_syllables: int) -> str:
        """
        Use LLM to rephrase Hindi text to fit within target syllable count.
        Falls back to the original text if LLM is not available.
        """
        if self._llm is None:
            logger.debug("No rephrase LLM loaded, returning original translation")
            return hindi_text

        prompt = (
            f"Rephrase the following Hindi sentence to be shorter and more concise, "
            f"using approximately {target_syllables} syllables. "
            f"Keep the meaning intact. Output ONLY the rephrased Hindi text.\n\n"
            f"Original: {hindi_text}\n"
            f"Rephrased:"
        )

        inputs = self._llm_tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._llm.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True,
            )

        result = self._llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the rephrased part after "Rephrased:"
        if "Rephrased:" in result:
            result = result.split("Rephrased:")[-1].strip()

        # Validate that result is not empty
        if not result.strip():
            return hindi_text

        return result.strip()

    def translate(self, segments) -> List[dict]:
        """
        Translate segments with syllable constraint enforcement.
        Never truncates — rephrases if too long.
        
        Args:
            segments: List of Segment objects with .text, .start, .end
        Returns:
            List of dicts with 'text', 'start', 'end' keys.
        """
        assert self._model is not None, "Call .load() first"

        hindi_segs = []
        for seg in segments:
            text = seg.text.strip() if isinstance(seg, dict) is False else seg["text"].strip()
            start = seg.start if isinstance(seg, dict) is False else seg["start"]
            end = seg.end if isinstance(seg, dict) is False else seg["end"]

            if not text:
                continue

            # Translate
            hindi_text = self._translate_text(text)

            # Syllable validation
            en_syllables = count_syllables_en(text)
            hi_syllables = count_syllables_hi(hindi_text)
            target_syllables = int(en_syllables * MAX_SYLLABLE_RATIO)

            if hi_syllables > target_syllables:
                logger.info(
                    f"Translation too long ({hi_syllables} vs {target_syllables} max syllables), "
                    f"attempting rephrase..."
                )
                hindi_text = self._rephrase_shorter(hindi_text, target_syllables)
                hi_syllables = count_syllables_hi(hindi_text)
                logger.info(f"After rephrase: {hi_syllables} syllables")

            hindi_segs.append({
                "start": start,
                "end": end,
                "text": hindi_text,
                "original_text": text,
                "syllable_ratio": hi_syllables / max(en_syllables, 1),
            })

            logger.debug(
                f"[{start:.1f}-{end:.1f}] EN({en_syllables}syl): {text[:40]}... → "
                f"HI({hi_syllables}syl): {hindi_text[:40]}..."
            )

        return hindi_segs
