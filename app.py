"""
Chatterbox TTS - Enterprise Gradio WebUI v3
============================================
New Features:
- Custom preset save/delete
- Incognito mode (no history/files/logs saved)

Environment: RTX 5070 Ti + CUDA 13 + PyTorch 2.9.1
"""

import gradio as gr
import torch
import soundfile as sf
import numpy as np
import tempfile
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List
from dataclasses import dataclass
from threading import Lock

# ==================== Configuration ====================


@dataclass
class AppConfig:
    app_name: str = "Chatterbox TTS Studio"
    version: str = "3.0.0"
    server_host: str = "0.0.0.0"
    server_port: int = 7860
    max_text_length: int = 5000
    output_dir: str = "./outputs"
    history_dir: str = "./history"
    presets_file: str = "./presets.json"
    log_file: str = "./app.log"


CONFIG = AppConfig()

# ==================== Logging (Supports Incognito Mode) ====================


class ConditionalLogger:
    """Conditional logger - supports incognito mode"""

    def __init__(self):
        self._enabled = True
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'))
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

        # File logging
        self._file_handler = None
        self._enable_file_logging()

    def _enable_file_logging(self):
        if self._file_handler is None:
            self._file_handler = logging.FileHandler(
                CONFIG.log_file, encoding='utf-8')
            self._file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s'))
            self._logger.addHandler(self._file_handler)

    def _disable_file_logging(self):
        if self._file_handler:
            self._logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

    def set_incognito(self, enabled: bool):
        """Set incognito mode"""
        self._enabled = not enabled
        if enabled:
            self._disable_file_logging()
        else:
            self._enable_file_logging()

    def info(self, msg):
        if self._enabled:
            self._logger.info(msg)

    def error(self, msg, **kwargs):
        # Errors are always logged to console
        self._logger.error(msg, **kwargs)


logger = ConditionalLogger()

# ==================== Language Configuration ====================
LANGUAGES = {
    "English": {"code": "en", "flag": "🇺🇸", "multilingual": False},
    "Chinese (中文)": {"code": "zh", "flag": "🇨🇳", "multilingual": True},
    "Japanese (日本語)": {"code": "ja", "flag": "🇯🇵", "multilingual": True},
    "Korean (한국어)": {"code": "ko", "flag": "🇰🇷", "multilingual": True},
    "French (Français)": {"code": "fr", "flag": "🇫🇷", "multilingual": True},
    "German (Deutsch)": {"code": "de", "flag": "🇩🇪", "multilingual": True},
    "Spanish (Español)": {"code": "es", "flag": "🇪🇸", "multilingual": True},
    "Italian (Italiano)": {"code": "it", "flag": "🇮🇹", "multilingual": True},
    "Portuguese (Português)": {"code": "pt", "flag": "🇵🇹", "multilingual": True},
    "Russian (Русский)": {"code": "ru", "flag": "🇷🇺", "multilingual": True},
    "Arabic (العربية)": {"code": "ar", "flag": "🇸🇦", "multilingual": True},
    "Dutch (Nederlands)": {"code": "nl", "flag": "🇳🇱", "multilingual": True},
    "Polish (Polski)": {"code": "pl", "flag": "🇵🇱", "multilingual": True},
    "Hindi (हिन्दी)": {"code": "hi", "flag": "🇮🇳", "multilingual": True},
    "Turkish (Türkçe)": {"code": "tr", "flag": "🇹🇷", "multilingual": True},
    "Swedish (Svenska)": {"code": "sv", "flag": "🇸🇪", "multilingual": True},
    "Danish (Dansk)": {"code": "da", "flag": "🇩🇰", "multilingual": True},
    "Finnish (Suomi)": {"code": "fi", "flag": "🇫🇮", "multilingual": True},
    "Norwegian (Norsk)": {"code": "no", "flag": "🇳🇴", "multilingual": True},
    "Greek (Ελληνικά)": {"code": "el", "flag": "🇬🇷", "multilingual": True},
    "Hebrew (עברית)": {"code": "he", "flag": "🇮🇱", "multilingual": True},
    "Malay (Bahasa Melayu)": {"code": "ms", "flag": "🇲🇾", "multilingual": True},
    "Swahili (Kiswahili)": {"code": "sw", "flag": "🇰🇪", "multilingual": True},
}

# Default presets
DEFAULT_PRESETS = {
    "Default (Balanced)": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 1.0, "description": "Balanced settings, suitable for most scenarios", "builtin": True},
    "News Broadcast": {"exaggeration": 0.2, "cfg_weight": 0.7, "temperature": 0.8, "description": "Professional and steady broadcast style", "builtin": True},
    "Storytelling": {"exaggeration": 0.7, "cfg_weight": 0.5, "temperature": 1.1, "description": "Vivid and expressive", "builtin": True},
    "Customer Service": {"exaggeration": 0.4, "cfg_weight": 0.6, "temperature": 0.9, "description": "Friendly and professional tone", "builtin": True},
    "Audiobook": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 1.0, "description": "Comfortable reading style", "builtin": True},
    "Emotional": {"exaggeration": 0.9, "cfg_weight": 0.4, "temperature": 1.2, "description": "Strong emotional expression", "builtin": True},
}

# ==================== Preset Manager ====================


class PresetManager:
    """Preset manager"""

    def __init__(self):
        self._presets = self._load()

    def _load(self) -> dict:
        """Load presets"""
        presets = DEFAULT_PRESETS.copy()
        if os.path.exists(CONFIG.presets_file):
            try:
                with open(CONFIG.presets_file, 'r', encoding='utf-8') as f:
                    user_presets = json.load(f)
                    presets.update(user_presets)
            except:
                pass
        return presets

    def _save(self):
        """Save user presets (only non-builtin ones)"""
        user_presets = {k: v for k, v in self._presets.items()
                        if not v.get("builtin", False)}
        with open(CONFIG.presets_file, 'w', encoding='utf-8') as f:
            json.dump(user_presets, f, ensure_ascii=False, indent=2)

    def get_all(self) -> dict:
        return self._presets

    def get_names(self) -> List[str]:
        return list(self._presets.keys())

    def get(self, name: str) -> Optional[dict]:
        return self._presets.get(name)

    def add(self, name: str, exaggeration: float, cfg_weight: float, temperature: float, description: str) -> str:
        """Add new preset"""
        if not name or not name.strip():
            return "❌ Preset name cannot be empty"

        name = name.strip()

        if name in self._presets and self._presets[name].get("builtin", False):
            return f"❌ Cannot overwrite builtin preset '{name}'"

        self._presets[name] = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "description": description,
            "builtin": False
        }
        self._save()
        return f"✅ Preset '{name}' saved"

    def delete(self, name: str) -> str:
        """Delete preset"""
        if name not in self._presets:
            return f"❌ Preset '{name}' does not exist"

        if self._presets[name].get("builtin", False):
            return f"❌ Cannot delete builtin preset '{name}'"

        del self._presets[name]
        self._save()
        return f"✅ Preset '{name}' deleted"


preset_manager = PresetManager()

# ==================== Model Manager ====================


class ModelManager:
    """Dual model manager"""

    def __init__(self):
        self._english_model = None
        self._multilingual_model = None
        self._lock = Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def device(self):
        return self._device

    def load_english_model(self):
        with self._lock:
            if self._english_model is None:
                logger.info("Loading English model...")
                from chatterbox.tts import ChatterboxTTS
                self._english_model = ChatterboxTTS.from_pretrained(
                    device=self._device)
                logger.info("English model loaded")
            return self._english_model

    def load_multilingual_model(self):
        with self._lock:
            if self._multilingual_model is None:
                logger.info("Loading multilingual model...")
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                self._multilingual_model = ChatterboxMultilingualTTS.from_pretrained(
                    device=self._device)
                logger.info("Multilingual model loaded")
            return self._multilingual_model

    def generate(
        self,
        text: str,
        language_code: str = "en",
        use_multilingual: bool = False,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> Tuple[np.ndarray, int]:

        kwargs = {"exaggeration": exaggeration, "cfg_weight": cfg_weight}

        if audio_prompt_path and os.path.exists(audio_prompt_path):
            kwargs["audio_prompt_path"] = audio_prompt_path

        if use_multilingual and language_code != "en":
            model = self.load_multilingual_model()
            kwargs["language_id"] = language_code
        else:
            model = self.load_english_model()

        wav = model.generate(text, **kwargs)
        return wav.squeeze().cpu().numpy(), model.sr

    def unload_all(self):
        with self._lock:
            if self._english_model:
                del self._english_model
                self._english_model = None
            if self._multilingual_model:
                del self._multilingual_model
                self._multilingual_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_status(self) -> str:
        en = "✅" if self._english_model else "❌"
        ml = "✅" if self._multilingual_model else "❌"
        return f"English: {en} | Multilingual: {ml}"


model_manager = ModelManager()

# ==================== Incognito Mode State ====================
incognito_mode = False


def set_incognito_mode(enabled: bool) -> str:
    global incognito_mode
    incognito_mode = enabled
    logger.set_incognito(enabled)

    if enabled:
        return "🕵️ Incognito mode enabled\n• Audio files not saved\n• History not recorded\n• Log file not written"
    else:
        return "👁️ Incognito mode disabled\n• Files and records saved normally"

# ==================== Utility Functions ====================


def ensure_dirs():
    if not incognito_mode:
        Path(CONFIG.output_dir).mkdir(parents=True, exist_ok=True)
        Path(CONFIG.history_dir).mkdir(parents=True, exist_ok=True)


def get_system_info() -> str:
    info = [f"PyTorch: {torch.__version__}"]
    if torch.cuda.is_available():
        info.append(f"CUDA: {torch.version.cuda}")
        info.append(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info.append(f"VRAM: {mem:.1f}GB")
    else:
        info.append("⚠️ CPU Mode")
    return " | ".join(info)


def get_gpu_stats() -> str:
    if not torch.cuda.is_available():
        return "GPU not available"
    used = torch.cuda.memory_allocated(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pct = (used / total) * 100
    bar_len = 20
    filled = int(bar_len * used / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    return f"[{bar}] {used:.1f}G / {total:.1f}G ({pct:.0f}%)"


def save_audio(audio_np: np.ndarray, sample_rate: int, fmt: str = "wav") -> str:
    """Save audio - returns temp file in incognito mode"""
    if incognito_mode:
        # Incognito mode: use temp file
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
            sf.write(f.name, audio_np, sample_rate)
            return f.name
    else:
        # Normal mode: save to outputs
        ensure_dirs()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{ts}.{fmt}"
        filepath = os.path.join(CONFIG.output_dir, filename)
        sf.write(filepath, audio_np, sample_rate)
        logger.info(f"Audio saved: {filepath}")
        return filepath


def save_to_history(text: str, language: str, audio_path: str, params: dict):
    """Save history - skipped in incognito mode"""
    if incognito_mode:
        return

    ensure_dirs()
    history_file = os.path.join(CONFIG.history_dir, "history.json")
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            pass

    record = {
        "timestamp": datetime.now().isoformat(),
        "text": text[:100] + "..." if len(text) > 100 else text,
        "language": language,
        "audio_path": audio_path,
        "params": params
    }
    history.insert(0, record)
    history = history[:100]

    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def get_history_display() -> str:
    history_file = os.path.join(CONFIG.history_dir, "history.json")
    if not os.path.exists(history_file):
        return "No history records"
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except:
        return "No history records"

    lines = []
    for i, r in enumerate(history[:20], 1):
        ts = r.get("timestamp", "")[:16].replace("T", " ")
        txt = r.get("text", "")[:30]
        lang = r.get("language", "")[:8]
        lines.append(f"{i}. [{ts}] {lang} | {txt}")
    return "\n".join(lines) if lines else "No history records"


def clear_history() -> str:
    history_file = os.path.join(CONFIG.history_dir, "history.json")
    if os.path.exists(history_file):
        os.remove(history_file)
    return "✅ History cleared"

# ==================== Core Generation Function ====================


def generate_speech(
    text: str,
    language: str,
    reference_audio: Optional[str],
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    seed: int,
    use_random_seed: bool,
    output_format: str,
    progress=gr.Progress()
) -> Tuple[Optional[str], str, str]:

    start_time = time.time()

    if not text or not text.strip():
        return None, "❌ Please enter text", get_gpu_stats()

    text = text.strip()
    if len(text) > CONFIG.max_text_length:
        return None, f"❌ Text too long (max {CONFIG.max_text_length} chars)", get_gpu_stats()

    try:
        progress(0.1, desc="Preparing...")

        lang_config = LANGUAGES.get(
            language, {"code": "en", "multilingual": False})
        lang_code = lang_config["code"]
        use_multilingual = lang_config["multilingual"]
        flag = lang_config.get("flag", "")

        if not use_random_seed and seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        progress(0.3, desc="Loading model...")
        progress(0.5, desc="Generating...")

        audio_np, sample_rate = model_manager.generate(
            text=text,
            language_code=lang_code,
            use_multilingual=use_multilingual,
            audio_prompt_path=reference_audio,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        progress(0.9, desc="Saving...")

        audio_path = save_audio(audio_np, sample_rate, output_format)

        duration = len(audio_np) / sample_rate
        elapsed = time.time() - start_time
        rtf = elapsed / duration

        # Save history (skipped in incognito mode)
        save_to_history(text, language, audio_path, {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "seed": seed if not use_random_seed else "random"
        })

        progress(1.0, desc="Done!")

        clone_info = " | 🎭 Cloned" if reference_audio else ""
        model_type = "Multilingual" if use_multilingual else "English"
        incognito_info = " | 🕵️ Incognito" if incognito_mode else ""

        status = (
            f"✅ Success!\n"
            f"⏱️ Duration: {duration:.2f}s | Time: {elapsed:.2f}s | RTF: {rtf:.2f}x\n"
            f"🌍 {flag} {language} | Model: {model_type}{clone_info}{incognito_info}"
        )

        return audio_path, status, get_gpu_stats()

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return None, f"❌ Error: {str(e)}", get_gpu_stats()


def apply_preset(preset_name: str) -> Tuple[float, float, float, str]:
    p = preset_manager.get(preset_name)
    if p:
        desc = p.get("description", "")
        builtin = "📌 Builtin" if p.get("builtin") else "👤 Custom"
        return p.get("exaggeration", 0.5), p.get("cfg_weight", 0.5), p.get("temperature", 1.0), f"{builtin} | {desc}"
    return 0.5, 0.5, 1.0, "⚠️ Preset not found"


def save_new_preset(name: str, exaggeration: float, cfg_weight: float, temperature: float, description: str):
    """Save new preset"""
    result = preset_manager.add(
        name, exaggeration, cfg_weight, temperature, description)
    new_choices = preset_manager.get_names()
    return result, gr.update(choices=new_choices)


def delete_preset(name: str):
    """Delete preset"""
    result = preset_manager.delete(name)
    new_choices = preset_manager.get_names()
    return result, gr.update(choices=new_choices)


def unload_models() -> str:
    model_manager.unload_all()
    return "✅ Models unloaded"

# ==================== Gradio Interface ====================


def create_app():
    ensure_dirs()

    css = """
    .main-title {text-align:center; font-size:2.2em; font-weight:bold; 
                 background:linear-gradient(90deg,#667eea,#764ba2);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
    .subtitle {text-align:center; color:#666; margin-bottom:1em;}
    .incognito-on {background-color: #1a1a2e !important; color: #00ff00 !important;}
    footer {display:none !important;}
    """

    with gr.Blocks(title=CONFIG.app_name, theme=gr.themes.Soft(), css=css) as app:

        # Title
        gr.HTML(f"""
        <div class="main-title">🎙️ {CONFIG.app_name}</div>
        <div class="subtitle">Multilingual TTS · Voice Cloning · Emotion Control | v{CONFIG.version}</div>
        """)

        # System status bar
        with gr.Row():
            gr.Markdown(f"**{get_system_info()}**")
            gpu_status = gr.Textbox(value=get_gpu_stats(
            ), label="GPU", interactive=False, scale=1)
            refresh_btn = gr.Button("🔄", scale=0, min_width=50)

            # Incognito mode toggle
            incognito_toggle = gr.Checkbox(
                label="🕵️ Incognito Mode", value=False, scale=0)

        incognito_status = gr.Textbox(
            value="", label="", interactive=False, visible=False)

        gr.Markdown("---")

        with gr.Tabs():
            # ===== Speech Generation =====
            with gr.TabItem("🎵 Speech Generation"):
                with gr.Row():
                    # Left column
                    with gr.Column(scale=1):
                        text_input = gr.Textbox(
                            label="📝 Input Text", placeholder="Enter text to synthesize...", lines=8)

                        with gr.Row():
                            language = gr.Dropdown(choices=list(
                                LANGUAGES.keys()), value="English", label="🌍 Language", scale=2)
                            output_format = gr.Radio(
                                choices=["wav", "mp3"], value="wav", label="Format", scale=1)

                        gr.Markdown("#### 🎭 Voice Cloning (Optional)")
                        reference_audio = gr.Audio(
                            label="Reference Audio (5-15 sec)", type="filepath", sources=["upload", "microphone"])

                    # Right column
                    with gr.Column(scale=1):
                        gr.Markdown("#### ⚙️ Parameter Settings")

                        preset_select = gr.Dropdown(
                            choices=preset_manager.get_names(), value="Default (Balanced)", label="📋 Preset")
                        preset_info = gr.Textbox(
                            label="", interactive=False, lines=1)

                        exaggeration = gr.Slider(
                            0.0, 1.0, 0.5, step=0.05, label="🎭 Emotion Exaggeration", info="0=Flat, 1=Exaggerated")
                        cfg_weight = gr.Slider(
                            0.0, 1.0, 0.5, step=0.05, label="🎯 CFG Weight", info="Controls adherence to reference audio")
                        temperature = gr.Slider(
                            0.1, 2.0, 1.0, step=0.1, label="🌡️ Temperature")

                        with gr.Row():
                            seed = gr.Number(
                                value=42, label="🎲 Seed", precision=0, scale=2)
                            use_random_seed = gr.Checkbox(
                                value=True, label="Random", scale=1)

                generate_btn = gr.Button(
                    "🚀 Generate Speech", variant="primary", size="lg")

                with gr.Row():
                    output_audio = gr.Audio(
                        label="🔊 Result", type="filepath", scale=2)
                    status_output = gr.Textbox(
                        label="📊 Status", lines=4, interactive=False, scale=1)

                # Examples
                gr.Markdown("#### 📚 Examples")
                gr.Examples(
                    examples=[
                        ["Hello! Welcome to Chatterbox TTS.", "English"],
                        ["你好！欢迎使用语音合成系统。", "Chinese (中文)"],
                        ["こんにちは！音声合成へようこそ。", "Japanese (日本語)"],
                        ["Bonjour! Bienvenue!", "French (Français)"],
                    ],
                    inputs=[text_input, language]
                )

            # ===== Preset Management =====
            with gr.TabItem("📋 Preset Management"):
                gr.Markdown("### Custom Presets")
                gr.Markdown(
                    "Create your own parameter presets for quick style switching.")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### ➕ Create New Preset")
                        new_preset_name = gr.Textbox(
                            label="Preset Name", placeholder="e.g., My Style")
                        new_preset_desc = gr.Textbox(
                            label="Description", placeholder="e.g., Suitable for reading poetry")

                        with gr.Row():
                            new_exaggeration = gr.Slider(
                                0.0, 1.0, 0.5, step=0.05, label="Emotion Exaggeration")
                            new_cfg = gr.Slider(
                                0.0, 1.0, 0.5, step=0.05, label="CFG Weight")
                            new_temp = gr.Slider(
                                0.1, 2.0, 1.0, step=0.1, label="Temperature")

                        save_preset_btn = gr.Button(
                            "💾 Save Preset", variant="primary")
                        save_preset_result = gr.Textbox(
                            label="Result", interactive=False)

                    with gr.Column():
                        gr.Markdown("#### 🗑️ Delete Preset")
                        delete_preset_select = gr.Dropdown(
                            choices=preset_manager.get_names(), label="Select preset to delete")
                        delete_preset_btn = gr.Button(
                            "🗑️ Delete", variant="stop")
                        delete_preset_result = gr.Textbox(
                            label="Result", interactive=False)

                        gr.Markdown("---")
                        gr.Markdown("#### 📜 Current Preset List")
                        preset_list = gr.Textbox(
                            value="\n".join([f"{'📌' if p.get('builtin') else '👤'} {k}: {p.get('description', '')}"
                                             for k, p in preset_manager.get_all().items()]),
                            label="",
                            lines=10,
                            interactive=False
                        )

            # ===== History =====
            with gr.TabItem("📜 History"):
                history_display = gr.Textbox(
                    value=get_history_display(), label="Recent Records", lines=15, interactive=False)
                with gr.Row():
                    refresh_history_btn = gr.Button("🔄 Refresh")
                    clear_history_btn = gr.Button("🗑️ Clear", variant="stop")
                history_status = gr.Textbox(label="Status", interactive=False)

            # ===== System Settings =====
            with gr.TabItem("⚙️ System"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🕵️ Incognito Mode Info")
                        gr.Markdown("""
                        When incognito mode is enabled:
                        - ❌ Generated audio not saved to outputs directory
                        - ❌ History not recorded
                        - ❌ Log file not written
                        - ✅ Audio can still be played and downloaded (temp file)
                        
                        Suitable for processing sensitive content.
                        """)

                    with gr.Column():
                        gr.Markdown("### Model Management")
                        model_status = gr.Textbox(
                            value=model_manager.get_status(), label="Model Status", interactive=False)
                        unload_btn = gr.Button("🔓 Unload Models (Free VRAM)")
                        unload_result = gr.Textbox(
                            label="Result", interactive=False)

                gr.Markdown("---")
                gr.Markdown(f"""
                ### About
                **{CONFIG.app_name}** v{CONFIG.version}
                
                Based on [Chatterbox](https://github.com/resemble-ai/chatterbox) | MIT License
                """)

        # ===== Event Bindings =====
        refresh_btn.click(fn=lambda: get_gpu_stats(), outputs=[gpu_status])

        # Incognito mode
        incognito_toggle.change(fn=set_incognito_mode, inputs=[
                                incognito_toggle], outputs=[incognito_status])

        # Preset
        preset_select.change(fn=apply_preset, inputs=[preset_select], outputs=[
                             exaggeration, cfg_weight, temperature, preset_info])

        # Generate
        generate_btn.click(
            fn=generate_speech,
            inputs=[text_input, language, reference_audio, exaggeration,
                    cfg_weight, temperature, seed, use_random_seed, output_format],
            outputs=[output_audio, status_output, gpu_status]
        )

        # Save preset
        save_preset_btn.click(
            fn=save_new_preset,
            inputs=[new_preset_name, new_exaggeration,
                    new_cfg, new_temp, new_preset_desc],
            outputs=[save_preset_result, preset_select]
        )

        # Delete preset
        delete_preset_btn.click(
            fn=delete_preset,
            inputs=[delete_preset_select],
            outputs=[delete_preset_result, preset_select]
        )

        # History
        refresh_history_btn.click(
            fn=get_history_display, outputs=[history_display])
        clear_history_btn.click(fn=clear_history, outputs=[history_status])

        # Unload models
        unload_btn.click(fn=unload_models, outputs=[unload_result])

    return app


# ==================== Main Program ====================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"🎙️  {CONFIG.app_name} v{CONFIG.version}")
    print("=" * 60)
    print(get_system_info())
    print("=" * 60)
    print("\n🚀 Starting...\n")

    app = create_app()
    app.launch(
        server_name=CONFIG.server_host,
        server_port=CONFIG.server_port,
        share=False,
        inbrowser=True,
        show_error=True
    )
