"""
Chatterbox TTS - Enterprise Gradio WebUI v3
============================================
新增功能:
- 自定义预设保存/删除
- 隐身模式（不保存记录/文件/日志）

环境: RTX 5070 Ti + CUDA 13 + PyTorch 2.9.1
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

# ==================== 配置 ====================


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

# ==================== 日志（支持隐身模式）====================


class ConditionalLogger:
    """条件日志器 - 支持隐身模式"""

    def __init__(self):
        self._enabled = True
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'))
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

        # 文件日志
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
        """设置隐身模式"""
        self._enabled = not enabled
        if enabled:
            self._disable_file_logging()
        else:
            self._enable_file_logging()

    def info(self, msg):
        if self._enabled:
            self._logger.info(msg)

    def error(self, msg, **kwargs):
        # 错误始终记录到控制台
        self._logger.error(msg, **kwargs)


logger = ConditionalLogger()

# ==================== 语言配置 ====================
LANGUAGES = {
    "English (英语)": {"code": "en", "flag": "🇺🇸", "multilingual": False},
    "中文 (Chinese)": {"code": "zh", "flag": "🇨🇳", "multilingual": True},
    "日本語 (Japanese)": {"code": "ja", "flag": "🇯🇵", "multilingual": True},
    "한국어 (Korean)": {"code": "ko", "flag": "🇰🇷", "multilingual": True},
    "Français (French)": {"code": "fr", "flag": "🇫🇷", "multilingual": True},
    "Deutsch (German)": {"code": "de", "flag": "🇩🇪", "multilingual": True},
    "Español (Spanish)": {"code": "es", "flag": "🇪🇸", "multilingual": True},
    "Italiano (Italian)": {"code": "it", "flag": "🇮🇹", "multilingual": True},
    "Português (Portuguese)": {"code": "pt", "flag": "🇵🇹", "multilingual": True},
    "Русский (Russian)": {"code": "ru", "flag": "🇷🇺", "multilingual": True},
    "العربية (Arabic)": {"code": "ar", "flag": "🇸🇦", "multilingual": True},
    "Nederlands (Dutch)": {"code": "nl", "flag": "🇳🇱", "multilingual": True},
    "Polski (Polish)": {"code": "pl", "flag": "🇵🇱", "multilingual": True},
    "हिन्दी (Hindi)": {"code": "hi", "flag": "🇮🇳", "multilingual": True},
    "Türkçe (Turkish)": {"code": "tr", "flag": "🇹🇷", "multilingual": True},
    "Svenska (Swedish)": {"code": "sv", "flag": "🇸🇪", "multilingual": True},
    "Dansk (Danish)": {"code": "da", "flag": "🇩🇰", "multilingual": True},
    "Suomi (Finnish)": {"code": "fi", "flag": "🇫🇮", "multilingual": True},
    "Norsk (Norwegian)": {"code": "no", "flag": "🇳🇴", "multilingual": True},
    "Ελληνικά (Greek)": {"code": "el", "flag": "🇬🇷", "multilingual": True},
    "עברית (Hebrew)": {"code": "he", "flag": "🇮🇱", "multilingual": True},
    "Bahasa Melayu (Malay)": {"code": "ms", "flag": "🇲🇾", "multilingual": True},
    "Kiswahili (Swahili)": {"code": "sw", "flag": "🇰🇪", "multilingual": True},
}

# 默认预设
DEFAULT_PRESETS = {
    "默认 (Balanced)": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 1.0, "description": "平衡设置，适合大多数场景", "builtin": True},
    "新闻播报 (News)": {"exaggeration": 0.2, "cfg_weight": 0.7, "temperature": 0.8, "description": "专业稳重的播报风格", "builtin": True},
    "故事讲述 (Story)": {"exaggeration": 0.7, "cfg_weight": 0.5, "temperature": 1.1, "description": "生动富有表现力", "builtin": True},
    "客服助手 (Service)": {"exaggeration": 0.4, "cfg_weight": 0.6, "temperature": 0.9, "description": "友好专业的语气", "builtin": True},
    "有声书 (Audiobook)": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 1.0, "description": "舒适的朗读风格", "builtin": True},
    "情感表达 (Emotional)": {"exaggeration": 0.9, "cfg_weight": 0.4, "temperature": 1.2, "description": "强烈的情感表达", "builtin": True},
}

# ==================== 预设管理 ====================


class PresetManager:
    """预设管理器"""

    def __init__(self):
        self._presets = self._load()

    def _load(self) -> dict:
        """加载预设"""
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
        """保存用户预设（只保存非内置的）"""
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
        """添加新预设"""
        if not name or not name.strip():
            return "❌ 预设名称不能为空"

        name = name.strip()

        if name in self._presets and self._presets[name].get("builtin", False):
            return f"❌ 不能覆盖内置预设 '{name}'"

        self._presets[name] = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "description": description,
            "builtin": False
        }
        self._save()
        return f"✅ 预设 '{name}' 已保存"

    def delete(self, name: str) -> str:
        """删除预设"""
        if name not in self._presets:
            return f"❌ 预设 '{name}' 不存在"

        if self._presets[name].get("builtin", False):
            return f"❌ 不能删除内置预设 '{name}'"

        del self._presets[name]
        self._save()
        return f"✅ 预设 '{name}' 已删除"


preset_manager = PresetManager()

# ==================== 模型管理器 ====================


class ModelManager:
    """双模型管理器"""

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
                logger.info("加载英文模型...")
                from chatterbox.tts import ChatterboxTTS
                self._english_model = ChatterboxTTS.from_pretrained(
                    device=self._device)
                logger.info("英文模型加载完成")
            return self._english_model

    def load_multilingual_model(self):
        with self._lock:
            if self._multilingual_model is None:
                logger.info("加载多语言模型...")
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                self._multilingual_model = ChatterboxMultilingualTTS.from_pretrained(
                    device=self._device)
                logger.info("多语言模型加载完成")
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
        return f"英文: {en} | 多语言: {ml}"


model_manager = ModelManager()

# ==================== 隐身模式状态 ====================
incognito_mode = False


def set_incognito_mode(enabled: bool) -> str:
    global incognito_mode
    incognito_mode = enabled
    logger.set_incognito(enabled)

    if enabled:
        return "🕵️ 隐身模式已开启\n• 不保存音频文件\n• 不记录历史\n• 不写入日志文件"
    else:
        return "👁️ 隐身模式已关闭\n• 正常保存文件和记录"

# ==================== 工具函数 ====================


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
        info.append(f"显存: {mem:.1f}GB")
    else:
        info.append("⚠️ CPU 模式")
    return " | ".join(info)


def get_gpu_stats() -> str:
    if not torch.cuda.is_available():
        return "GPU 不可用"
    used = torch.cuda.memory_allocated(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pct = (used / total) * 100
    bar_len = 20
    filled = int(bar_len * used / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    return f"[{bar}] {used:.1f}G / {total:.1f}G ({pct:.0f}%)"


def save_audio(audio_np: np.ndarray, sample_rate: int, fmt: str = "wav") -> str:
    """保存音频 - 隐身模式下返回临时文件"""
    if incognito_mode:
        # 隐身模式：使用临时文件
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
            sf.write(f.name, audio_np, sample_rate)
            return f.name
    else:
        # 正常模式：保存到 outputs
        ensure_dirs()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{ts}.{fmt}"
        filepath = os.path.join(CONFIG.output_dir, filename)
        sf.write(filepath, audio_np, sample_rate)
        logger.info(f"音频已保存: {filepath}")
        return filepath


def save_to_history(text: str, language: str, audio_path: str, params: dict):
    """保存历史 - 隐身模式下跳过"""
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
        return "暂无历史记录"
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except:
        return "暂无历史记录"

    lines = []
    for i, r in enumerate(history[:20], 1):
        ts = r.get("timestamp", "")[:16].replace("T", " ")
        txt = r.get("text", "")[:30]
        lang = r.get("language", "")[:8]
        lines.append(f"{i}. [{ts}] {lang} | {txt}")
    return "\n".join(lines) if lines else "暂无历史记录"


def clear_history() -> str:
    history_file = os.path.join(CONFIG.history_dir, "history.json")
    if os.path.exists(history_file):
        os.remove(history_file)
    return "✅ 历史已清空"

# ==================== 核心生成函数 ====================


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
        return None, "❌ 请输入文本", get_gpu_stats()

    text = text.strip()
    if len(text) > CONFIG.max_text_length:
        return None, f"❌ 文本过长 (最大 {CONFIG.max_text_length} 字)", get_gpu_stats()

    try:
        progress(0.1, desc="准备中...")

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

        progress(0.3, desc="加载模型...")
        progress(0.5, desc="生成中...")

        audio_np, sample_rate = model_manager.generate(
            text=text,
            language_code=lang_code,
            use_multilingual=use_multilingual,
            audio_prompt_path=reference_audio,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        progress(0.9, desc="保存...")

        audio_path = save_audio(audio_np, sample_rate, output_format)

        duration = len(audio_np) / sample_rate
        elapsed = time.time() - start_time
        rtf = elapsed / duration

        # 保存历史（隐身模式下跳过）
        save_to_history(text, language, audio_path, {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "seed": seed if not use_random_seed else "random"
        })

        progress(1.0, desc="完成!")

        clone_info = " | 🎭 克隆" if reference_audio else ""
        model_type = "多语言" if use_multilingual else "英文"
        incognito_info = " | 🕵️ 隐身" if incognito_mode else ""

        status = (
            f"✅ 成功!\n"
            f"⏱️ 时长: {duration:.2f}s | 耗时: {elapsed:.2f}s | RTF: {rtf:.2f}x\n"
            f"🌍 {flag} {language} | 模型: {model_type}{clone_info}{incognito_info}"
        )

        return audio_path, status, get_gpu_stats()

    except Exception as e:
        logger.error(f"生成失败: {e}", exc_info=True)
        return None, f"❌ 错误: {str(e)}", get_gpu_stats()


def apply_preset(preset_name: str) -> Tuple[float, float, float, str]:
    p = preset_manager.get(preset_name)
    if p:
        desc = p.get("description", "")
        builtin = "📌 内置" if p.get("builtin") else "👤 自定义"
        return p.get("exaggeration", 0.5), p.get("cfg_weight", 0.5), p.get("temperature", 1.0), f"{builtin} | {desc}"
    return 0.5, 0.5, 1.0, "⚠️ 预设不存在"


def save_new_preset(name: str, exaggeration: float, cfg_weight: float, temperature: float, description: str):
    """保存新预设"""
    result = preset_manager.add(
        name, exaggeration, cfg_weight, temperature, description)
    new_choices = preset_manager.get_names()
    return result, gr.update(choices=new_choices)


def delete_preset(name: str):
    """删除预设"""
    result = preset_manager.delete(name)
    new_choices = preset_manager.get_names()
    return result, gr.update(choices=new_choices)


def unload_models() -> str:
    model_manager.unload_all()
    return "✅ 模型已卸载"

# ==================== Gradio 界面 ====================


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

        # 标题
        gr.HTML(f"""
        <div class="main-title">🎙️ {CONFIG.app_name}</div>
        <div class="subtitle">多语言语音合成 · 语音克隆 · 情感控制 | v{CONFIG.version}</div>
        """)

        # 系统状态栏
        with gr.Row():
            gr.Markdown(f"**{get_system_info()}**")
            gpu_status = gr.Textbox(value=get_gpu_stats(
            ), label="GPU", interactive=False, scale=1)
            refresh_btn = gr.Button("🔄", scale=0, min_width=50)

            # 隐身模式开关
            incognito_toggle = gr.Checkbox(
                label="🕵️ 隐身模式", value=False, scale=0)

        incognito_status = gr.Textbox(
            value="", label="", interactive=False, visible=False)

        gr.Markdown("---")

        with gr.Tabs():
            # ===== 语音生成 =====
            with gr.TabItem("🎵 语音生成"):
                with gr.Row():
                    # 左栏
                    with gr.Column(scale=1):
                        text_input = gr.Textbox(
                            label="📝 输入文本", placeholder="请输入要合成的文本...", lines=8)

                        with gr.Row():
                            language = gr.Dropdown(choices=list(
                                LANGUAGES.keys()), value="English (英语)", label="🌍 语言", scale=2)
                            output_format = gr.Radio(
                                choices=["wav", "mp3"], value="wav", label="格式", scale=1)

                        gr.Markdown("#### 🎭 语音克隆（可选）")
                        reference_audio = gr.Audio(
                            label="参考音频 (5-15秒)", type="filepath", sources=["upload", "microphone"])

                    # 右栏
                    with gr.Column(scale=1):
                        gr.Markdown("#### ⚙️ 参数设置")

                        preset_select = gr.Dropdown(
                            choices=preset_manager.get_names(), value="默认 (Balanced)", label="📋 预设")
                        preset_info = gr.Textbox(
                            label="", interactive=False, lines=1)

                        exaggeration = gr.Slider(
                            0.0, 1.0, 0.5, step=0.05, label="🎭 情感夸张度", info="0=平淡, 1=夸张")
                        cfg_weight = gr.Slider(
                            0.0, 1.0, 0.5, step=0.05, label="🎯 CFG 权重", info="控制对参考音频的遵循程度")
                        temperature = gr.Slider(
                            0.1, 2.0, 1.0, step=0.1, label="🌡️ 温度")

                        with gr.Row():
                            seed = gr.Number(
                                value=42, label="🎲 种子", precision=0, scale=2)
                            use_random_seed = gr.Checkbox(
                                value=True, label="随机", scale=1)

                generate_btn = gr.Button(
                    "🚀 生成语音", variant="primary", size="lg")

                with gr.Row():
                    output_audio = gr.Audio(
                        label="🔊 结果", type="filepath", scale=2)
                    status_output = gr.Textbox(
                        label="📊 状态", lines=4, interactive=False, scale=1)

                # 示例
                gr.Markdown("#### 📚 示例")
                gr.Examples(
                    examples=[
                        ["Hello! Welcome to Chatterbox TTS.", "English (英语)"],
                        ["你好！欢迎使用语音合成系统。", "中文 (Chinese)"],
                        ["こんにちは！音声合成へようこそ。", "日本語 (Japanese)"],
                        ["Bonjour! Bienvenue!", "Français (French)"],
                    ],
                    inputs=[text_input, language]
                )

            # ===== 预设管理 =====
            with gr.TabItem("📋 预设管理"):
                gr.Markdown("### 自定义预设")
                gr.Markdown("创建自己的参数预设，方便快速切换不同风格。")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### ➕ 新建预设")
                        new_preset_name = gr.Textbox(
                            label="预设名称", placeholder="例如: 我的风格")
                        new_preset_desc = gr.Textbox(
                            label="描述", placeholder="例如: 适合朗读诗歌")

                        with gr.Row():
                            new_exaggeration = gr.Slider(
                                0.0, 1.0, 0.5, step=0.05, label="情感夸张度")
                            new_cfg = gr.Slider(
                                0.0, 1.0, 0.5, step=0.05, label="CFG 权重")
                            new_temp = gr.Slider(
                                0.1, 2.0, 1.0, step=0.1, label="温度")

                        save_preset_btn = gr.Button(
                            "💾 保存预设", variant="primary")
                        save_preset_result = gr.Textbox(
                            label="结果", interactive=False)

                    with gr.Column():
                        gr.Markdown("#### 🗑️ 删除预设")
                        delete_preset_select = gr.Dropdown(
                            choices=preset_manager.get_names(), label="选择要删除的预设")
                        delete_preset_btn = gr.Button("🗑️ 删除", variant="stop")
                        delete_preset_result = gr.Textbox(
                            label="结果", interactive=False)

                        gr.Markdown("---")
                        gr.Markdown("#### 📜 当前预设列表")
                        preset_list = gr.Textbox(
                            value="\n".join([f"{'📌' if p.get('builtin') else '👤'} {k}: {p.get('description', '')}"
                                             for k, p in preset_manager.get_all().items()]),
                            label="",
                            lines=10,
                            interactive=False
                        )

            # ===== 历史记录 =====
            with gr.TabItem("📜 历史记录"):
                history_display = gr.Textbox(
                    value=get_history_display(), label="最近记录", lines=15, interactive=False)
                with gr.Row():
                    refresh_history_btn = gr.Button("🔄 刷新")
                    clear_history_btn = gr.Button("🗑️ 清空", variant="stop")
                history_status = gr.Textbox(label="状态", interactive=False)

            # ===== 系统设置 =====
            with gr.TabItem("⚙️ 系统"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🕵️ 隐身模式说明")
                        gr.Markdown("""
                        开启隐身模式后：
                        - ❌ 不保存生成的音频文件到 outputs 目录
                        - ❌ 不记录历史
                        - ❌ 不写入日志文件
                        - ✅ 音频仍可播放和下载（临时文件）
                        
                        适合处理敏感内容时使用。
                        """)

                    with gr.Column():
                        gr.Markdown("### 模型管理")
                        model_status = gr.Textbox(
                            value=model_manager.get_status(), label="模型状态", interactive=False)
                        unload_btn = gr.Button("🔓 卸载模型（释放显存）")
                        unload_result = gr.Textbox(
                            label="结果", interactive=False)

                gr.Markdown("---")
                gr.Markdown(f"""
                ### 关于
                **{CONFIG.app_name}** v{CONFIG.version}
                
                基于 [Chatterbox](https://github.com/resemble-ai/chatterbox) | MIT License
                """)

        # ===== 事件绑定 =====
        refresh_btn.click(fn=lambda: get_gpu_stats(), outputs=[gpu_status])

        # 隐身模式
        incognito_toggle.change(fn=set_incognito_mode, inputs=[
                                incognito_toggle], outputs=[incognito_status])

        # 预设
        preset_select.change(fn=apply_preset, inputs=[preset_select], outputs=[
                             exaggeration, cfg_weight, temperature, preset_info])

        # 生成
        generate_btn.click(
            fn=generate_speech,
            inputs=[text_input, language, reference_audio, exaggeration,
                    cfg_weight, temperature, seed, use_random_seed, output_format],
            outputs=[output_audio, status_output, gpu_status]
        )

        # 保存预设
        save_preset_btn.click(
            fn=save_new_preset,
            inputs=[new_preset_name, new_exaggeration,
                    new_cfg, new_temp, new_preset_desc],
            outputs=[save_preset_result, preset_select]
        )

        # 删除预设
        delete_preset_btn.click(
            fn=delete_preset,
            inputs=[delete_preset_select],
            outputs=[delete_preset_result, preset_select]
        )

        # 历史
        refresh_history_btn.click(
            fn=get_history_display, outputs=[history_display])
        clear_history_btn.click(fn=clear_history, outputs=[history_status])

        # 卸载模型
        unload_btn.click(fn=unload_models, outputs=[unload_result])

    return app


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"🎙️  {CONFIG.app_name} v{CONFIG.version}")
    print("=" * 60)
    print(get_system_info())
    print("=" * 60)
    print("\n🚀 启动中...\n")

    app = create_app()
    app.launch(
        server_name=CONFIG.server_host,
        server_port=CONFIG.server_port,
        share=False,
        inbrowser=True,
        show_error=True
    )
