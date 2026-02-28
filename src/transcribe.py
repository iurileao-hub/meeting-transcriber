#!/usr/bin/env python3
"""
Meeting Transcriber - Sistema de transcrição com identificação de speakers.

Uso:
    python transcribe.py <audio_file> [opções]

Exemplos:
    python transcribe.py reuniao.wav
    python transcribe.py reuniao.mp3 --model medium --language pt
    python transcribe.py reuniao.wav --num-speakers 4 --output transcripts/
    python transcribe.py reuniao.wav --mode fast --notify
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable

# Ensure src is importable
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Phase 3 imports - new features
from src.backends import get_backend, TranscriptionResult
from src.backends.base import get_default_device
from src.i18n import get_translator
from src.progress import ProgressReporter, Stage
from src.notify import notify
from src.vocabulary import load_vocabulary
from src.normalize import normalize_text


def configure_warnings(verbose: bool = False) -> None:
    """Configura filtros de warnings para reduzir ruído no output.

    Args:
        verbose: Se True, mostra todos os warnings (útil para debug).
    """
    if verbose:
        return  # Não filtra nada em modo verbose

    # Suprimir warnings de deprecação do torchaudio (conhecido, será resolvido upstream)
    warnings.filterwarnings(
        "ignore",
        message=".*torchaudio._backend.list_audio_backends has been deprecated.*",
        category=UserWarning,
    )

    # Suprimir warning de versão do pyannote (modelos funcionam normalmente)
    warnings.filterwarnings(
        "ignore",
        message=".*Model was trained with pyannote.audio.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Model was trained with torch.*",
        category=UserWarning,
    )

    # Suprimir warning estatístico de std() com poucos dados
    warnings.filterwarnings(
        "ignore",
        message=".*std\\(\\): degrees of freedom is <= 0.*",
        category=UserWarning,
    )

    # Suprimir FutureWarnings de bibliotecas externas
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="speechbrain.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="pyannote.*",
    )

    # Reduzir verbosidade do logging de bibliotecas externas
    logging.getLogger("whisperx").setLevel(logging.ERROR)
    logging.getLogger("whisperx.asr").setLevel(logging.ERROR)
    logging.getLogger("whisperx.vads").setLevel(logging.ERROR)
    logging.getLogger("whisperx.diarize").setLevel(logging.ERROR)
    logging.getLogger("faster_whisper").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("pyannote").setLevel(logging.ERROR)
    logging.getLogger("pyannote.audio").setLevel(logging.ERROR)
    logging.getLogger("speechbrain").setLevel(logging.ERROR)


class SuppressOutput:
    """Context manager para suprimir prints indesejados de bibliotecas externas."""

    def __init__(self, suppress_stdout: bool = False, suppress_stderr: bool = True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None
        self._devnull = None

    def __enter__(self):
        self._devnull = open(os.devnull, "w")
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = self._devnull
        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = self._devnull
        return self

    def __exit__(self, *args):
        try:
            if self._stdout is not None:
                sys.stdout = self._stdout
            if self._stderr is not None:
                sys.stderr = self._stderr
        finally:
            if self._devnull is not None:
                self._devnull.close()
        return False  # Don't suppress exceptions


@contextmanager
def _allow_legacy_torch_load():
    """Temporarily allow pickle-based torch.load for trusted pyannote models.

    PyTorch 2.6+ defaults to weights_only=True for security. Pyannote models
    require the legacy behavior. This narrows the security exception to only
    the specific code that needs it.
    """
    import torch
    import torch.serialization as torch_ser

    original_load = torch.load
    original_ser_load = torch_ser.load

    def patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched
    torch_ser.load = patched
    try:
        yield
    finally:
        torch.load = original_load
        torch_ser.load = original_ser_load


class TranscriptionError(Exception):
    """Erro durante o processo de transcrição."""

    pass


# Formatos de áudio suportados pelo whisperX/ffmpeg
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".opus"}

# System directories that should never be written to (security)
# Note: /private/var/folders is used for temp dirs on macOS, so we allow it
FORBIDDEN_OUTPUT_PATHS = {
    "/etc", "/sys", "/proc", "/root", "/var/log", "/boot",
    "/usr/bin", "/usr/sbin", "/usr/lib", "/bin", "/sbin",
    "/System", "/Library",  # macOS system directories
    str(Path.home() / ".ssh"),
    str(Path.home() / ".aws"),
    str(Path.home() / ".kube"),
}


def validate_output_path(output_dir: str) -> Path:
    """Validate output directory path for security.

    Prevents writing to system directories or other sensitive locations.

    Args:
        output_dir: Output directory path (relative or absolute).

    Returns:
        Resolved absolute Path object.

    Raises:
        TranscriptionError: If path is in a forbidden location.
    """
    path = Path(output_dir).resolve()
    path_str = str(path)

    # Check against forbidden paths
    for forbidden in FORBIDDEN_OUTPUT_PATHS:
        if path_str == forbidden or path_str.startswith(forbidden + "/"):
            raise TranscriptionError(
                f"Cannot write to system directory: {forbidden}\n"
                f"Please choose a different output location."
            )

    return path


def validate_audio_file(audio_path: Path) -> None:
    """Valida se o arquivo de áudio existe e tem formato suportado.

    Args:
        audio_path: Caminho para o arquivo de áudio.

    Raises:
        TranscriptionError: Se o arquivo não existir ou formato inválido.
    """
    if not audio_path.exists():
        raise TranscriptionError(f"Arquivo não encontrado: {audio_path}")

    suffix = audio_path.suffix.lower()
    if suffix not in SUPPORTED_AUDIO_FORMATS:
        raise TranscriptionError(
            f"Formato não suportado: '{suffix}'\n"
            f"Formatos aceitos: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}"
        )


def format_timestamp(seconds: float) -> str:
    """Formata segundos como timestamp legível [MM:SS] ou [HH:MM:SS].

    Args:
        seconds: Tempo em segundos.

    Returns:
        String formatada como [MM:SS] ou [HH:MM:SS] para áudios > 1h.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes:02d}:{secs:02d}]"


def save_as_text(result: dict, output_file: Path) -> None:
    """Salva transcrição em formato texto simples.

    Formato:
        [MM:SS] SPEAKER_XX: Texto do segmento.

    Args:
        result: Dicionário com resultado da transcrição.
        output_file: Caminho do arquivo de saída (.txt).
    """
    lines = []
    for segment in result.get("segments", []):
        timestamp = format_timestamp(segment.get("start", 0))
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        lines.append(f"{timestamp} {speaker}: {text}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))


def save_as_markdown(result: dict, output_file: Path) -> None:
    """Salva transcrição em formato Markdown estruturado.

    Formato:
        # Transcrição: filename
        **Data:** ... | **Idioma:** ... | **Speakers:** N

        ---
        [MM:SS - MM:SS] **SPEAKER_XX**
        > Texto do segmento.

    Args:
        result: Dicionário com resultado da transcrição.
        output_file: Caminho do arquivo de saída (.md).
    """
    metadata = result.get("metadata", {})
    segments = result.get("segments", [])

    lines = [
        f"# Transcrição: {metadata.get('source_file', 'audio')}",
        "",
        f"**Data:** {metadata.get('transcribed_at', 'N/A')[:10]} | "
        f"**Idioma:** {metadata.get('language', 'N/A')} | "
        f"**Modelo:** {metadata.get('model', 'N/A')} | "
        f"**Speakers:** {metadata.get('num_speakers', 'N/A')}",
        "",
        "---",
        "",
    ]

    for segment in segments:
        start = format_timestamp(segment.get("start", 0))
        end = format_timestamp(segment.get("end", 0))
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()

        lines.append(f"{start} - {end} **{speaker}**")
        lines.append(f"> {text}")
        lines.append("")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# Formatos de saída suportados
SUPPORTED_OUTPUT_FORMATS = {"json", "txt", "md", "all"}


def _save_results(
    result: dict,
    output_dir: str,
    audio_stem: str,
    output_format: str,
) -> list[Path]:
    """Save transcription results in requested formats.

    Args:
        result: Transcription result dictionary with segments and metadata.
        output_dir: Directory to save files to.
        audio_stem: Base name for output files (without extension).
        output_format: Format to save ('json', 'txt', 'md', or 'all').

    Returns:
        List of saved file paths.

    Raises:
        TranscriptionError: If output path is forbidden or write fails.
    """
    output_path = validate_output_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    formats_to_save = (
        ["json", "txt", "md"] if output_format == "all" else [output_format]
    )

    for fmt in formats_to_save:
        if fmt == "json":
            output_file = output_path / f"{audio_stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            saved_files.append(output_file)
        elif fmt == "txt":
            output_file = output_path / f"{audio_stem}.txt"
            save_as_text(result, output_file)
            saved_files.append(output_file)
        elif fmt == "md":
            output_file = output_path / f"{audio_stem}.md"
            save_as_markdown(result, output_file)
            saved_files.append(output_file)

    return saved_files


def transcribe(
    audio_path: str,
    model_size: str = "large-v3",
    language: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    output_dir: str = "data/transcripts",
    output_format: str = "all",
    device: str | None = None,
    verbose: bool = False,
    mode: str = "meeting",
    translator: Callable[[str], str] | None = None,
    progress: ProgressReporter | None = None,
    vocabulary: list[str] | None = None,
    send_notify: bool = False,
    enable_diarization: bool = False,
) -> dict:
    """
    Transcreve áudio com identificação de speakers.

    Args:
        audio_path: Caminho para arquivo de áudio (.wav, .mp3, etc.)
        model_size: Tamanho do modelo Whisper (tiny, base, small, medium, large-v3)
        language: Código do idioma (pt, en, etc.) ou None para detecção automática
        num_speakers: Número exato de speakers (se conhecido)
        min_speakers: Número mínimo de speakers esperado
        max_speakers: Número máximo de speakers esperado
        output_dir: Diretório para salvar os arquivos de saída
        output_format: Formato de saída (json, txt, md, all)
        device: Dispositivo de processamento (cpu, cuda, mps)
        verbose: Se True, mostra warnings e logs detalhados
        mode: Modo de transcrição (fast, meeting, precise)
        translator: Função de tradução para i18n
        progress: Reporter de progresso
        vocabulary: Lista de termos customizados
        send_notify: Se True, envia notificação ao finalizar
        enable_diarization: Se True, ativa diarização no modo fast

    Returns:
        Dicionário com transcrição e metadados

    Raises:
        TranscriptionError: Se ocorrer erro em qualquer etapa do processo.
    """
    start_time = time.time()

    # Validate speaker parameters
    if num_speakers is not None and num_speakers <= 0:
        raise TranscriptionError("num_speakers must be positive")
    if min_speakers is not None and min_speakers <= 0:
        raise TranscriptionError("min_speakers must be positive")
    if max_speakers is not None and max_speakers <= 0:
        raise TranscriptionError("max_speakers must be positive")
    if min_speakers and max_speakers and min_speakers > max_speakers:
        raise TranscriptionError("min_speakers cannot exceed max_speakers")

    # Auto-detect device if not specified
    if device is None:
        device = get_default_device()

    # Initialize defaults for new parameters
    if translator is None:
        translator = get_translator()

    # Determine UI language from translator
    ui_lang = getattr(translator, "lang", "en")

    # Get the appropriate backend for the mode with configuration
    try:
        backend = get_backend(
            mode,
            model_size=model_size,
            device=device,
            enable_diarization=enable_diarization,
            # hf_token loaded from env by backend if needed
        )
    except ValueError as e:
        raise TranscriptionError(translator("messages.invalid_mode")) from e

    # Determine number of stages from backend
    total_stages = backend.total_stages

    if progress is None:
        progress = ProgressReporter(total_stages=total_stages, lang=ui_lang)

    audio_path_obj = Path(audio_path)
    validate_audio_file(audio_path_obj)

    # Create progress callback for backend
    stage_map = {
        "loading": Stage.LOADING,
        "vad": Stage.VAD,
        "transcribing": Stage.TRANSCRIBING,
        "aligning": Stage.ALIGNING,
        "diarizing": Stage.DIARIZING,
    }
    current_stage_name = [None]  # Use list to allow modification in closure

    def progress_callback(stage_name: str, percent: float) -> None:
        """Callback for backend to report progress."""
        # Validate percent range
        if not isinstance(percent, (int, float)):
            return
        percent = max(0, min(100, percent))  # Clamp to 0-100

        stage = stage_map.get(stage_name)
        if stage:
            # Advance to next stage if we're starting a new one
            if current_stage_name[0] != stage_name:
                if current_stage_name[0] is not None:
                    # Complete previous stage at 100% before advancing
                    prev_stage = stage_map.get(current_stage_name[0])
                    if prev_stage:
                        progress.update(prev_stage, 100)
                    progress.advance()
                current_stage_name[0] = stage_name
            progress.update(stage, percent)

    # Transcribe using the backend
    try:
        with _allow_legacy_torch_load():
            with SuppressOutput(
                suppress_stdout=not verbose, suppress_stderr=not verbose
            ):
                backend_result = backend.transcribe(
                    audio_path=audio_path,
                    language=language,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    progress_callback=progress_callback,
                )
    except ValueError as e:
        progress.error(str(e))
        raise TranscriptionError(str(e)) from e
    except Exception as e:
        progress.error(str(e) if verbose else "Transcription pipeline failed")
        if verbose:
            raise TranscriptionError(f"Transcription failed: {e}") from e
        raise TranscriptionError(
            "Transcription failed. Run with --verbose for details.\n"
            "Check your HuggingFace token and model access permissions."
        ) from e

    # Apply text normalization to segments
    segments = backend_result.segments
    for segment in segments:
        if "text" in segment:
            segment["text"] = normalize_text(segment["text"])

    # Convert TranscriptionResult to dict format
    result = {
        "segments": segments,
        "metadata": {
            "source_file": audio_path_obj.name,
            "transcribed_at": datetime.now().isoformat(),
            "model": model_size,
            "language": backend_result.language,
            "device": device,
            "mode": mode,
            "backend": backend.name,
            "num_segments": len(segments),
            **backend_result.metadata,
        },
    }

    # Count unique speakers
    speakers = set()
    for seg in segments:
        if "speaker" in seg:
            speakers.add(seg["speaker"])
    result["metadata"]["num_speakers"] = len(speakers)

    # Add vocabulary info if used
    if vocabulary:
        result["metadata"]["vocabulary_terms"] = len(vocabulary)

    # Advance past the last backend stage before saving
    if current_stage_name[0] is not None:
        prev_stage = stage_map.get(current_stage_name[0])
        if prev_stage:
            progress.update(prev_stage, 100)
        progress.advance()

    # Stage: Saving (update progress directly, not via callback)
    progress.update(Stage.SAVING, 0)

    # Save results
    try:
        saved_files = _save_results(result, output_dir, audio_path_obj.stem, output_format)
        progress.update(Stage.SAVING, 100)
    except TranscriptionError:
        raise
    except Exception as e:
        progress.error(str(e))
        raise TranscriptionError(
            f"Failed to save transcription: {e}\n"
            f"Check write permissions for '{output_dir}'."
        ) from e

    # Calculate duration
    duration = time.time() - start_time

    # Show completion message
    progress.advance()  # Move past saving stage
    if saved_files:
        progress.complete(str(saved_files[0]), duration)

    # Print summary
    print(f"\n  {translator('stages.saving')}: {len(saved_files)} files")
    for f in saved_files:
        print(f"    - {f}")
    print(f"\n  {translator('output.segments')}: {result['metadata']['num_segments']}")
    print(f"  {translator('output.speakers')}: {result['metadata']['num_speakers']}")

    # Send notification if requested
    if send_notify:
        notify(
            title=translator("messages.complete"),
            message=f"{audio_path_obj.name} - {result['metadata']['num_segments']} segments",
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Transcreve áudio de reuniões com identificação de speakers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python transcribe.py reuniao.wav
  python transcribe.py reuniao.mp3 --model medium --language pt
  python transcribe.py reuniao.wav --num-speakers 4
  python transcribe.py reuniao.wav --mode fast --notify

Modos de transcrição:
  fast    - MLX backend, sem diarização (mais rápido)
  meeting - WhisperX backend com diarização (padrão)
  precise - Granite backend, maior precisão

Modelos disponíveis (menor → maior):
  tiny, base, small, medium, large-v3

Formatos de saída:
  json  - Completo com timestamps de cada palavra (para processamento)
  txt   - Texto simples com speakers (para leitura rápida)
  md    - Markdown formatado (para revisão/edição)
  all   - Gera todos os formatos acima (padrão)

Nota: Requer token HuggingFace configurado no .env para speaker diarization.
        """,
    )

    parser.add_argument(
        "audio_file",
        help="Caminho para o arquivo de áudio",
    )
    parser.add_argument(
        "--mode",
        default="meeting",
        choices=["fast", "meeting", "precise"],
        help="Modo de transcrição: fast (MLX, sem diarização), meeting (WhisperX, padrão), precise (Granite)",
    )
    parser.add_argument(
        "--model", "-m",
        default="large-v3",
        help="Model size. Standard: tiny/base/small/medium/large-v3. "
             "MLX extras: large-v3-turbo, distil-large-v3, large-v3-8bit",
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Código do idioma (pt, en, etc.). Se não especificado, detecta automaticamente.",
    )
    parser.add_argument(
        "--num-speakers", "-n",
        type=int,
        default=None,
        help="Número exato de speakers (se conhecido)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Número mínimo de speakers",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Número máximo de speakers",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/transcripts",
        help="Diretório de saída (default: data/transcripts)",
    )
    parser.add_argument(
        "--format", "-f",
        default="all",
        choices=["json", "txt", "md", "all"],
        help="Formato de saída: json (técnico), txt (simples), md (markdown), all (todos). Default: all",
    )
    parser.add_argument(
        "--device", "-d",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Dispositivo de processamento (default: auto-detecta mps/cuda/cpu)",
    )
    parser.add_argument(
        "--ui-lang",
        default=None,
        help="Idioma da interface (en, pt). Default: detecta automaticamente.",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Envia notificação do sistema ao finalizar (macOS)",
    )
    parser.add_argument(
        "--vocab",
        default=None,
        help="Caminho para arquivo de vocabulário customizado (um termo por linha)",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization in fast mode (requires HuggingFace token)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostra warnings e logs detalhados (útil para debug)",
    )

    args = parser.parse_args()

    # Configure warning filters (only when running as CLI)
    configure_warnings(verbose=args.verbose)

    # Auto-detect device if not specified
    if args.device is None:
        args.device = get_default_device()

    # Initialize i18n translator
    translator = get_translator(args.ui_lang)

    # Determine UI language for progress reporter
    ui_lang = getattr(translator, "lang", "en")

    # Load vocabulary
    vocabulary = []
    if args.vocab:
        vocabulary = load_vocabulary(extra_files=[args.vocab])
    else:
        # Auto-load default vocabulary if exists
        default_vocab = Path(__file__).parent.parent / "vocab" / "default.txt"
        if default_vocab.exists():
            vocabulary = load_vocabulary()

    try:
        transcribe(
            audio_path=args.audio_file,
            model_size=args.model,
            language=args.language,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            output_dir=args.output,
            output_format=args.format,
            device=args.device,
            verbose=args.verbose,
            mode=args.mode,
            translator=translator,
            vocabulary=vocabulary if vocabulary else None,
            send_notify=args.notify,
            enable_diarization=args.diarize,
        )
    except TranscriptionError as e:
        print(f"\n{translator('messages.error')}: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n{translator('messages.cancelled')}", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
