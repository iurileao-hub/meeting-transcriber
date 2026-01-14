#!/usr/bin/env python3
"""
Meeting Transcriber - Sistema de transcrição com identificação de speakers.

Uso:
    python transcribe.py <audio_file> [opções]

Exemplos:
    python transcribe.py reuniao.wav
    python transcribe.py reuniao.mp3 --model medium --language pt
    python transcribe.py reuniao.wav --num-speakers 4 --output transcripts/
"""

import argparse
import gc
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


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
        if self._stdout:
            sys.stdout = self._stdout
        if self._stderr:
            sys.stderr = self._stderr
        if self._devnull:
            self._devnull.close()


# Aplicar configuração de warnings na importação
configure_warnings()

# Fix para PyTorch 2.6+ que mudou weights_only=True por padrão
# Necessário para carregar modelos pyannote que usam formato antigo
# Os modelos do HuggingFace/pyannote são confiáveis
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Força weights_only=False, sobrescrevendo qualquer valor
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Patch também no módulo serialization para garantir
import torch.serialization
torch.serialization.load = _patched_torch_load


class TranscriptionError(Exception):
    """Erro durante o processo de transcrição."""

    pass


# Formatos de áudio suportados pelo whisperX/ffmpeg
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}


def load_hf_token() -> str:
    """Carrega token HuggingFace do ambiente ou .env.

    Returns:
        Token de autenticação do HuggingFace.

    Raises:
        TranscriptionError: Se o token não for encontrado.
    """
    env_file = Path(__file__).parent.parent / ".env"
    load_dotenv(env_file)

    token = os.getenv("HF_TOKEN")
    if not token:
        raise TranscriptionError(
            "Token HuggingFace não encontrado.\n"
            "Configure HF_TOKEN no arquivo .env ou como variável de ambiente.\n"
            "Obtenha seu token em: https://huggingface.co/settings/tokens"
        )
    return token


def get_compute_type(device: str) -> str:
    """Retorna o compute_type ideal para o dispositivo.

    Args:
        device: Dispositivo de processamento (cpu, cuda, mps).

    Returns:
        Tipo de computação otimizado para o dispositivo.
    """
    if device == "cpu":
        return "int8"  # Mais rápido em CPU
    return "float16"  # GPU (cuda/mps) - 2x mais rápido


def get_batch_size(device: str, model_size: str) -> int:
    """Calcula batch_size ideal baseado em recursos disponíveis.

    Args:
        device: Dispositivo de processamento.
        model_size: Tamanho do modelo Whisper.

    Returns:
        Tamanho do batch otimizado.
    """
    if device == "cpu":
        return 8  # Conservador para CPU
    if model_size in ("tiny", "base", "small"):
        return 32  # Modelos menores permitem batches maiores
    return 16  # medium, large-v3


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


def transcribe(
    audio_path: str,
    model_size: str = "large-v3",
    language: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    output_dir: str = "data/transcripts",
    output_format: str = "all",
    device: str = "cpu",
    verbose: bool = False,
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

    Returns:
        Dicionário com transcrição e metadados

    Raises:
        TranscriptionError: Se ocorrer erro em qualquer etapa do processo.
    """
    import whisperx

    audio_path_obj = Path(audio_path)
    validate_audio_file(audio_path_obj)

    compute_type = get_compute_type(device)
    batch_size = get_batch_size(device, model_size)

    # [1/5] Carregar modelo
    try:
        print(f"[1/5] Carregando modelo {model_size} ({compute_type})...")
        # Suprimir prints de bibliotecas externas durante carregamento
        with SuppressOutput(suppress_stdout=not verbose, suppress_stderr=not verbose):
            model = whisperx.load_model(
                model_size,
                device,
                compute_type=compute_type,
                language=language,
            )
    except Exception as e:
        raise TranscriptionError(
            f"Falha ao carregar modelo '{model_size}': {e}\n"
            "Verifique sua conexão com a internet para download do modelo."
        ) from e

    # [2/5] Transcrever áudio
    try:
        print(f"[2/5] Transcrevendo {audio_path} (batch_size={batch_size})...")
        with SuppressOutput(suppress_stdout=not verbose, suppress_stderr=not verbose):
            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result.get("language", language or "unknown")
    except Exception as e:
        raise TranscriptionError(
            f"Falha ao transcrever áudio '{audio_path}': {e}\n"
            "Verifique se o arquivo de áudio está corrompido ou se FFmpeg está instalado."
        ) from e

    print(f"      Idioma detectado: {detected_language}")

    # Liberar memória do modelo de transcrição
    del model
    gc.collect()

    # [3/5] Alinhar palavras
    try:
        print("[3/5] Alinhando palavras...")
        with SuppressOutput(suppress_stdout=not verbose, suppress_stderr=not verbose):
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=device,
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
    except Exception as e:
        raise TranscriptionError(
            f"Falha ao alinhar palavras: {e}\n"
            f"Modelo de alinhamento pode não estar disponível para '{detected_language}'."
        ) from e

    # Liberar memória do modelo de alinhamento
    del model_a
    gc.collect()

    # [4/5] Identificar speakers
    try:
        print("[4/5] Identificando speakers...")
        hf_token = load_hf_token()
        from whisperx.diarize import DiarizationPipeline

        with SuppressOutput(suppress_stdout=not verbose, suppress_stderr=not verbose):
            diarize_model = DiarizationPipeline(
                use_auth_token=hf_token,
                device=device,
            )

            diarize_kwargs = {}
            if num_speakers:
                diarize_kwargs["num_speakers"] = num_speakers
            if min_speakers:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers:
                diarize_kwargs["max_speakers"] = max_speakers

            diarize_segments = diarize_model(audio, **diarize_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)
    except TranscriptionError:
        raise  # Re-raise TranscriptionError (token não encontrado)
    except Exception as e:
        raise TranscriptionError(
            f"Falha na identificação de speakers: {e}\n"
            "Verifique se o token HuggingFace é válido e se você aceitou os termos:\n"
            "- https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "- https://huggingface.co/pyannote/segmentation-3.0"
        ) from e

    # Liberar memória do modelo de diarização
    del diarize_model
    gc.collect()

    # Adicionar metadados
    result["metadata"] = {
        "source_file": audio_path_obj.name,
        "transcribed_at": datetime.now().isoformat(),
        "model": model_size,
        "language": detected_language,
        "device": device,
        "compute_type": compute_type,
        "num_segments": len(result.get("segments", [])),
    }

    # Contar speakers únicos
    speakers = set()
    for seg in result.get("segments", []):
        if "speaker" in seg:
            speakers.add(seg["speaker"])
    result["metadata"]["num_speakers"] = len(speakers)

    # [5/5] Salvar resultado
    try:
        print("[5/5] Salvando resultado...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        base_name = audio_path_obj.stem
        saved_files = []

        # Determinar quais formatos salvar
        formats_to_save = []
        if output_format == "all":
            formats_to_save = ["json", "txt", "md"]
        else:
            formats_to_save = [output_format]

        # Salvar em cada formato solicitado
        for fmt in formats_to_save:
            if fmt == "json":
                output_file = output_path / f"{base_name}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                saved_files.append(output_file)
            elif fmt == "txt":
                output_file = output_path / f"{base_name}.txt"
                save_as_text(result, output_file)
                saved_files.append(output_file)
            elif fmt == "md":
                output_file = output_path / f"{base_name}.md"
                save_as_markdown(result, output_file)
                saved_files.append(output_file)

    except Exception as e:
        raise TranscriptionError(
            f"Falha ao salvar transcrição: {e}\n"
            f"Verifique permissões de escrita em '{output_dir}'."
        ) from e

    print(f"\n✓ Transcrição salva em:")
    for f in saved_files:
        print(f"  - {f}")
    print(f"\n  Segmentos: {result['metadata']['num_segments']}")
    print(f"  Speakers identificados: {result['metadata']['num_speakers']}")

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
        "--model", "-m",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Tamanho do modelo Whisper (default: large-v3)",
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
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Dispositivo de processamento (default: cpu)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostra warnings e logs detalhados (útil para debug)",
    )

    args = parser.parse_args()

    # Reconfigurar warnings se modo verbose
    if args.verbose:
        configure_warnings(verbose=True)

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
        )
    except TranscriptionError as e:
        print(f"\n✗ Erro: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠ Transcrição cancelada pelo usuário.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
