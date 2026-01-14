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
import json
import os
from datetime import datetime
from pathlib import Path

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


def load_hf_token() -> str:
    """Carrega token HuggingFace do ambiente ou .env"""
    token = os.getenv("HF_TOKEN")
    if not token:
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        token = line.strip().split("=", 1)[1].strip('"\'')
                        break
    if not token:
        raise ValueError(
            "Token HuggingFace não encontrado. "
            "Configure HF_TOKEN no .env ou como variável de ambiente."
        )
    return token


def transcribe(
    audio_path: str,
    model_size: str = "large-v3",
    language: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    output_dir: str = "data/transcripts",
    device: str = "cpu",
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
        output_dir: Diretório para salvar o JSON de saída
        device: Dispositivo de processamento (cpu, cuda, mps)

    Returns:
        Dicionário com transcrição e metadados
    """
    import whisperx

    print(f"[1/5] Carregando modelo {model_size}...")
    model = whisperx.load_model(
        model_size,
        device,
        compute_type="float32",  # float16 se GPU disponível
        language=language,
    )

    print(f"[2/5] Transcrevendo {audio_path}...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)

    detected_language = result.get("language", language or "unknown")
    print(f"      Idioma detectado: {detected_language}")

    print("[3/5] Alinhando palavras...")
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

    print("[4/5] Identificando speakers...")
    hf_token = load_hf_token()
    from whisperx.diarize import DiarizationPipeline
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

    # Adicionar metadados
    audio_path_obj = Path(audio_path)
    result["metadata"] = {
        "source_file": audio_path_obj.name,
        "transcribed_at": datetime.now().isoformat(),
        "model": model_size,
        "language": detected_language,
        "num_segments": len(result.get("segments", [])),
    }

    # Contar speakers únicos
    speakers = set()
    for seg in result.get("segments", []):
        if "speaker" in seg:
            speakers.add(seg["speaker"])
    result["metadata"]["num_speakers"] = len(speakers)

    print("[5/5] Salvando resultado...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{audio_path_obj.stem}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nTranscrição salva em: {output_file}")
    print(f"Segmentos: {result['metadata']['num_segments']}")
    print(f"Speakers identificados: {result['metadata']['num_speakers']}")

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
        help="Diretório de saída para o JSON (default: data/transcripts)",
    )
    parser.add_argument(
        "--device", "-d",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Dispositivo de processamento (default: cpu)",
    )

    args = parser.parse_args()

    # Validar arquivo de entrada
    if not Path(args.audio_file).exists():
        parser.error(f"Arquivo não encontrado: {args.audio_file}")

    transcribe(
        audio_path=args.audio_file,
        model_size=args.model,
        language=args.language,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        output_dir=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
