from pathlib import Path
from typing import Optional
import av


def extract_audio_from_video(video_path: Path, output_wav_path: Path) -> Optional[Path]:
    """
    Extract and resample the first audio stream from a video file into a
    16 kHz mono PCM WAV file using PyAV (no external ffmpeg binary required).

    Returns output_wav_path if an audio stream was extracted, or None if the
    video has no audio stream.
    """
    with av.open(str(video_path)) as in_container:
        audio_streams = [s for s in in_container.streams if s.type == "audio"]
        if not audio_streams:
            return None

        in_stream = audio_streams[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(output_wav_path), mode="w", format="wav") as out_container:
            out_stream = out_container.add_stream("pcm_s16le", rate=16000)
            out_stream.layout = "mono"

            wrote_frames = False
            for frame in in_container.decode(in_stream):
                frame.pts = None
                resampled_frames = resampler.resample(frame)
                if not isinstance(resampled_frames, list):
                    resampled_frames = [resampled_frames]
                for rframe in resampled_frames:
                    if rframe is None:
                        continue
                    for packet in out_stream.encode(rframe):
                        out_container.mux(packet)
                        wrote_frames = True

            # Flush resampler & encoder
            for rframe in resampler.resample(None) or []:
                for packet in out_stream.encode(rframe):
                    out_container.mux(packet)
                    wrote_frames = True

            for packet in out_stream.encode(None):
                out_container.mux(packet)
                wrote_frames = True

            if not wrote_frames:
                return None

    return output_wav_path
