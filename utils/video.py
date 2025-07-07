from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

def extract_audio(video_path, output_path="temp_audio.wav"):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_path, fps=16000, codec='pcm_s16le')
    return output_path

import torchaudio

def chunk_audio(audio_path, chunk_length=5.0):
    waveform, sr = torchaudio.load(audio_path)
    step = int(sr * chunk_length)
    chunks = []
    for i in range(0, waveform.size(1), step):
        end = min(i + step, waveform.size(1))
        chunks.append((waveform[:, i:end], i / sr, end / sr))
    return chunks

def burn_captions(video_path, captions, output_path):
    video = VideoFileClip(video_path)
    clips = [video]

    for cap in captions:
        txt = cap["text"]
        start = cap["start"]
        duration = cap["end"] - cap["start"]

        if not txt.strip():
            continue

        txt_clip = TextClip(txt, fontsize=24, color='white', bg_color='black', method='caption')
        txt_clip = txt_clip.set_position(('center', 'bottom')).set_start(start).set_duration(duration)
        clips.append(txt_clip)

    final = CompositeVideoClip(clips)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
