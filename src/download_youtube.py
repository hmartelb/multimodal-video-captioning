from __future__ import unicode_literals

import os

import youtube_dl
from tqdm import tqdm
from pydub import AudioSegment

if __name__ == '__main__':
    VIDEOS_FOLDER = os.path.join("datasets", "MSVD", "videos")
    AUDIOS_FOLDER = os.path.join("datasets", "MSVD", "audios")

    failures = []
    with tqdm(os.listdir(VIDEOS_FOLDER)) as progress:
        for video_name in progress:
            video_id, start, end = video_name.split(".")[0].split("_")
            start, end = int(start), int(end)
            
            progress.set_postfix({"Processing file": f"{video_id}", "Failures": len(failures)})

            output_name = os.path.join(AUDIOS_FOLDER, video_name.replace("avi", "wav"))
            if not os.path.isfile(output_name):
                try:
                    ydl_opts = {
                        "outtmpl": output_name,
                        "format": "bestaudio/best",
                        "postprocessors": [
                            {
                                "key": "FFmpegExtractAudio",
                                "preferredcodec": "wav",
                                # "preferredquality": "192",
                            }
                        ],
                    }
                    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                    
                    # Trim the audio from start to end (in AudioSegment 1 second = 1000 samples)
                    song = AudioSegment.from_file(output_name)
                    song = song[1000*start:1000*end]
                    song.export(output_name, format="wav")

                except:
                    failures.append(video_name)

    print(f"Finished with {len(failures)} failues.")
    print(failures)
