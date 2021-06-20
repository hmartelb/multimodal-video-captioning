from __future__ import unicode_literals

import os

import numpy as np
import youtube_dl
from tqdm import tqdm
from pydub import AudioSegment

def read_video_filename(video_name):
    filename = video_name.split(".")[0]
    parts = filename.split("_")
    video_id = '_'.join(parts[:-2])
    start = parts[-2]
    end = parts[-1]
    start, end = int(start), int(end)
    return video_id, start, end



if __name__ == '__main__':
    VIDEOS_FOLDER = os.path.join("datasets", "MSVD", "videos")
    AUDIOS_FOLDER = os.path.join("datasets", "MSVD", "audios")
    AUDIOS_FULL_FOLDER = os.path.join("datasets", "MSVD", "audios_full")

    for folder in [AUDIOS_FOLDER, AUDIOS_FULL_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    
    ## get video list
    video_list = np.loadtxt('video_list.txt', dtype=str) # > ls [videos folder]/*.ave > video_list.txt
    # video_list = os.listdir(VIDEOS_FOLDER)

    ## Some video_id are repeated
    ## get { video_id : [start,end] }
    ## download video_id -> Trim base on [start,end]
    vid_time_dict = {}
    with tqdm(video_list) as progress:
        for video_name in progress:
            vid, start, end = read_video_filename(video_name)
            if vid not in vid_time_dict:
                vid_time_dict[vid] = []
            vid_time_dict[vid].append([start,end])
    print("Total Video:", len(vid_time_dict))

    failures_vid = []
    
    with tqdm(sorted(vid_time_dict.keys())) as progress:
        for video_id in progress:
            progress.set_postfix({"Processing file": f"{video_id}", "Failures": len(failures_vid)})

            audio_path_full = os.path.join(AUDIOS_FULL_FOLDER, f"{video_id}.wav")
            if not os.path.isfile(audio_path_full):
                try:
                    ydl_opts = {
                        "outtmpl": audio_path_full,
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
                    song = AudioSegment.from_file(audio_path_full)

                    for chunk in vid_time_dict[vid]:
                        start, end = chunk
                        song_trim = song[1000*start:1000*end]
                        audio_path_trim = os.path.join(AUDIOS_FOLDER, f"{video_id}_{start}_{end}.wav")
                        song_trim.export(audio_path_trim, format="wav")

                except:
                    failures_vid.append(video_name)

    print(f"Finished with {len(failures_vid)} failues.")
    error_txt = os.path.join(AUDIOS_FULL_FOLDER, "error.txt")
    np.savetxt(error_txt, np.array(failures_vid, dtype=str))
