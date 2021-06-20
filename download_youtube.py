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

def youtube_download(audio_path_full, video_id):
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

def trim(audio_path_full, video_id, chucks, AUDIOS_FOLDER):
    song = AudioSegment.from_file(audio_path_full)
    # Trim the audio from start to end (in AudioSegment 1 second = 1000 samples)

    for chunk in chucks:
        start, end = chunk
        song_trim = song[1000*start:1000*end]
        audio_path_trim = os.path.join(AUDIOS_FOLDER, f"{video_id}_{start}_{end}.wav")
        song_trim.export(audio_path_trim, format="wav")


if __name__ == '__main__':
    VIDEOS_FOLDER = os.path.join("datasets", "MSVD", "videos")
    AUDIOS_FOLDER = os.path.join("datasets", "MSVD", "audios")
    AUDIOS_FULL_FOLDER = os.path.join("datasets", "MSVD", "audios_full")

    for folder in [AUDIOS_FOLDER, AUDIOS_FULL_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    
    ## get video list
    video_list = np.loadtxt('./datasets/MSVD/video_list.txt', dtype=str) # > ls [videos folder]/*.ave > video_list.txt
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

    failures_youtube = []    
    failures_trim = []    
    with tqdm(sorted(vid_time_dict.keys())) as progress:
        for video_id in progress:
            progress.set_postfix({"Processing file": f"{video_id}", "Failures": len(failures_youtube)})

            audio_path_full = os.path.join(AUDIOS_FULL_FOLDER, f"{video_id}.wav")
            if not os.path.isfile(audio_path_full):
                try:
                    youtube_download(audio_path_full, video_id)
                except Exception as e:
                    failures_youtube.append([video_id, str(e)])

                # try:
                #     chunks = vid_time_dict[video_id]
                #     trim(audio_path_full, video_id, chucks, AUDIOS_FOLDER)
                # except:
                #     failures_trim.append(video_id)


    print(f"Finished with {len(failures_youtube)} youtube failues, {len(failures_trim)} trim failues.")
    error_youtube = os.path.join("datasets", "MSVD", "error_youtube.txt")
    error_trim = os.path.join("datasets", "MSVD", "error_trim.txt")
    with open(error_youtube, 'w+') as f:
        lines = [','.join(l) for l in failures_youtube]
        f.write("\n".join(lines))
    with open(error_trim, 'w+') as f:
        f.write("\n".join(failures_trim))
