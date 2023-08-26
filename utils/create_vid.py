import os

import ffmpeg


def create_vid_from_recon_folder(folder_path, extension="mp4"):
    ts_file_path = os.path.join(folder_path, "timestamps.txt")
    with open(ts_file_path, 'r', encoding="utf-8") as ts_file:
        lines = ts_file.readlines()
        _, start_ts = lines[0].split()
        _, end_ts = lines[-1].split()
    duration = float(end_ts) - float(start_ts)
    frame_count = len(lines)
    fps = round(frame_count / duration)
    frame_paths = os.path.join(folder_path, "frame_%010d.png")
    vid_path = os.path.normpath(folder_path) + "_{}Hz.{}".format(fps, extension)
    if os.path.exists(vid_path):
        os.remove(vid_path)
    stream = ffmpeg.input(frame_paths, framerate=fps)
    stream = ffmpeg.output(stream, vid_path, crf=11, preset='slow')
    stream = stream.global_args('-loglevel', 'error')
    ffmpeg.run(stream)
