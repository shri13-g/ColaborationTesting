import yt_dlp
import os
print(os.getcwd())
ydl_opts = {
    'format': 'mp4',  # specify format
    'outtmpl': 'downloaded_video3.%(ext)s',  # output filename template
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=LYigiwbaX_U&t=3222s'])

