IMAGE = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff"]
VIDEO = [".mp4", ".mov", ".avi", ".flv", ".wmv", ".mkv"]
AUDIO = [".mp3", ".wav", ".flac", ".aac"]

def check_type(filename, type):
    EXTS = []
    if type == "image":
        EXTS = IMAGE
    elif type == "video":
        EXTS = VIDEO
    elif type == "audio":
        EXTS = AUDIO
    else:
        return []
    
    return any(filename.endswith(ext) for ext in EXTS)      
