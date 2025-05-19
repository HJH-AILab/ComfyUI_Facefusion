import os
import folder_paths

from facefusion import state_manager

def use_comfyui_path():
    import facefusion
    import facefusion.filesystem as filesystem

    def resolve_relative_path_for_comfyui(path):
        models_root = folder_paths.get_folder_paths("facefusion")[0]
        return os.path.join(models_root, path.replace("../.assets/models/", ""))
    filesystem.resolve_relative_path = resolve_relative_path_for_comfyui

    def list_directory_for_comfyui(directory_path):
        directory_path = f"{facefusion.ROOT_DIR}/{directory_path}"
        if filesystem.is_directory(directory_path):
            file_paths = sorted(os.listdir(directory_path))
            files = []

            for file_path in file_paths:
                file_name, file_extension = os.path.splitext(file_path)

                if not file_name.startswith(('.', '__')):
                    files.append(
                    {
                        'name': file_name,
                        'extension': file_extension,
                        'path': os.path.join(directory_path, file_path)
                    })
            return files
        return None
    filesystem.list_directory = list_directory_for_comfyui


    def sanitize_path_for_windows_for_comfyui(path):
        return path
    filesystem.sanitize_path_for_windows = sanitize_path_for_windows_for_comfyui

def skip_hash_check():
    import facefusion.download as download
    original_download_hashes = download.conditional_download_hashes
    def download_hashes_for_comfyui(hash_set):
        if state_manager.get_item("skip_hash_check"):
            return True
        return original_download_hashes(hash_set)
    download.conditional_download_hashes = download_hashes_for_comfyui

    original_validate_source_paths = download.validate_source_paths
    def validate_source_paths_for_comfyui(source_set):
        if state_manager.get_item("skip_hash_check"):
            return source_set,[]
        return original_validate_source_paths(source_set)
    download.validate_source_paths = validate_source_paths_for_comfyui

def override_defs_for_comfyui():
    use_comfyui_path()
    skip_hash_check()
    
