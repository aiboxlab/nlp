from pathlib import Path

def get(resource: str) -> Path:
    path = Path(__file__).parent / resource
    if path.exists() and path.is_dir():
        return path
    path.mkdir()
    return path

