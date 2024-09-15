from pathlib import Path
from dotenv import load_dotenv
import os


def find_project_root(current_path: Path = Path(__file__).resolve()) -> Path:
    """프로젝트 루트 디렉토리를 찾는 함수"""
    while not (current_path / ".git").exists():
        parent = current_path.parent
        if parent == current_path:
            raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다.")
        current_path = parent
    return current_path


def load_env_vars():
    """환경 변수를 로드하는 함수"""
    project_root = find_project_root()
    env_path = project_root / ".env"
    load_dotenv(env_path)

    if "PROJECT_ROOT" not in os.environ:
        os.environ["PROJECT_ROOT"] = str(project_root)


# 환경 변수 로드
load_env_vars()
