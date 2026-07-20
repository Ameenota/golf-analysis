from pathlib import Path

from src.pro_preprocessing import artifact_paths


def test_artifact_paths_preserve_original_filename_stem(tmp_path):
    csv_path, json_path = artifact_paths(
        "data/benchmark/pro_videos/pro_14_greg_norman_face-on.mp4", tmp_path
    )

    assert csv_path == tmp_path / "pro_14_greg_norman_face-on.csv"
    assert json_path == tmp_path / "pro_14_greg_norman_face-on.json"


def test_artifact_paths_do_not_depend_on_source_directory(tmp_path):
    first = artifact_paths(Path("one/pro_47_tiger_woods_down-the-line.mp4"), tmp_path)
    second = artifact_paths(Path("two/pro_47_tiger_woods_down-the-line.mov"), tmp_path)

    assert first == second
