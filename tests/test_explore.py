from src.data.ingest import clean
from src.explore import run

from .test_ingest import raw_frame  # noqa: F401


def test_eda_writes_all_outputs(raw_frame, tmp_path):  # noqa: F811
    df, _ = clean(raw_frame)
    files = run(df, tmp_path)
    assert len(files) == 6
    for f in files:
        assert f.exists() and f.stat().st_size > 0
