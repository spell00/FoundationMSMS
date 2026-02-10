import numpy as np

from build_foundation_lcms import _bin, make_rt_windows


def test_bin_floor():
    assert _bin(10.0, 10.0, 1.0) == 0
    assert _bin(10.9, 10.0, 1.0) == 0
    assert _bin(11.0, 10.0, 1.0) == 1
    assert _bin(12.1, 10.0, 1.0) == 2


def test_make_rt_windows_creates_files(tmp_path):
    coords = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [0, 0, 4],
        ],
        dtype=np.int32,
    )
    vals = np.ones((5,), dtype=np.float32)
    npz_path = tmp_path / "sample.npz"
    np.savez_compressed(npz_path, coords=coords, vals=vals)

    make_rt_windows(tmp_path, window_sec=2, stride_sec=2)

    outputs = sorted(tmp_path.glob("sample__t*.npz"))
    assert len(outputs) == 3

    data0 = np.load(outputs[0])
    data1 = np.load(outputs[1])
    data2 = np.load(outputs[2])

    assert data0["coords"].shape[0] == 2
    assert data1["coords"].shape[0] == 2
    assert data2["coords"].shape[0] == 1
