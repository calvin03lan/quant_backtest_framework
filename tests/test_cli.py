import pytest

from quant_backtest.cli import build_parser


def test_load_data_accepts_explicit_codes():
    args = build_parser().parse_args(
        [
            "load-data",
            "--codes",
            "SPY.US",
            "510300.SH",
            "--start",
            "2020-01-01",
        ]
    )

    assert args.codes == ["SPY.US", "510300.SH"]
    assert args.start == "2020-01-01"


def test_load_data_rejects_codes_with_pool():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["load-data", "--codes", "SPY.US", "--pool", "sample"]
        )
