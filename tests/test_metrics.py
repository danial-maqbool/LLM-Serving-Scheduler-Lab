from llm_scheduler_lab.metrics import percentile


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([1.0], 0.99) == 1.0
