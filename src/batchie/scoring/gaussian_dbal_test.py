from batchie.scoring import gaussian_dbal


def test_iter_combination():
    results = []
    for i in range(10):
        results.append(gaussian_dbal.get_combination_at_sorted_index(i, 5, 2))

    expected = [
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
    ]

    assert results == expected
