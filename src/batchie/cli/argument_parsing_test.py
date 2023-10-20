from batchie.cli import argument_parsing


def test_cast_dict_to_type():
    result = argument_parsing.cast_dict_to_type(
        {"a": "1", "b": "false"},
        {"a": int, "b": bool},
    )

    assert result == {"a": 1, "b": False}
