import pytest
from click import Context, Command
from strix.utilities.click_callbacks import get_unknown_options

input_args_1 = (["--output", "1", "--debug"], {"output": 1.0})
input_args_2 = (["--output", "1", "--debug", "--mask", "label"], {"output": 1.0, "debug": True, "mask": "label"})
input_args_3 = (["--value", "-1", '-D', 'aaa'], {"value": -1.0, "D": "aaa"})

@pytest.mark.parametrize('args, expected', [input_args_1, input_args_2, input_args_3])
def test_parse_additional_options(args, expected):
    ctx = Context(command=Command('test'))
    ctx.args = args

    params = get_unknown_options(ctx, verbose=True)
    assert params == expected
