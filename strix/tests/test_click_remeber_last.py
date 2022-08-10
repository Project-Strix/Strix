import click
import pytest
from functools import partial
from strix.utilities.click_callbacks import NumericChoice as Choice, dump_params
from strix.utilities.click import OptionEx, CommandEx
from strix.utilities.utils import get_items

option = partial(click.option, cls=OptionEx)
command = partial(click.command, cls=CommandEx)

def test_prompt_remember_last_choice(runner, tmp_path):
    @command()
    @option(
        "-g", type=Choice(["none", "day", "week", "month"]), prompt=True
    )
    def cli_normal_case(g):
        pass

    out_path = tmp_path / 'history.json'

    @command()
    @option("-d", is_flag=True, default=True)
    @option("-p", prompt=False, type=str, default="yes")
    @option("-g", type=Choice(["none", "day", "week", "month"]), default='day', prompt=True, show_default=True)
    @option("-dump", type=bool, default=True, callback=partial(dump_params, output_path=out_path))
    def cli_remeber_case(d, p, g):
        pass

    result = runner.invoke(cli_normal_case, [], input="none")
    assert "(1: none, 2: day, 3: week, 4: month)" in result.output

    result = runner.invoke(cli_remeber_case, [], input="week")
    print(result.output)
    assert "(1: none, 2: day, 3: week, 4: month) [day]:" in result.output

    dumped_options = (get_items(out_path, allow_filenotfound=True))
    assert "d" not in dumped_options and "p" not in dumped_options

@pytest.mark.parametrize('prompt', [True, False])
def test_remeber_options(prompt, runner, tmp_path):
    @command(context_settings={
        "prompt_in_default_map": prompt,
        "default_map": {"g": "day"}
    })
    @option("-g", type=Choice(["none", "day", "week", "month"]), default='day', prompt=True, show_default=True)
    def cli_remeber_case2(g):
        pass

    result = runner.invoke(cli_remeber_case2, [], input="none")
    print(result.output)
    if prompt:
        assert "(1: none, 2: day, 3: week, 4: month) [day]: none" in result.output
    else:
        assert len(result.output) == 0
