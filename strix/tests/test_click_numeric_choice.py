import click
import pytest
from strix.utilities.click_callbacks import NumericChoice as Choice


def test_choices_list_in_prompt(runner, monkeypatch):
    @click.command()
    @click.option(
        "-g", type=Choice(["none", "day", "week", "month"]), prompt=True
    )
    def cli_with_choices(g):
        pass

    @click.command()
    @click.option(
        "-g", type=Choice(["none", "day", "week", "month"]), prompt=True, show_choices=False
    )
    def cli_without_choices(g):
        pass

    result = runner.invoke(cli_with_choices, [], input="none")
    assert "(1: none, 2: day, 3: week, 4: month)" in result.output

    result = runner.invoke(cli_without_choices, [], input="none")
    print(result.output)
    assert "(1: none, 2: day, 3: week, 4: month)" not in result.output