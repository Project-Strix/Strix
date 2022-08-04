import click
import pytest
from pathlib import Path
from functools import partial
from strix.utilities.click_callbacks import NumericChoice as Choice, freeze_option
from strix.utilities.click import OptionEx, CommandEx

option = partial(click.option, cls=OptionEx)
command = partial(click.command, cls=CommandEx)

def test_prompt_proj1(runner):
    filepath = "/homes/clwang"

    @command()
    @option("--project", prompt=True, type=click.Path(), default=Path.cwd(), help="Project folder path")
    def cli_prompt_case(project):
        print(project)

    result = runner.invoke(cli_prompt_case, ["--project", filepath], input="\n")
    assert filepath in result.output

def test_prompt_proj2(runner):
    @command()
    @option("--project", prompt=True, type=click.Path(), default=Path.cwd(), help="Project folder path")
    def cli_prompt_case2(project):
        pass

    result = runner.invoke(cli_prompt_case2, input="\n")
    print('result:', result.output)
    assert str(Path.cwd()) in result.output
