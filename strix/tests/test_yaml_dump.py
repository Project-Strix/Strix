from pathlib import Path
import strix.utilities.oyaml as yaml

def test_yaml_dump(tmp_path):
    content = Path.cwd()
    with open(tmp_path / "dumping.yml", "w") as f:
        yaml.dump(content, f, sort_keys=True)

    with open(tmp_path / "dumping.yml") as f:
        a = f.readline().replace("\n", "")
        assert a == str(Path.cwd())
