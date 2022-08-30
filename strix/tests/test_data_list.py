import pytest
import json 
from strix.utilities.utils import parse_datalist

labeled_content = [
    {"image": "./test1.nii", "label": "./lab1.nii"},
    {"image": "./test2.nii", "label": "./lab2.nii"},
    {"image": "./test3.nii", "label": "./lab3.nii"},
]

unlabeled_content = [
    {"image": "./test4.nii"},
    {"image": "./test5.nii"},
]

def test_datalist_dict_json(tmp_path):
    content = {"labeled": labeled_content, "unlabeled": unlabeled_content}
    data_file = tmp_path/"datalist.json"
    with open(data_file, 'w') as f:
        json.dump(content, f, indent=2)

    labeled, unlabeled = parse_datalist(data_file, has_unlabel=True)
    assert len(labeled) == 3
    assert len(unlabeled) == 2

    labeled = parse_datalist(data_file, has_unlabel=False)
    assert len(labeled) == 3

    content = {"labeled": labeled_content}
    data_file2 = tmp_path/"datalist2.json"
    with open(data_file2, 'w') as f:
        json.dump(content, f, indent=2)
    
    with pytest.raises(Exception):
        parse_datalist.__wrapped__(data_file2, has_unlabel=True)


def test_datalist_list_json(tmp_path):
    data_file = tmp_path/"datalist.json"
    with open(data_file, 'w') as f:
        json.dump(labeled_content, f, indent=2)

    labeled_list = parse_datalist(data_file, has_unlabel=True)
    assert len(labeled_list) == 3