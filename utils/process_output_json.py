import json


def process_errors(js_path="output/4440f2a3598d9533247a1a11c54aadae_errors-7999.json"):
    with open(js_path) as f:
        js = json.load(f)

    for (k, v) in js.items():
        if "relativeSE" in k:
            print(k, np.mean(v), np.std(v))
