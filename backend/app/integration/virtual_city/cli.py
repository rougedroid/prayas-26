import argparse, json
from pathlib import Path
from .config import VCConfig
from .integration import export_optimizer_result, publish_optimizer_result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--output-dir", default="artifacts/virtual_city")
    p.add_argument("--publish", action="store_true")
    p.add_argument("--datasource-name", default="Prayas Optimized City Plan")
    a = p.parse_args()
    result = json.loads(Path(a.input).read_text(encoding="utf-8"))

    if a.publish:
        out = publish_optimizer_result(
            result, VCConfig.from_env(), a.output_dir, a.datasource_name
        )
        print(json.dumps(out, indent=2))
    else:
        print(export_optimizer_result(result, a.output_dir))

if __name__ == "__main__":
    main()
