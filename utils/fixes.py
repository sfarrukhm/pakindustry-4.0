import sys
import yaml
from pathlib import Path

########################
# CV FIXES
########################

def fix_roboflow_yaml(dataset_dir: str):
    """
    Fixes Roboflow-exported data.yaml by removing redundant dataset_dir prefix.
    Example:
      train: defect-1/train/images  ->  train/images
    """
    dataset_dir = Path(dataset_dir)
    yaml_path = dataset_dir / "data.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"❌ data.yaml not found in {dataset_dir}")

    # Load yaml
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Fix train/val/test paths
    for key in ["train", "val", "test"]:
        if key in data:
            p = Path(str(data[key]))
            if p.parts and p.parts[0] == dataset_dir.name:
                # Strip redundant dataset folder name
                new_path = Path(*p.parts[1:])
                data[key] = str(new_path)

    # Save updated yaml
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✅ Fixed {yaml_path}")
    for key in ["train", "val", "test"]:
        if key in data:
            print(f"  {key}: {data[key]}")


########################
# PREDICTIVE MAINTENANCE FIXES (example placeholder)
########################

def fix_sensor_csv(file_path: str):
    pass


########################
# FORECASTING FIXES (example placeholder)
########################

def fix_timeseries_index(file_path: str):
    pass


########################
# COMMAND-LINE DISPATCH
########################

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fixes.py <fix_name> <args...>")
        print("Available fixes: fix_roboflow_yaml, fix_sensor_csv, fix_timeseries_index")
        sys.exit(1)

    fix_name = sys.argv[1]
    args = sys.argv[2:]

    if fix_name == "fix_roboflow_yaml":
        fix_roboflow_yaml(*args)
    elif fix_name == "fix_sensor_csv":
        fix_sensor_csv(*args)
    elif fix_name == "fix_timeseries_index":
        fix_timeseries_index(*args)
    else:
        print(f"❌ Unknown fix: {fix_name}")
