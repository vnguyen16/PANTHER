import subprocess
from pathlib import Path
from paquo.projects import QuPathProject

QUPATH = r"C:\Users\Vivian\paquo_qupath\QuPath-0.5.1\QuPath-0.5.1 (console).exe"
PROJECT = Path(r"C:\Users\Vivian\Documents\qupath_projects\project.qpproj")
# GROOVY  = Path(r"C:\Users\Vivian\Documents\qupath_ann_testing\scripts\tile_and_detect_cells_run2.groovy")
GROOVY  = Path(r"C:\Users\Vivian\Documents\qupath_ann_testing\scripts\tile_and_detect_cells_run2.groovy")

with QuPathProject(str(PROJECT), mode="r") as proj:
    images = list(proj.images)

for idx, entry in enumerate(images):
    print(f"\n=== Processing {idx}: {entry.image_name} ===")

    cmd = [
        QUPATH, "script",
        str(GROOVY),
        "--project", str(PROJECT),
        "--image", str(idx)
    ]

    result = subprocess.run(cmd)

    print("Return code:", result.returncode)
    if result.returncode != 0:
        print("Error occurred. Stopping.")
        break
