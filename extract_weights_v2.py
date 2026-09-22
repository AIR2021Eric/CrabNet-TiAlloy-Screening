"""extract_weights_v2.py - extract the 2021-02-23 (v2) mat2vec weights INTO STAGING ONLY.
Never touches models/trained_models or the v1 backup."""
import os, glob, shutil, py7zr

ARCHIVE_CANDIDATES = [
    "trained_models_2021-02-23 0230h mat2vec (bsz_auto).7z",
    os.path.join("archives", "trained_models_2021-02-23 0230h mat2vec (bsz_auto).7z"),
]
archive_name = next((p for p in ARCHIVE_CANDIDATES if os.path.exists(p)), None)
if archive_name is None:
    raise FileNotFoundError("v2 archive not found in project root or archives/ - adjust path.")

STAGING = os.path.join("models", "v2_staging")
os.makedirs(STAGING, exist_ok=True)

print(f"Reading archive {archive_name} ...")
with py7zr.SevenZipFile(archive_name, mode="r") as z:
    names = z.getnames()
    weight_files = [n for n in names if n.endswith(".pth") or n.endswith(".tar")]
    print(f"Found {len(weight_files)} weight members:")
    for f in weight_files:
        print(f" - {f}")
    z.extract(path=STAGING, targets=weight_files)   # extract INTO staging, not "."

# Flatten ONLY inside staging (never walk the project root!)
root_abs = os.path.abspath(STAGING)
moved = 0
for sub in glob.glob(os.path.join(STAGING, "**", "*.pth"), recursive=True):
    if os.path.abspath(os.path.dirname(sub)) != root_abs:
        shutil.move(sub, os.path.join(STAGING, os.path.basename(sub)))
        moved += 1
print(f"\nFlattened {moved} nested files into {STAGING}.")

n = len(glob.glob(os.path.join(STAGING, "*.pth")))
print(f"Staging now contains {n} .pth files (expect same count as v1, ~83-84).")
print("\nNEXT STEPS:")
print("  python inspect_keys.py models/v2_staging/elasticity_log10(K_VRH)0.pth   # expect UNPATCHED")
print("  python patch_weights_dir.py models/v2_staging")
print("  python inspect_keys.py models/v2_staging/elasticity_log10(K_VRH)0.pth   # expect PATCHED")