import os
import py7zr
import shutil

archive_name = "trained_models_2020-12-04 1138h mat2vec (bsz_auto).7z"
dest_folder = "models/trained_models"
os.makedirs(dest_folder, exist_ok=True)

print(f"Reading archive {archive_name}...")
with py7zr.SevenZipFile(archive_name, mode='r') as z:
    all_files = z.getnames()
    
    # Find ALL files that end in .pth or .tar
    weight_files = [f for f in all_files if f.endswith('.pth') or f.endswith('.tar')]
    
    print(f"Found {len(weight_files)} weight files. Extracting all of them...")
    for f in weight_files:
        print(f" - {f}")
        
    # Extract them to the current directory
    z.extract(path=".", targets=weight_files)

# Move them into the correct models/trained_models folder
print("\nOrganizing files into models/trained_models/ ...")
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith('.pth') or file.endswith('.tar'):
            src = os.path.join(root, file)
            dst = os.path.join(dest_folder, file)
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.move(src, dst)

print("\nDone! All weights are now inside models/trained_models/")
print("Please check that folder to see the exact filenames!")