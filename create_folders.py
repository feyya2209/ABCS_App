import os

# 1. Where are your original 70+ breed folders located?
original_dir = r"C:\Users\USER\Documents\FYP AUGMENT & CLASSIFY IMAGES\ABCS_App\dataset_original\train"

# 2. Where do you want the new EMPTY folders to be created?
augmented_dir = r"C:\Users\USER\Documents\FYP AUGMENT & CLASSIFY IMAGES\ABCS_App\For training results page\2_Augmented"
combined_dir = r"C:\Users\USER\Documents\FYP AUGMENT & CLASSIFY IMAGES\ABCS_App\For training results page\3_Combined"

# 3. Get the exact names of all the original folders
print("Scanning original dataset...")
breeds = [f for f in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, f))]

# 4. Generate the empty folders instantly
for breed in breeds:
    # Create the Augmented folder
    os.makedirs(os.path.join(augmented_dir, breed), exist_ok=True)
    # Create the Combined folder
    os.makedirs(os.path.join(combined_dir, breed), exist_ok=True)

print(f"✅ Success! Created {len(breeds)} empty folders inside '2_Augmented_Only'.")
print(f"✅ Success! Created {len(breeds)} empty folders inside '3_Combined_Final'.")