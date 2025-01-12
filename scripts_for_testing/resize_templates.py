import cv2
import os

# Directory containing the full card templates
input_dir = "../templates/ranks"  # Replace with your directory path
output_dir = "..\\templates\\cards_resized"  # Directory to save partial templates
fx = 81/62  # x scalar
fy = 114/90  # y scalar


# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each image file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # Load the full template
        file_path = os.path.join(input_dir, filename)
        template = cv2.imread(file_path)
        template = cv2.resize(template, None, fx=fx, fy=fy,
                              interpolation=cv2.INTER_AREA)  # Scale template to game resolution

        if template is None:
            print(f"Skipping {filename}: Unable to load image.")
            continue

        # Save the cropped template
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
        cv2.imwrite(output_path, template)
        print(f"Saved resized template: {output_path}")

print("All resized templates saved successfully!")
