import cv2
import numpy as np
import os


def match_templates(image_path, templates_dir, threshold=0.8):
    """
    Perform template matching and output cropped images showing the matched region for each template.

    Args:
        image_path (str): Path to the image in which we want to find the templates.
        templates_dir (str): Directory containing the template images.
        threshold (float): Threshold to filter the match score (default is 0.8).

    Returns:
        List[Dict]: List of templates with their match score.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # List to store results
    results = []

    # Loop through each template in the directory
    for template_name in os.listdir(templates_dir):
        template_path = os.path.join(templates_dir, template_name)
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)

        # Find the maximum match score
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Store the result if the score exceeds the threshold
        if max_val >= threshold:
            results.append({
                "template": template_name,
                "match_score": max_val,
                "position": max_loc,
            })

            # Get the dimensions of the template
            h, w = gray_template.shape

            # Crop the matched area from the original image
            cropped_image = image[max_loc[1]:max_loc[1] + h, max_loc[0]:max_loc[0] + w]

            # Save the cropped image
            output_image_path = f"C:\_\solitaire_bot\\templates\supposedly_perfect_matches_test\{template_name}"
            cv2.imwrite(output_image_path, cropped_image)
            print(f"Saved cropped match for {template_name} at {output_image_path}")

    return results


image_path = "C:\_\solitaire_bot\\templates\\test_screenshots\game_screenshot_4.bmp"
templates_directory = "C:\_\solitaire_bot\\templates\cards_higher_res"

results = match_templates(image_path, templates_directory, threshold=0.97)
for result in results:
    print(f"Template: {result['template']}, Match Score: {result['match_score']}")
