import cv2
import numpy as np
import os


def match_templates(image_path, templates_dir, threshold=0.8):
    """
    Perform template matching and output the match score for each template.

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
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

        # Find the maximum match score
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Store the result if the score exceeds the threshold
        if max_val >= threshold:
            results.append({
                "template": template_name,
                "match_score": max_val,
                "position": max_loc,
            })

            # Optionally, draw a rectangle around the match for debugging
            h, w = gray_template.shape
            cv2.rectangle(image, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)

    # Save the result image with rectangles for debugging
    cv2.imwrite("debug_matched_templates.png", image)

    return results


for z in range(1, 16):
    image_path = f"C:\_\solitaire_bot\\templates\card_mismatch_screenshots\game_screenshot_{z}.bmp"
    templates_directory = "C:\_\solitaire_bot\\templates\supposedly_perfect_matches"

    results = match_templates(image_path, templates_directory, threshold=0.95)
    print(f"\ngame_screenshot_{z}:")
    for result in results:
        print(f"Template: {result['template']}, Match Score: {result['match_score']}")
