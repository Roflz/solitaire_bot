import cv2


def find_best_scale(template, screenshot, scales, threshold=0.8):
    """
    Dynamically adjust the template scale to match cards in the screenshot.

    Args:
        template (numpy.ndarray): The template image.
        screenshot (numpy.ndarray): The screenshot image.
        scales (list): List of scaling factors to test.
        threshold (float): Matching confidence threshold.

    Returns:
        float: Best scale value.
        numpy.ndarray: Best resized template.
    """
    best_scale = None
    best_resized_template = None
    best_score = 0

    for scale in scales:
        resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(screenshot, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_scale = scale
            best_resized_template = resized_template

    return best_scale, best_resized_template

template = "templates/cards/clover_10.png"  # template to find best scale for
screenshot = "game_screenshot.bmp"  # screenshot to gauge template against


scales = [0.5, 0.75, 1.0, 1.25, 1.5]  # Test different scaling factors
best_scale, best_template = find_best_scale(template, screenshot, scales)
print(f"Best scale: {best_scale}")