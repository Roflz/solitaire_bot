import cv2


def upscale_image(image, scale_factor=2):
    """
    Upscale the image using interpolation to improve resolution.

    Args:
        image (numpy.ndarray): The input image to upscale.
        scale_factor (int): Factor by which to upscale the image.

    Returns:
        numpy.ndarray: The upscaled image.
    """
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    upscaled_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    return upscaled_image


screenshot_path = 'C:\_\solitaire_bot\\templates\cards\spades_4.png'

screenshot = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
upscaled_screenshot = upscale_image(screenshot, scale_factor=2)

# Show the upscaled screenshot
cv2.imwrite("spades_4_upscaled.png", screenshot)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
