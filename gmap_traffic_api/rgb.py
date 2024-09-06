import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image_path = './gmap_traffic_api/screenshot.png'
img = Image.open(image_path)

# Convert the image to RGB mode if it's not already
img = img.convert("RGB")

# Function to display the image and capture mouse click
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        # Get the pixel value at the clicked point
        x, y = int(event.xdata), int(event.ydata)
        rgb_value = img.getpixel((x, y))
        print(f'Clicked at ({x}, {y}) - RGB: {rgb_value}')
        plt.close()  # Close the plot after clicking

# Display the image
fig, ax = plt.subplots()
ax.imshow(img)
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()


# green (22, 224, 152) (17, 214, 143)
# yellow (255, 207, 67) 66
# red (242, 78, 66)
# bold red (169, 39, 39)
