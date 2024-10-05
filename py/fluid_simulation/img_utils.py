from PIL import Image
import os


def images_to_gif(png_dir):
    # Create a list of PNG files in the directory, sorted by name
    png_files = sorted([f for f in os.listdir(png_dir) if f.endswith(".png")])

    # Open all images and append them to a list
    images = [Image.open(os.path.join(png_dir, f)) for f in png_files]

    # Save the first image as a GIF, appending the rest of the images
    gif_path = os.path.join(png_dir, "output.gif")
    images[0].save(
        gif_path,
        save_all=True,  # Save all frames
        append_images=images[1:],  # The rest of the images
        duration=1,  # Duration between frames in milliseconds
        loop=0,  # 0 means infinite loop
    )

    print(f"GIF saved at {gif_path}")


if __name__ == "__main__":
    png_dir = "../assets/cpp_screenshots/pressure_free"
    images_to_gif(png_dir=png_dir)
