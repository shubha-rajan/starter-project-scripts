import argparse
import io
import glob
import os
import csv

from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw


def rename_files(label):
    files = glob.glob(f"{label}/*.jpg")
    for fileno, filename in enumerate(files):
        new_name = f"{label}/{label}{fileno}.jpg"
        if new_name in files:
            print(
                f"Failed to rename {filename} to {new_name}.",
                "File already exists")
        else:
            os.rename(filename, new_name)

def generate_csv(labels, bucket_name, n, rename=False):
    if rename:
        for label in labels:
            rename_files(label)

    with open('output.csv', mode='w') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        for label in labels:
            files = glob.glob(f"{label}/*.jpg")
            for index, path in enumerate(files):
                vertices = crop_iterations(path, n)
                if index % 8 == 0:
                    writer.writerow(["VALIDATE",
                                    f"gs://{bucket_name}/{path}",
                                    label])
                elif index % 4 == 0:
                    writer.writerow(["TEST", 
                                    f"gs://{bucket_name}/{path}",
                                    label])
                else: 
                    writer.writerow(["TRAIN",
                                    f"gs://{bucket_name}/{path}",
                                    label])
                    
# get_crop_hint and crop_to_hint were adapted from https://cloud.google.com/vision/docs/crop-hints

def get_crop_hint(path):
    """Detect crop hints on a single image and return the first result."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    crop_hints_params = types.CropHintsParams()
    image_context = types.ImageContext(crop_hints_params=crop_hints_params)

    response = client.crop_hints(image=image, image_context=image_context)
    hints = response.crop_hints_annotation.crop_hints

    vertices = hints[0].bounding_poly.vertices

    return vertices

def crop_to_hint(image_file):
    """Crop the image using the hints in the vector list."""
    vects = get_crop_hint(image_file)

    im = Image.open(image_file)
    im2 = im.crop([vects[0].x, vects[0].y,
                  vects[2].x - 1, vects[2].y - 1])
    im2.save(image_file, 'JPEG')
    print(f'Saved new image to {image_file}')

def crop_iterations(path, n):
    iterations = 0
    while iterations < n:
        crop_to_hint(path)
        iterations += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bucket_name', help='the name of the GCS Bucket')
    parser.add_argument('--labels', nargs='+', help='the labels for the dataset')
    parser.add_argument('--crop_iterations', help='the number of times to run crophints on each image')
    parser.add_argument('--rename', default=False,  help='whether or not to rename files')
    args = parser.parse_args()

    generate_csv(args.labels, args.bucket_name, int(args.crop_iterations), args.rename)
