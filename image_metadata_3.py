import exifread
from PIL import Image

def extract_metadata(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)

    metadata = {}

    # Get image size
    metadata['image_size'] = f"{Image.open(image_path).size[0]}x{Image.open(image_path).size[1]}"

    # Get image height
    metadata['image_height'] = Image.open(image_path).size[1]

    # Get image width
    metadata['image_width'] = Image.open(image_path).size[0]

    # Get image location
    if 'GPSLatitude' in tags and 'GPSLongitude' in tags:
        latitude = exifread.gps_to_decimal(tags['GPSLatitude'].values)
        longitude = exifread.gps_to_decimal(tags['GPSLongitude'].values)
        metadata['image_location'] = f"{latitude}, {longitude}"
        metadata['image_location_link'] = f"https://www.google.com/maps/place/{latitude},{longitude}"
    elif 'GPSDestLatitude' in tags and 'GPSDestLongitude' in tags:
        latitude = exifread.gps_to_decimal(tags['GPSDestLatitude'].values)
        longitude = exifread.gps_to_decimal(tags['GPSDestLongitude'].values)
        metadata['image_location'] = f"{latitude}, {longitude}"
        metadata['image_location_link'] = f"https://www.google.com/maps/place/{latitude},{longitude}"
    else:
        metadata['image_location'] = "Unknown"
        metadata['image_location_link'] = ""

    # Get image date
    if 'DateTimeOriginal' in tags:
        metadata['image_date'] = tags['DateTimeOriginal'].values
    else:
        metadata['image_date'] = "Unknown"

    return metadata

# Example usage
image_path = "your_image.jpg"
metadata = extract_metadata(image_path)
print(metadata)
