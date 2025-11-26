import xml.etree.ElementTree as ET
import os


def parse_pothole_xml(xml_path):
    """Parses a Pascal VOC XML file to extract pothole bounding boxes."""
    if not os.path.exists(xml_path):
        print(f"Error: File not found {xml_path}")
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []

    for obj in root.findall("object"):
        name = obj.find("name").text

        if name == "pothole":
            bndbox = obj.find("bndbox")

            # VOC is 1-based. Python is 0-based. We subtract 1.
            xmin = int(bndbox.find("xmin").text) - 1
            ymin = int(bndbox.find("ymin").text) - 1
            xmax = int(bndbox.find("xmax").text) - 1
            ymax = int(bndbox.find("ymax").text) - 1

            boxes.append([xmin, ymin, xmax, ymax])

    return boxes
