"""
mascxml_to_txt.py — MASC XML to raw text converter
---------------------------------
Input:
 - A folder containing MASC XML files.

Output:
    - A folder containing text files, one for each XML file.

Usage:
    python mascxml_to_txt.py <folder_path>


"""

import argparse
import os
import xml.etree.ElementTree as ET

def process_xml_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".xml"):
            xml_path = os.path.join(folder_path, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(folder_path, "txt", txt_filename)
            if not os.path.exists(os.path.dirname(txt_path)):
                os.makedirs(os.path.dirname(txt_path))
            
            with open(txt_path, "w", encoding="utf-8") as f:
                for segment in root.findall(".//segment"):
                    text = segment.find("text").text
                    f.write(text + "\n")
                    
    return len([filename for filename in os.listdir(folder_path) if filename.lower().endswith(".xml")])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process XML files in a folder and create corresponding text files.")
    parser.add_argument('folder_path', type=str, help='Path to the folder containing XML files.')
    args = parser.parse_args()
    num_transformed_files = process_xml_files(args.folder_path)
    print(f"{num_transformed_files} XML files have been transformed.")
    if num_transformed_files == 0:
        raise ValueError("No XML files found in the specified folder.")