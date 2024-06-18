import os
import requests
import time

# Define API endpoint URL
api_url = 'http://192.168.10.238:10006/process_image'

# Define folder path containing images
folder_path = '/home/hera/workspace/unilm/kosmos-2.5/kosmos_test'

# Define headers
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

# Iterate over files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Construct file path
        file_path = os.path.join(folder_path, filename)

        # Define payload data (JSON body)
        payload = {
            "image": file_path,
            "do_ocr": True,
            "do_md": False
        }

        # Send POST request
        response = requests.post(api_url, json=payload, headers=headers)

        time.sleep(2)

        # Handle response
        if response.status_code == 200:
            # Construct output file path (same location with .txt extension)
            output_file_path = os.path.join(folder_path, os.path.splitext(filename)[0] + '.txt')

            # Write response JSON to file
            with open(output_file_path, 'w') as file:
                file.write(response.text)

            print(f"Processed {file_path}. Response saved to {output_file_path}")
        else:
            print(f"Failed to process {file_path}. Status code: {response.status_code}")
