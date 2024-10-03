import requests
from memory_profiler import profile

# Path to the image file you want to upload
image_path = "images/PXL_20240423_224710650.jpg"

def test_api():
    with open(image_path, 'rb') as image_file:
        files = {'file': image_file}  # The key 'file' must match the FastAPI endpoint's argument

        # Send the POST request
        response = requests.post("http://127.0.0.1:8000/api/v1/predict_markers/", files=files)
        print(response.json())

        image_file.seek(0)
        files = {'file': image_file}

        response = requests.post("http://127.0.0.1:8000/api/v1/predict_pgc/", files=files)
        print(response.json())


test_api()

requests.get("http://127.0.0.1:8000/api/v1/status")
# print(response.json())
# Print the response from the server
# print(response.json())