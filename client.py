import requests
import zipfile
import os
import sys

def zip(path: str):
    # convert all images to jpg and zip them
    with zipfile.ZipFile("data.zip", 'w') as zipObj:
        for f in os.listdir(path):
            zipObj.write(os.path.join(path, f))

def post(url: str, file_path: str):
    # post the zip file to the server
    with open(file_path, 'rb') as f:
        # set content type to application/json
        r = requests.post(url, files={'file': f})
        # save the response
        with open('response.zip', 'wb') as f:
            f.write(r.content)

if __name__ == '__main__':
    
    args = sys.argv[1:]

    if len(args) <= 1:
        print('Not enough arguments')
        print('Usage: python client.py <flag> <path_to_data>')
        exit(1)

    flag = args[0]
    path = args[1]

    if flag == '-z':
        zip_name = path

    elif flag == '-d':
        zip(path)
        zip_name = 'data.zip'

    else:
        print('Invalid flag')
        print('Usage: python client.py <flag> <path_to_data>')
        exit(1)

    post(url='http://127.0.0.1:5000/infer', file_path=zip_name)
