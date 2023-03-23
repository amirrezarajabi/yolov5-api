import os, shutil
from os.path import basename
import time
from flask import Flask, jsonify, request, send_file
from werkzeug.utils import secure_filename
import zipfile

from run import prepare_model, run

TMP_ROUTE = 'tmp'
INPUT_ROUTE = 'data2'
OUTPUT_ROUTE = 'output'
MODEL1, DEVICE1 = prepare_model(weights="./model_dataset/dog/best1.pt")
MODEL2, DEVICE2 = prepare_model(weights="./model_dataset/dog/best2.pt")

def zipdir(path, ziph):
    # ziph is zipfile handle
    for folderName, subfolders, filenames in os.walk(path):
       for filename in filenames:
           #create complete filepath of file in directory
           filePath = os.path.join(folderName, filename)
           # Add file to zip
           ziph.write(filePath, basename(filePath))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024 * 1024

@app.before_request
def cleanup():
    for folder in [INPUT_ROUTE, OUTPUT_ROUTE, TMP_ROUTE]:
        shutil.rmtree(folder)
        os.mkdir(folder)
        
@app.route('/infer', methods=['GET'])
def wrong_method():
    return jsonify({'message': 'please use post method!'})

@app.route('/infer', methods=['POST'])
def infer():

    file = request.files.get('file')
    if file and file.filename != '' and file.filename.endswith('.zip'):
        try:
            secure_name = secure_filename(file.filename)
            input_zip = TMP_ROUTE + '/' + secure_name
            file.save(input_zip)

            with zipfile.ZipFile(input_zip, 'r') as zip_ref:
                zip_ref.extractall(INPUT_ROUTE)
            
            tic = time.time()
            run(model=MODEL1, device=DEVICE1, save_directory=f"./{OUTPUT_ROUTE}/", source=INPUT_ROUTE+"/content/drive/MyDrive/output_client/*.jpg", postfix_img_="_1")
            run(model=MODEL2, device=DEVICE2, save_directory=f"./{OUTPUT_ROUTE}/", source=INPUT_ROUTE+"/content/drive/MyDrive/output_client/*.jpg", postfix_img_="_2")
            print("Calling run function: ", time.time() - tic)

            output_zip = TMP_ROUTE + '/data.zip'
            with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipdir(OUTPUT_ROUTE, zipf)  

            return send_file(output_zip, mimetype='application/zip')
        except:
            return jsonify({'message': 'failure', 'error': 'something went wrong.'})
    
    elif file is None:
        return jsonify({'message': 'failure', 'error': 'file is required.'})
    
    elif file.filename == '':
        return jsonify({'message': 'failure', 'error': 'file name cannot be empty.'})
    
    elif not file.filename.endswith('.zip'):
        return jsonify({'message': 'failure', 'error': 'file must be a zip file.'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
