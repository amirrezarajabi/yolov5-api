import os, shutil
from os.path import basename
import time
from flask import Flask, jsonify, request, send_file
from werkzeug.utils import secure_filename
import zipfile

from run import prepare_model, run

TMP_ROUTE = 'tmp'
INPUT_ROUTE = 'input'
OUTPUT_ROUTE = 'output'

# Loading dog models
MODEL1, DEVICE1 = prepare_model(weights="./model_dataset/dog/best1.pt")
# Loading horse models
MODEL3, DEVICE3 = prepare_model(weights="./model_dataset/horse/best1.pt")


# Preparing to use the model by specifying the type of animal
DOGMODELS = [MODEL1]
DOGDEVICES = [DEVICE1]

HORSEMODELS = [MODEL3]
HORSEDEVICES = [DEVICE3]

ANIMALSMODELS = {
    "dog":DOGMODELS,
    "horse":HORSEMODELS
}

ANIMALSDEVICES = {
    "dog":DOGDEVICES,
    "horse":HORSEDEVICES
}

IMGSZ = {
    "dog":(512, 512),
    "horse":(1024, 1024)
}

def zipdir(name: str, path: str):
    # ziph is zipfile handle
    with zipfile.ZipFile(name, 'w') as zipObj:
      for root, dirs, files in os.walk(path):
          for file in files:
              zipObj.write(os.path.join(root, file))

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
    print('Got file!')
    animal = request.args.get('animal')
    print('Animal: ', animal)
    if file and file.filename != '' and file.filename.endswith('.zip'):
        try:
            secure_name = secure_filename(file.filename)
            input_zip = TMP_ROUTE + '/' + secure_name
            file.save(input_zip)

            with zipfile.ZipFile(input_zip, 'r') as zip_ref:
                zip_ref.extractall(INPUT_ROUTE)
            
            samples = os.listdir(INPUT_ROUTE + "/content/output/")
            
            for i, sample_name in enumerate(samples):
                tic = time.time()
                os.mkdir(f"./{OUTPUT_ROUTE}/{sample_name}")
                run(model=ANIMALSMODELS[animal], device=ANIMALSDEVICES[animal], save_directory=f"./{OUTPUT_ROUTE}/{sample_name}", \
                    source=f'{INPUT_ROUTE}/content/output/{sample_name}/*.jpg', imgsz=IMGSZ[animal])
                print(f"Calling run function for sample {i}:", time.time() - tic)

            output_zip = TMP_ROUTE + '/data.zip'
            zipdir(output_zip, OUTPUT_ROUTE)

            return send_file(output_zip, mimetype='application/zip')
        except Exception as e:
            print(e)
            return jsonify({'message': 'failure', 'error': 'something went wrong.'})
    
    elif file is None:
        return jsonify({'message': 'failure', 'error': 'file is required.'})
    
    elif file.filename == '':
        return jsonify({'message': 'failure', 'error': 'file name cannot be empty.'})
    
    elif not file.filename.endswith('.zip'):
        return jsonify({'message': 'failure', 'error': 'file must be a zip file.'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
