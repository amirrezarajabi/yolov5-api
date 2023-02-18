from PIL import Image
import os
import sys

input_dir = sys.argv[1] if len(sys.argv) > 1 else './data/'
output_dir = sys.argv[2] if len(sys.argv) > 2 else './new_data/'

print(f'input_dir: {input_dir}')
print(f'output_dir: {output_dir}')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):

    if not filename.endswith(".tif"):
        continue

    new_filename = filename[:-4] + '.jpg'

    old_name = os.path.join(input_dir, filename)
    im = Image.open(old_name)

    new_name = os.path.join(output_dir, new_filename)
    im.save(new_name)
    
    print(f'{os.path.join(input_dir, filename)} -> {new_name}')
