import os
from tqdm import tqdm
from pdf2image import convert_from_path

root_path = './search_engine/corpus'
pdf_path = os.path.join(root_path,'pdf')
pdf_files = [file for file in os.listdir(pdf_path) if file.endswith('pdf')]

output_path = os.path.join(root_path, 'img_pdf')
if not os.path.exists(output_path):
    os.makedirs(output_path)

for filename in tqdm(pdf_files):
    filepath = os.path.join(pdf_path,filename)
    imgname = filename.split('.pdf')[0]
    images = convert_from_path(filepath)
    for i, image in enumerate(images):
        idx = i + 1
        image.save(os.path.join(output_path, f'{imgname}_{idx}.jpg'), 'JPEG')