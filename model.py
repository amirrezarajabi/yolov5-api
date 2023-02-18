from tqdm import tqdm
import numpy as np
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from typing import Any, BinaryIO, List, Optional, Tuple, Union
from PIL import Image, ImageColor, ImageDraw, ImageFont
import warnings

COLORS = ['blue', 'green', 'red', 'cyan', 'yellow', 'black', 'black']

DOG_LABEL_NAMES = ['dog_lungworm', 'dog_Peitschenwurm', 'dog_Hakenwurm', 'dog_Kokzidien', 'dog_Spulwurm', 'dog_kleiner Leberegel'] 

DIVIDE_BY = 4

def generate_box(obj):
    box = obj.get('bbox')
    xmin = float(box.get('left') // DIVIDE_BY)
    ymin = float(box.get('top') // DIVIDE_BY)
    xmax = xmin + float(box.get('width') // DIVIDE_BY)
    ymax = ymin + float(box.get('height') // DIVIDE_BY)
    
    return [xmin, ymin, xmax, ymax]

def generate_label(obj):

    if obj.get('title') in DOG_LABEL_NAMES:
      return DOG_LABEL_NAMES.index(obj.get('title'))
    
    print('[generate_label] auxiliary function mislabel.')
    return 0

def generate_target(objects): 
    num_objs = len(objects)

    boxes = []
    labels = []
    for i in objects:
        boxes.append(generate_box(i))
        labels.append(generate_label(i))

    boxes = torch.as_tensor(boxes, dtype=torch.float32) 
    labels = torch.as_tensor(labels, dtype=torch.int64) 
    
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    
    return target

class MaskDataset(Dataset):
    
    def __init__(self, images, annotations, test=False):
        self.images = [np.array(image)[::DIVIDE_BY, ::DIVIDE_BY, :] for image in images]
        self.annotations = annotations
        self.test = test
        self.transform = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        target = {}

        if self.annotations:
            file_annot = self.annotations[idx]
            target = generate_target(file_annot)
        
        to_tensor = torchvision.transforms.ToTensor()

        if not self.test:
            transformed = self.transform(image=img, bboxes=target['boxes'], class_labels=target['labels'])
            img = transformed['image']
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32) 
            target['labels'] = torch.as_tensor(transformed['class_labels'], dtype=torch.int64) 

        # change to tensor
        img = to_tensor(img)

        return img, target
    

def collate_fn(batch):
    return tuple(zip(*batch))

def make_prediction(model, img, threshold):
    preds = model(img)
    for i in range(len(preds)):
        idx_list = []
        for idx, score in enumerate(preds[i]['scores']) :
            if score > threshold : #select idx which meets the threshold
                idx_list.append(idx)
        preds[i]['boxes'] = preds[i]['boxes'][idx_list]
        preds[i]['labels'] = preds[i]['labels'][idx_list]
        preds[i]['scores'] = preds[i]['scores'][idx_list]
    return preds

def draw_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
) -> torch.Tensor:

    """
    Draws bounding boxes on given image.
    The values of the input image should be uint8 between 0 and 255.
    If fill is True, Resulting Tensor should be saved as PNG image.

    Args:
        image (Tensor): Tensor of shape (C x H x W) and dtype uint8.
        boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
            the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
            `0 <= ymin < ymax < H`.
        labels (List[str]): List containing the labels of bounding boxes.
        colors (color or list of colors, optional): List containing the colors
            of the boxes or single color for all boxes. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for boxes.
        fill (bool): If `True` fills the bounding box with specified color.
        width (int): Width of bounding box.
        font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
            also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
            `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size (int): The requested font size in points.

    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with bounding boxes plotted.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size(0) not in {1, 3}:
        raise ValueError("Only grayscale and RGB images are supported")
    elif (boxes[:, 0] > boxes[:, 2]).any() or (boxes[:, 1] > boxes[:, 3]).any():
        raise ValueError(
            "Boxes need to be in (xmin, ymin, xmax, ymax) format. Use torchvision.ops.box_convert to convert them"
        )

    num_boxes = boxes.shape[0]

    if num_boxes == 0:
        warnings.warn("boxes doesn't contain any box. No box was drawn")
        return image

    if labels is None:
        labels: Union[List[str], List[None]] = [None] * num_boxes  # type: ignore[no-redef]
    elif len(labels) != num_boxes:
        raise ValueError(
            f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
        )

    if colors is None:
        colors = COLORS
    elif isinstance(colors, list):
        if len(colors) < num_boxes:
            raise ValueError(f"Number of colors ({len(colors)}) is less than number of boxes ({num_boxes}). ")
    else:  # colors specifies a single color for all boxes
        colors = [colors] * num_boxes

    colors = [(ImageColor.getrgb(color) if isinstance(color, str) else color) for color in colors]

    if font is None:
        if font_size is not None:
            warnings.warn("Argument 'font_size' will be ignored since 'font' is not set.")
        txt_font = ImageFont.load_default()
    else:
        txt_font = ImageFont.truetype(font=font, size=font_size or 20)


    # Handle Grayscale images
    if image.size(0) == 1:
        image = torch.tile(image, (3, 1, 1))

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    img_boxes = boxes.to(torch.int64).tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")
    else:
        draw = ImageDraw.Draw(img_to_draw)

    for bbox, color, label in zip(img_boxes, colors, labels):  # type: ignore[arg-type]
        if fill:
            fill_color = color + (100,)
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if label is not None:
            margin = width + 1
            draw.text((bbox[0] + margin, bbox[3] + margin), label, fill=color, font=txt_font)

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)

retina = None
device = None

def init():
    global retina, device

    retina = torchvision.models.detection.retinanet_resnet50_fpn(num_classes = 6, weights=None, weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

    if torch.cuda.is_available():
        torch.rand(1).cuda()

    retina.load_state_dict(torch.load(f'./retina_30.pt'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    retina.to(device) 
    retina.eval()
    retina(torch.rand(1, 3, 2514, 2456).to(device))

def run(input_dir='data', output_dir='output', batch_size = 16):
    global retina, device
    if retina is None or device is None:
        print('the model is not initialized')
        return

    dir_path = input_dir + '/content/drive/MyDrive/output_client/'
    dir_files = os.listdir(dir_path)
    image_files = [dir for dir in dir_files if dir.endswith('.jpg')]

    arr1 = []
    for i, image in enumerate(image_files):
        img = Image.open(dir_path + image)
        arr1.append(img)
    
    test_dataset = MaskDataset(arr1, None, test=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    del arr1
    
    preds_adj_all = []

    for im, annot in test_data_loader:
        im = list(img.to(device) for img in im)
        with torch.no_grad():
            preds_adj = make_prediction(retina, im, 0.5)
            preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
            preds_adj_all.append(preds_adj)

    del test_dataset
    del test_data_loader

    non_empty_preds = []

    for batch_i, pred_batch in enumerate(preds_adj_all):
        for sample_i, pred in enumerate(pred_batch):
            if len(pred['scores']) > 0:
                non_empty_preds.append({**pred, "bi": batch_i, "si": sample_i})

    non_empty_preds.sort(key=lambda pred: -pred['scores'].max())
    non_empty_preds = non_empty_preds[:10]

    del preds_adj_all

    arr2 = []

    for pred in non_empty_preds:
        bi, si = pred['bi'], pred['si']
        index = bi * batch_size + si
        img = Image.open(dir_path + image_files[index])
        img = np.array(img)
        arr2.append(img)

    to_tensor = torchvision.transforms.ToTensor()
    NEW_DIVIDE = 2

    for i, pred in enumerate(non_empty_preds):
        boxes, labels, scores = pred['boxes'], pred['labels'], pred['scores']
        colors = [COLORS[j] for j in labels]

        labels = ['{:s}: {:.2f}'.format(DOG_LABEL_NAMES[l], s) for s, l in zip(scores, labels)]

        boxes = boxes.mul(DIVIDE_BY / NEW_DIVIDE)

        img = arr2[i][::NEW_DIVIDE, ::NEW_DIVIDE, :]
        img = to_tensor(img)
        img = img.mul(255).byte()

        img = draw_bounding_boxes(img, boxes, width=3,
                            colors=colors,
                            labels=labels,
                            )
    
        img = torchvision.transforms.ToPILImage()(img)
        img.save(f'./{output_dir}/{i}.jpg')

# if __name__ == "__main__":
#     init()
#     run()
