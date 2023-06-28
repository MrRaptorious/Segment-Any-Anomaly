import os
import matplotlib.pyplot as plt
import SAA as SegmentAnyAnomaly
from utils.training_utils import *
from tqdm import tqdm
import cv2
import numpy as np
import glob


dino_config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
dino_checkpoint = 'weights/groundingdino_swint_ogc.pth'
sam_checkpoint = 'weights/sam_vit_h_4b8939.pth'
box_threshold = 0.1
text_threshold = 0.1
eval_resolution = 1024
device = f"cuda:0"
root_dir = 'result'

# get the model
model = SegmentAnyAnomaly.Model(
    dino_config_file=dino_config_file,
    dino_checkpoint=dino_checkpoint,
    sam_checkpoint=sam_checkpoint,
    box_threshold=box_threshold,
    text_threshold=text_threshold,
    out_size=eval_resolution,
    device=device,
)

model = model.to(device)

def process_image(heatmap, image):
    heatmap = heatmap.astype(float)
    heatmap = (heatmap - heatmap.min()) / heatmap.max() * 255
    heatmap = heatmap.astype(np.uint8)
    heat_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    visz_map = cv2.addWeighted(heat_map, 0.5, image, 0.5, 0)
    visz_map = cv2.cvtColor(visz_map, cv2.COLOR_BGR2RGB)

    visz_map = visz_map.astype(float)
    visz_map = visz_map / visz_map.max()
    return visz_map


def inference(image, anomaly_description, object_name, object_number, mask_number, area_threashold):
    textual_prompts = [
        [anomaly_description, object_name]
    ]  # detect prompts, filtered phrase
    property_text_prompts = f'the image of {object_name} have {object_number} dissimilar {object_name}, with a maximum of {mask_number} anomaly. The anomaly would not exceed {area_threashold} object area. '

    model.set_ensemble_text_prompts(textual_prompts, verbose=True)
    model.set_property_text_prompts(property_text_prompts, verbose=True)

    image = cv2.resize(image, (eval_resolution, eval_resolution))
    score, appendix = model(image)
    similarity_map = appendix['similarity_map']

    image_show = cv2.resize(image, (eval_resolution, eval_resolution))
    similarity_map = cv2.resize(similarity_map, (eval_resolution, eval_resolution))
    score = cv2.resize(score, (eval_resolution, eval_resolution))

    viz_score = process_image(score, image_show)
    viz_sim = process_image(similarity_map, image_show)

    return viz_score, viz_sim


def main():
    multiple_images()

def multiple_images():

    # get all images
    # All files and directories ending with .txt and that don't begin with a dot:
    images = glob.glob("/workspaces/Segment-Any-Anomaly/data/<datasetname>/bad/*.png")
    result_dir = "/workspaces/Segment-Any-Anomaly/result/<datasetname>"

    # parameter
    image = None
    anomaly_description = None
    object_name = None
    object_number = None
    mask_number = None
    area_threashold = None

    # arguments
    anomaly_description = ''
    object_name = ''
    object_number = 6
    mask_number = 1
    area_threashold = 0.3

    for image_path in tqdm(images):
        image = cv2.imread(image_path)

        # calc anomaly
        score, sim = inference(image,anomaly_description,object_name,object_number, mask_number, area_threashold)

        # create one single image
        result_image = np.concatenate((score, sim), axis=1) * 255
        
        # save to disk
        file_name = os.path.basename(image_path)
        result_path = os.path.join(result_dir,file_name)
        cv2.imwrite(result_path, result_image, [cv2.IMWRITE_PNG_COMPRESSION, 8])



def single_image():
    # parameter
    image = None
    anomaly_description = None
    object_name = None
    object_number = None
    mask_number = None
    area_threashold = None

    # arguments
    image = cv2.imread('<path_to_file>')
    anomaly_description = ''
    object_name = ''
    object_number = 6
    mask_number = 1
    area_threashold = 0.3

    # calc anomaly
    score, sim = inference(image,anomaly_description,object_name,object_number, mask_number, area_threashold)

    # show images
    result_image = np.concatenate((score, sim), axis=1)
    cv2.imshow("result", result_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
