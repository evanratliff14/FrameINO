# import inference code
from preprocess.SAM3D.SAM3DObjects.notebook.inference import Inference, load_image, load_single_mask


def reconstruct(image, mask):
    # load model
    tag = "hf"
    config_path = f"SAM3DObjects/checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    # load image and mask

    # run model
    output = inference(image, mask, seed=42)

    # export gaussian splat
    mesh = output["glb"]
    return mesh