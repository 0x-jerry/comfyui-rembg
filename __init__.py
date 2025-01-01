import torch
from rembg import new_session, remove
import torchvision.transforms.v2 as T
from PIL import Image
import numpy as np
import folder_paths

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ImageRemoveBackgroundRembg:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        model_names = [x for x in folder_paths.get_filename_list("rembg")]
        return {
            "required": {
                "model_name": (model_names, ),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "remove_background"
    CATEGORY = "image"
    TITLE = "Image Remove Background (rembg)"


    def remove_background(self, image, model_name):
        model_path = folder_paths.get_full_path("rembg", model_name)
        
        session = new_session(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        image = image.permute([0, 3, 1, 2])
        output = []
        for img in image:
            img = T.ToPILImage()(img)
            img = remove(img, session=session)
            output.append(T.ToTensor()(img))

        output = torch.stack(output, dim=0)
        output = output.permute([0, 2, 3, 1])
        mask = output[:, :, :, 3] if output.shape[3] == 4 else torch.ones_like(output[:, :, :, 0])

        return(output, mask,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Image Remove Background (rembg)": ImageRemoveBackgroundRembg
}


