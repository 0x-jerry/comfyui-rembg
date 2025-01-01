import torch
import torchvision.transforms.v2 as T
import folder_paths
from rembg import new_session, remove


class LoadRembgModelNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        model_names = [x for x in folder_paths.get_filename_list("rembg")]

        return {
            "required": {
                "model_name": (model_names,),
            },
        }

    RETURN_TYPES = ("REMBG",)
    FUNCTION = "process"
    CATEGORY = "Rembg"
    TITLE = "Load Rembg Model"

    def process(self, model_name):
        model_path = folder_paths.get_full_path("rembg", model_name)

        session = new_session(model_path, providers=["CPUExecutionProvider"])

        return (session,)


class RemoveImageBackgroundNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("REMBG",),
                "image": ("IMAGE",),
            },
            "optional": {
                "alpha_matting": ("BOOLEAN",),
                "alpha_matting_foreground_threshold": (
                    "INT",
                    {"default": 240, "step": 1},
                ),
                "alpha_matting_background_threshold": (
                    "INT",
                    {"default": 10, "step": 1},
                ),
                "alpha_matting_erode_size": ("INT", {"default": 10, "step": 1}),
                "post_process_mask": ("BOOLEAN",),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    FUNCTION = "process"
    CATEGORY = "Rembg"
    TITLE = "Image Remove Background"

    def process(
        self,
        image,
        model,
        *args,
        **kwargs,
    ):
        session = model

        image = image.permute([0, 3, 1, 2])
        output = []
        to_image = T.ToPILImage()
        to_dtype = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
        for img in image:
            img = to_image(img)
            img = remove(
                img,
                session=session,
                *args,
                **kwargs,
            )
            img = to_dtype(img)
            output.append(img)

        output = torch.stack(output, dim=0)
        output = output.permute([0, 2, 3, 1])
        mask = (
            output[:, :, :, 3]
            if output.shape[3] == 4
            else torch.ones_like(output[:, :, :, 0])
        )

        return (
            output,
            mask,
        )


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "Load Rembg Model": LoadRembgModelNode,
    "Rembg Remove background": RemoveImageBackgroundNode,
}
