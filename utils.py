import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import torch

class VideoWriter:
    def __init__(self, filename, fps=30, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)
    
    def add(self, img):
        img = img / 2 + 0.5
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img = img.numpy().transpose(1, 2, 0)
        
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        
        self.writer.write_frame(img)
    
    def close(self):
        if self.writer:
            self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *kw):
        self.close()


def imshow(img):
    img = img / 2 + 0.5
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_coords(res):
    mgrid = torch.meshgrid([torch.linspace(-1, 1, steps=res)] * 2)
    return torch.stack(mgrid, dim=-1).reshape(-1, 2)