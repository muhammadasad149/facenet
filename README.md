# Face Embeddings with FaceNet PyTorch

This repository contains Python code for extracting facial embeddings from images using the FaceNet PyTorch library. Facial embeddings are numerical representations of faces that can be used for various tasks such as face recognition and verification.

## Getting Started

### Prerequisites

Before you begin, you need to install the required libraries:

- [facenet_pytorch](https://github.com/timesler/facenet-pytorch)
- [torch](https://pytorch.org/)
- [torchvision](https://pytorch.org/)
- [PIL (Pillow)](https://pillow.readthedocs.io/en/stable/)

You can install these libraries using `pip`:

```bash
pip install facenet-pytorch torch torchvision pillow
```

### Usage

1. Clone this repository:

```bash
git clone https://github.com/muhammadasad149/facenet
```

2. Import the necessary modules in your Python script:

```python
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
```

3. Create an instance of the MTCNN (Multi-task Cascaded Convolutional Networks) detector for face detection:

```python
mtcnn = MTCNN()
```

4. Load an image and use MTCNN to detect faces and return bounding boxes:

```python
image = Image.open("path_to_image.jpg")
boxes, _ = mtcnn.detect(image)
```

5. Create an instance of the Inception Resnet model for face embedding:

```python
resnet = InceptionResnetV1(pretrained='vggface2').eval()
```

6. For each detected face, extract its embedding:

```python
for box in boxes:
    face = mtcnn.extract(image, box)
    face_embedding = resnet(face.unsqueeze(0))
    print(face_embedding)
```

Make sure to replace `"path_to_image.jpg"` with the path to your image file. This code will detect faces in the image and extract their embeddings.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FaceNet PyTorch](https://github.com/timesler/facenet-pytorch)
- [PyTorch](https://pytorch.org/)

Feel free to update this README with additional information or customize it according to your project's specific needs.
