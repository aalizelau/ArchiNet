import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image


class ArchiNet:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.styles = ['Achaemenid architecture','American craftsman style','American Foursquare architecture','Ancient Egyptian architecture',
    'Art Deco architecture','Art Nouveau architecture','Baroque architecture','Bauhaus architecture','Beaux-Arts architecture',
    'Byzantine architecture','Chicago school architecture','Colonial architecture','Deconstructivism','Edwardian architecture',
    'Georgian architecture','Gothic architecture','Greek Revival architecture','International style','Novelty architecture',
    'Palladian architecture','Postmodern architecture','Queen Anne architecture','Romanesque architecture','Russian Revival architecture',
    'Tudor Revival architecture']
    
    def preprocess(self, image):
        img = image.convert("RGB")
        img = img.resize((384, 384))
        arr = np.asarray(img).astype(np.float32) / 255.0

         # Normalize using ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        arr = np.transpose(arr, (2, 0, 1)) 
        arr = np.expand_dims(arr, axis=0) 
        return arr
    
    def predict(self, image_path, top_k=1):
        input_array = self.preprocess(image_path)
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: input_array})
        logits = output[0][0]  # shape: (num_classes,)
        probs = np.exp(logits) / np.sum(np.exp(logits))  # softmax
        top_indices = np.argsort(probs)[::-1][:top_k]
        results = [(self.styles[i], probs[i]) for i in top_indices]
        return results if top_k > 1 else results[0]
    
model = ArchiNet("./model/archinet.onnx")

def classify_image(image):
    results = model.predict(image, top_k=3)
    return {style: prob for style, prob in results}

# Build Gradio Interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="🏛️ Architectural Style Classifier",
    description="Upload an image of a building, and this AI model will identify the most likely architectural style, from classical to modern.",
    flagging_mode="never",
    examples=[
        ["./example_imgs/01.jpg"],
        ["./example_imgs/02.jpg"],
        ["./example_imgs/03.jpg"],
        ["./example_imgs/04.jpg"],
    ]
)

# Launch locally
if __name__ == "__main__":
    demo.launch()
