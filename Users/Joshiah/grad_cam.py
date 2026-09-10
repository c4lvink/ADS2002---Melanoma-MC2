import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        # Turn the model in evaluation for faster computations (ex: dropout are turned off, batch normalization use the mean and std computer in training and not current batch)
        self.model = model.eval()
        self.featuremaps = []
        self.gradients = []

        target_layer.register_forward_hook(self.save_featuremaps)
        target_layer.register_backward_hook(self.save_gradients)

    def save_featuremaps(self, module, input, output):
        self.featuremaps.append(output)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def get_cam_weights(self, grads):
        # compute the neuron weights (it is the mean of the gradients)
        return np.mean(grads, axis=(1, 2))

    def __call__(self, image, label=None):
        preds = self.model(image)
        self.model.zero_grad()

        if label is None:
            label = preds.argmax(dim=1).item()

        # computing the gradients with backward
        preds[:, label].backward()

        featuremaps = self.featuremaps[-1].cpu().data.numpy()[0, :]
        gradients = self.gradients[-1].cpu().data.numpy()[0, :]

        weights = self.get_cam_weights(gradients)
        cam = np.zeros(featuremaps.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            # multiply the feature map by the neuron weight
            cam += w * featuremaps[i]

        #ReLU because we only want the positive contribution
        cam = np.maximum(cam, 0)
        
        # At this point the cam is a 7x7 matrix, we need to resize it to the original image size (224x224)
        cam = cv2.resize(cam, image.shape[-2:][::-1])
        
        # Normalize the cam to be plotted with a heatmap
        cam_normalized = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        
        return label, cam_normalized
