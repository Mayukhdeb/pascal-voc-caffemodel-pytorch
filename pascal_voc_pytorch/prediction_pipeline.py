import torch 
import torch.nn as nn 
import cv2 

from .model import PascalVoCModel


class PascalVOCPredictionPipeline(nn.Module):
    def __init__(self,device, checkpoint, model = PascalVoCModel):
        super().__init__()

        self.model = model()
        state_dict = torch.load(checkpoint)
        self.model.load_state_dict(state_dict )
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        print('loaded model from: ' + checkpoint)

        self.classes =  [
                        "aeroplane",
                        "bicycle",
                        "bird",
                        "boat",
                        "bottle",
                        "bus",
                        "car",
                        "cat",
                        "chair",
                        "cow",
                        "diningtable",
                        "dog",
                        "horse",
                        "motorbike",
                        "person",
                        "pottedplant",
                        "sheep",
                        "sofa",
                        "train",
                        "tvmonitor"
                    ]

    def load_image_from_filename(self, filename, size = (227, 227)):

        im_original = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        im_original = cv2.resize(im_original, size)
        image_tensor = torch.tensor(im_original).permute(-1,0,1).unsqueeze(0).float()
        
        return image_tensor.to(self.device)


    def forward(self, x):

        pred = self.model(x)

        return pred

    def predict_from_filename(self, filename = 'test.jpg', parse_preds = True, topk = 3):

        image_tensor = self.load_image_from_filename(filename = filename)

        with torch.no_grad():
            pred = self.forward(image_tensor)

        if parse_preds == False:
            return pred.cpu()

        else:
            values, indices = torch.topk(pred[0], k = topk)

            classnames = [self.classes[i] for i in indices]
            logits  = [values[i].item() for i in range(len(values))]

            return {
                'classnames': classnames,
                'logits': logits
            }
