from pascal_voc_pytorch.prediction_pipeline  import PascalVOCPredictionPipeline

P = PascalVOCPredictionPipeline(
    checkpoint = 'checkpoints/model.pt',
    device = 'cuda'
)

results = P.predict_from_filename(
    filename = 'images/horse.jpg',
    topk = 3
)

print(results)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (10 , 5))

ax[0].imshow(plt.imread('images/horse.jpg')), ax[0].axis('off')
ax[1].bar(results['classnames'] ,results['logits'])

fig.savefig('out.jpg')