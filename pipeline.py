from transformers import ViTForImageClassification
import torch
from datasets import load_dataset # for cifar10 
from transformers import ViTImageProcessor # for tokenizing images
from hookfn import Probe
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


# for gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ViTForImageClassification.from_pretrained('vit_cifar10_finetuned')
model.to(device)

# load cifar10 (only small portion for demonstration purposes) 
train_ds, test_ds = load_dataset('cifar10', split=['train[:5000]', 'test[:2000]'])
# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label:id for id,label in id2label.items()}

print(f'id2label: {id2label}')


processor = ViTImageProcessor.from_pretrained("vit_cifar10_finetuned")
inputs = processor(images=test_ds[:]['img'], return_tensors="pt").to(device)

layer_names = [name for name, _ in model.named_modules() if "intermediate_act_fn" in name]
hook_manager = Probe(model)
hook_manager.register_hook(layers_=layer_names)
# hook_manager.register_hook() -> for all layers


# perform forward pass

with torch.no_grad():
  outputs = model(inputs.pixel_values)
logits = outputs.logits
logits.shape

y_true = test_ds[:]['label']
y_pred = outputs.logits.argmax(1).cpu().numpy()

labels = [i for i in label2id.keys()]

# # confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
# disp.plot(xticks_rotation=45)


# activations 
print(layer_names)
activations = hook_manager.layer_outs
print(f"Number of activations: {len(activations.keys())}")
for i in activations.keys():
    print(i, activations[i].shape)
    
    

## some viz
from act_viz import ActViz
viz = ActViz(id2label)
viz.visualize_activation(activations['vit.encoder.layer.0.intermediate.intermediate_act_fn'], y_true, id2label, method='PCA')
viz.visualize_activation(activations['vit.encoder.layer.0.intermediate.intermediate_act_fn'], y_true, id2label, method='T-SNE')
viz.plot_all(activations, y_true, id2label)


act_list = np.array([ activations[i].detach().cpu().numpy() for i in activations.keys()]) # shape: (num_layers, batch_size, seq_len, hidden_dim)
pcs_ = []
for i in range(act_list.shape[0]):
    pcs_.append(act_list[i][:, 0, :]) ## mean across sequence length
pcs_ = np.array(pcs_)
print(pcs_.shape)

# performing persistence homology
from ph import PersistenceHomology
ph = PersistenceHomology(y_true, y_pred, id2label)
dgms_full, dgms_sub = ph.ripser_persistence(pcs_) ## without PCA
dgms_full_pca, dgms_sub_pca = ph.ripser_persistence(pcs_=pcs_, dim=2, PCA=True) ## with PCA


# checking the bottleneck distance
from utils import compute_dist
for i in range(len(pcs_)):
  for j in range(i+1, len(pcs_)):
        print(f"(full-rank) Bottleneck distance between  {id2label[y_true[i]]} and {id2label[y_true[j]]}: {compute_dist(pcs_[i], pcs_[j], y_true, [i,j], id2label, dgms_full[i], dgms_full[j])}")
        print(f"(PCA) Bottleneck distance between  {id2label[y_true[i]]} and {id2label[y_true[j]]}: {compute_dist(pcs_[i], pcs_[j], y_true, [i,j], id2label, dgms_full_pca[i], dgms_full_pca[j])}")
        print(f"(full-rank - subsampled pts) Bottleneck distance between  {id2label[y_true[i]]} and {id2label[y_true[j]]} : {compute_dist(pcs_[i], pcs_[j], y_true, [i,j], id2label, dgms_sub[i], dgms_sub[j])}")
        print(f"(PCA - subsampled pts) Bottleneck distance between  {id2label[y_true[i]]} and {id2label[y_true[j]]} : {compute_dist(pcs_[i], pcs_[j], y_true, [i,j], id2label, dgms_sub_pca[i], dgms_sub_pca[j])}")
        
# checking for connected components 
## go thru this, use the mst processor to get the number of connected components
### https://persim.scikit-tda.org/en/latest/_modules/persim/visuals.html#plot_diagrams