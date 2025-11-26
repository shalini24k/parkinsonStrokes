import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from PIL import Image
import os

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(layout="wide", page_title="PD Detection System")
IMG_SIZE = 128 

# --- MODEL DEFINITION ---
class PDDenseNet(nn.Module):
    def __init__(self, improved=False, num_classes=2):
        super(PDDenseNet, self).__init__()
        self.densenet = models.densenet121(weights='DEFAULT')
        self.features = self.densenet.features
        in_features = self.densenet.classifier.in_features
        if improved:
            self.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, num_classes))
        else:
            self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

# --- VISUALIZATION HOOKS ---
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# --- HELPER: LOAD METRICS ---
def get_metrics(model_path, test_path, improved):
    if not os.path.exists(model_path) or not os.path.exists(test_path):
        return None, 0, None, []

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        data = ImageFolder(test_path, transform)
        if len(data) == 0: return None, 0, None, []
        
        loader = DataLoader(data, batch_size=32, shuffle=False)
        model = PDDenseNet(improved=improved, num_classes=len(data.classes))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device); model.eval()
        
        preds, labels = [], []
        with torch.no_grad():
            for i, l in loader:
                i = i.to(device)
                o = model(i)
                _, p = torch.max(o, 1)
                preds.extend(p.cpu().numpy())
                labels.extend(l.numpy())
        return model, accuracy_score(labels, preds), confusion_matrix(labels, preds), data.classes
    except Exception as e:
        return None, 0, None, []

# --- MAIN UI ---
st.title("🧠 Parkinson's Disease Detection System")

# 1. DROPDOWN
dataset_choice = st.selectbox("Select Dataset to Analyze", ["Geometric Strokes", "Handwritten Words"])

# 2. PATHS
if dataset_choice == "Geometric Strokes":
    test_folder = "./dataset_strokes/test"
    norm_path = "model_strokes_normal.pth"
    imp_path = "model_strokes_improved.pth"
    warning_text = "⚠️ Mode: **Strokes**. Please upload Spiral/Wave images."
else:
    test_folder = "./dataset_words/dataset-word/test"
    norm_path = "model_words_normal.pth"
    imp_path = "model_words_improved.pth"
    warning_text = "⚠️ Mode: **Words**. Please upload Handwriting images."

# 3. METRICS
st.markdown("---")
st.header(f"📊 Analysis: {dataset_choice}")

col1, col2 = st.columns(2)
model_n, acc_n, cm_n, classes = get_metrics(norm_path, test_folder, False)
model_i, acc_i, cm_i, _ = get_metrics(imp_path, test_folder, True)

with col1:
    st.subheader("Standard DenseNet")
    if model_n:
        st.metric("Accuracy", f"{acc_n*100:.1f}%")
        if cm_n is not None:
            fig, ax = plt.subplots(figsize=(3,2))
            sns.heatmap(cm_n, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            st.pyplot(fig)
    else:
        st.warning("Train Normal Model first.")

with col2:
    st.subheader("Improved DenseNet")
    if model_i:
        st.metric("Accuracy", f"{acc_i*100:.1f}%", delta=f"{(acc_i - acc_n)*100:.1f}%" if model_n else None)
        if cm_i is not None:
            fig, ax = plt.subplots(figsize=(3,2))
            sns.heatmap(cm_i, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
            st.pyplot(fig)
    else:
        st.warning("Train Improved Model first.")

# 4. PREDICTION & HD VISUALIZATION
st.markdown("---")
st.header("🔍 Prediction & Layer Visualization")
st.info(warning_text)

up_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

if up_file and model_i:
    c_img, c_res = st.columns([1, 2])
    
    # Process
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Load original for overlay
    original_img = Image.open(up_file).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    tensor = transform(original_img).unsqueeze(0).to(device)
    
    with c_img:
        st.image(original_img, width=150, caption="Input Image")
        
    with c_res:
        out = model_i(tensor)
        prob = torch.nn.functional.softmax(out, dim=1)
        conf, pred = torch.max(prob, 1)
        res_label = classes[pred.item()]
        
        if res_label.lower() == "parkinson":
            st.error(f"Prediction: {res_label} ({conf.item()*100:.1f}%)")
        else:
            st.success(f"Prediction: {res_label} ({conf.item()*100:.1f}%)")

        # --- HOOKS FOR TWO LAYERS ---
        activation.clear()
        
        # 1. Early Layer (DenseBlock 1) - Detects Edges
        h1 = model_i.features.denseblock1.register_forward_hook(get_activation('layer1'))
        
        # 2. Deep Layer (DenseBlock 3) - Detects Patterns
        h2 = model_i.features.denseblock3.register_forward_hook(get_activation('layer3'))
        
        _ = model_i(tensor)
        h1.remove()
        h2.remove()
        
        # --- VISUALIZATION FUNCTION ---
        def plot_layer_hd(layer_name, title_text, col_map):
            st.write(f"### {title_text}")
            if layer_name in activation:
                fmaps = activation[layer_name][0].cpu()
                
                # Get Top 3 most active channels
                channel_means = fmaps.view(fmaps.size(0), -1).mean(dim=1)
                _, top_indices = torch.topk(channel_means, 3)
                
                cols = st.columns(3)
                for idx, channel_idx in enumerate(top_indices):
                    # Normalize
                    fmap = fmaps[channel_idx]
                    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())
                    
                    # Upscale
                    fmap = fmap.unsqueeze(0).unsqueeze(0)
                    fmap = torch.nn.functional.interpolate(fmap, size=(IMG_SIZE, IMG_SIZE), mode='bilinear')
                    fmap = fmap.squeeze().numpy()
                    
                    fig, ax = plt.subplots()
                    ax.imshow(original_img)
                    ax.imshow(fmap, cmap=col_map, alpha=0.5) 
                    ax.axis('off')
                    
                    with cols[idx]:
                        st.caption(f"Channel {channel_idx.item()}")
                        st.pyplot(fig)

        # Plot Layer 1 (Blue/Purple Heatmap for simple edges)
        plot_layer_hd('layer1', "Layer 1: Early Features (Edges & Lines)", 'viridis')
        
        # Plot Layer 3 (Red/Yellow Heatmap for complex patterns)
        plot_layer_hd('layer3', "Layer 3: Deep Features (Tremors & Shapes)", 'inferno')

        # Top 10 Neurons
        st.write("### 🔢 Top 10 Active Neurons (Final Feature Vector)")
        feats = model_i.features(tensor)
        feats = nn.functional.adaptive_avg_pool2d(nn.functional.relu(feats), (1, 1))
        vec = torch.flatten(feats, 1).detach().cpu().numpy()[0]
        idx = vec.argsort()[-10:][::-1]
        st.table({"Neuron Index": idx, "Activation Score": vec[idx]})