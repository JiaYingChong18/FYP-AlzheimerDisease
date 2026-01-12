from turtle import pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import zipfile
import io
import re
import json
from PIL import Image
from torchvision import models, transforms
import pandas as pd
import pandas as pd
import altair as alt
import streamlit as st
from io import BytesIO
from matplotlib import gridspec          # Fixes the 'gridspec' NameError
import matplotlib.patches as patches
from huggingface_hub import hf_hub_download
# =============================================================================
# 1. CONFIGURATION
# =============================================================================
st.set_page_config(page_title="AD-Identify: Multimodal XAI", layout="wide")

CLASS_NAMES = ['Non Demented', 'Dementia']
PREPROCESSING_REMOVE_DARK_SLICES = 20
TARGET_DEPTH = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSIFICATION_THRESHOLD = 0.35

def run_occlusion_st(model, feature_volume, target_class):
    model.eval()
    feature_volume = feature_volume.to(DEVICE)

    with torch.no_grad():
        baseline_prob = F.softmax(
            model(feature_volume.unsqueeze(0)), dim=1
        )[0, target_class].item()

    scores = []
    for s in range(TARGET_DEPTH):
        occluded = feature_volume.clone()
        occluded[:, s, :, :] = 0
        with torch.no_grad():
            prob = F.softmax(
                model(occluded.unsqueeze(0)), dim=1
            )[0, target_class].item()
        scores.append(baseline_prob - prob)

    return np.array(scores)


class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class MedicalNetResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = [BasicBlock3D(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

# ============================================================================
# EXPLAINABLE AI: GRAD-CAM
# ============================================================================

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output): 
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_in, grad_out): 
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, class_idx].backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
        cam = F.relu(torch.sum(weights * self.activations, dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode='trilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# ============================================================================
# EXPLAINABLE AI: OCCLUSION SENSITIVITY
# ============================================================================

def run_occlusion(model, x, class_idx):
    """Analyze importance of each slice through occlusion"""
    model.eval()
    baseline_prob = F.softmax(model(x), dim=1)[0, class_idx].item()
    impacts = []
    
    for s in range(x.shape[2]):
        occluded = x.clone()
        occluded[:, :, s, :, :] = 0
        with torch.no_grad():
            prob = F.softmax(model(occluded), dim=1)[0, class_idx].item()
            impacts.append(baseline_prob - prob)
    
    return np.array(impacts)

# ============================================================================
# DATA PREPROCESSING (Exact match with training)
# ============================================================================

def extract_patient_id(filename):
    """Extract patient ID from OASIS-1 filename"""
    match = re.search(r'OAS1_(\d+)_', filename)
    return int(match.group(1)) if match else None

def extract_mpr_and_slice(filename):
    """Extract MPR number and slice number for sorting"""
    mpr = re.search(r'mpr-(\d+)', filename, re.I)
    slc = re.search(r'_(\d+)\.(jpg|png)', filename, re.I)
    return int(mpr.group(1)) if mpr else 0, int(slc.group(1)) if slc else 0

def load_slice_as_grayscale(img_data, target_size=224):
    """Load slice as grayscale"""
    img = Image.open(BytesIO(img_data)).convert('L')
    img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

def process_zip_file(zip_file, target_depth=32, target_size=224):
    all_files = []
    
    with zipfile.ZipFile(zip_file, 'r') as zf:
        for filename in zf.namelist():
            # Match Kaggle's regex and file extension support
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Use the exact Kaggle regex to find slice number
                match = re.search(r'_(\d+)\.(jpg|png|jpeg)', filename, re.I)
                if match:
                    slice_num = int(match.group(1))
                    img_data = zf.read(filename)
                    all_files.append((img_data, slice_num))
    
    if not all_files:
        raise ValueError("No valid image files found in ZIP")
    
    all_files.sort(key=lambda x: x[1])
    
    # LOADING: Exact Kaggle method (resize then divide by 255.0)
    slices = []
    for img_data, _ in all_files:
        img = Image.open(BytesIO(img_data)).convert('L')
        # Kaggle uses default PIL resize; Streamlit was using BILINEAR
        img = img.resize((target_size, target_size)) 
        slices.append(np.array(img) / 255.0)
        
    volume = np.stack(slices, axis=0)
    
    # DEPTH FOCUS: (Matches Kaggle 0.15 to 0.85)
    if len(volume) > target_depth:
        start_idx = int(len(volume) * 0.15)
        end_idx = int(len(volume) * 0.85)
        volume = volume[start_idx:end_idx]
    
    # LINEAR INTERPOLATION: (Matches Kaggle)
    if len(volume) != target_depth:
        indices = np.linspace(0, len(volume) - 1, target_depth)
        indices_floor = np.floor(indices).astype(int)
        indices_ceil = np.ceil(indices).astype(int)
        weights = indices - indices_floor
        
        volume_resized = []
        for f, c, w in zip(indices_floor, indices_ceil, weights):
            if f == c:
                volume_resized.append(volume[f])
            else:
                interpolated = volume[f] * (1 - w) + volume[c] * w
                volume_resized.append(interpolated)
        volume = np.stack(volume_resized, axis=0)
    
    # NORMALIZATION: (Matches Kaggle volume.mean() / volume.std())
    volume = (volume - volume.mean()) / (volume.std() + 1e-8)
    
    return torch.FloatTensor(volume).unsqueeze(0).unsqueeze(0)

# ============================================================================
# ROI INTERPRETATION
# ============================================================================

def get_roi_interpretation(heatmap, pred_idx):
    """Analyze heatmap activation to identify clinical regions of interest"""
    if pred_idx == 0: 
        return "Analysis: Consistent with normal brain structure."

    # ROI coordinates for hippocampal region (224x224)
    roi_y_start, roi_y_end = 110, 160
    roi_x_start, roi_x_end = 70, 150

    # Find peak activation
    y_peak, x_peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    # Check if peak is within hippocampal ROI
    is_in_hippocampus = (roi_y_start <= y_peak <= roi_y_end and 
                         roi_x_start <= x_peak <= roi_x_end)

    if is_in_hippocampus:
        return "⚠️ ROI Finding: High Attention in Medial Temporal Lobe (Hippocampus). High correlation with Alzheimer's biomarkers."
    else:
        return "AI Insight: Attention focused on broader structural changes (e.g., ventricular enlargement)."

# ============================================================================
# VISUALIZATION
# ============================================================================
def create_xai_visualization(input_tensor, model, pred_idx, prob, class_names, threshold=CLASSIFICATION_THRESHOLD):
    """Create comprehensive XAI visualization with fixed Y-axis scaling"""
    
    # Generate Grad-CAM
    gcam = GradCAM3D(model, model.layer4)
    heatmap_3d = gcam(input_tensor, pred_idx)
    
    # Generate Occlusion Sensitivity
    occ_scores = run_occlusion(model, input_tensor, pred_idx)
    
    # Find statistics for dynamic scaling
    top_slice = np.argmax(occ_scores)
    max_impact = np.max(occ_scores)
    min_impact = np.min(occ_scores)
    impact_range = max_impact - min_impact if (max_impact - min_impact) > 0 else 0.1
    
    # Prepare data (rotate 90° left for proper anatomical view)
    slice_data = input_tensor[0, 0, top_slice].cpu().numpy()
    heatmap_data = heatmap_3d[top_slice]
    mri_rotated = np.rot90(slice_data, k=1)
    heatmap_rotated = np.rot90(heatmap_data, k=1)
    
    # Create figure
    plt.style.use('dark_background')
    ROTATED_ASPECT = 2.0
    
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.4, 1], width_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])  # Anatomical View
    ax2 = fig.add_subplot(gs[0, 1])  # Grad-CAM
    ax3 = fig.add_subplot(gs[1, :])  # Depth Profile
    
    # --- Row 1, Left: Anatomical View ---
    ax1.imshow(mri_rotated, cmap='gray', aspect=ROTATED_ASPECT)
    ax1.set_title(f"Prediction: {class_names[pred_idx]} ({prob:.1f}%)\nSlice {top_slice} (Anatomical Proportions)", 
                  pad=15, fontsize=12)
    ax1.axis('off')
    
    # --- Row 1, Right: Grad-CAM with ROI ---
    ax2.imshow(mri_rotated, cmap='gray', aspect=ROTATED_ASPECT)
    ax2.imshow(heatmap_rotated, cmap='jet', alpha=0.45, aspect=ROTATED_ASPECT)
    
    # ROI box for rotated view
    roi_box_rot = patches.Rectangle((110, 73), 50, 80, linewidth=2, 
                                    edgecolor='#00ffcc', facecolor='none', 
                                    linestyle='--', label='Hippocampal ROI')
    ax2.add_patch(roi_box_rot)
    ax2.set_title("Grad-CAM XAI\n(ROI Spatial Attention)", pad=15, fontsize=12)
    ax2.axis('off')
    ax2.legend(handles=[roi_box_rot], loc='lower right', fontsize=10)
    
    # --- Row 2: Depth Importance Profile (FIXED SECTION) ---
    ax3.plot(occ_scores, color='#00ffcc', marker='o', markersize=6, 
             linewidth=2, label='Slice Impact')
    
    # Fill between the line and the 0-baseline
    ax3.fill_between(range(len(occ_scores)), occ_scores, 0, color='#00ffcc', alpha=0.15)
    
    # Reference lines
    ax3.axhline(0, color='white', linewidth=0.8, linestyle='-', alpha=0.3) # Zero baseline
    ax3.axvline(top_slice, color='#ff3366', linestyle='--', linewidth=2, label='Critical Plane')
    
    # Dynamic Y-Axis: Handle negative values and provide text headroom
    # We set the bottom to slightly below the minimum value, 
    # and the top to 40% above the max value to fit the annotation.
    lower_limit = min_impact - (0.1 * impact_range) if min_impact < 0 else -0.005
    upper_limit = max_impact + (0.4 * impact_range)
    ax3.set_ylim(lower_limit, upper_limit) 
    
    # Annotation with better positioning
    ax3.annotate(f'Peak Impact: {max_impact:.4f}\nSlice {top_slice}',
                 xy=(top_slice, max_impact), 
                 xytext=(top_slice + 0.5, max_impact + (0.05 * impact_range)),
                 color='#ff3366', 
                 fontweight='bold', 
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='#ff3366', lw=1.5))

    ax3.set_title("Depth Importance Profile (Structural Impact on Confidence)", pad=20, fontsize=12)
    ax3.set_xlabel("Slice Index (Depth)", fontsize=10)
    ax3.set_ylabel("Confidence Drop / Change", fontsize=10)
    ax3.grid(True, alpha=0.1, linestyle='--')
    ax3.legend(loc='upper right', fontsize=10)

    # --- Final Layout Adjustments ---
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, hspace=0.4)
    
    roi_text = get_roi_interpretation(heatmap_data, pred_idx)
    plt.figtext(0.5, 0.02, roi_text, wrap=True, horizontalalignment='center',
                verticalalignment='bottom', fontsize=12, fontweight='bold', color='white',
                bbox={'facecolor': '#ff3366', 'alpha': 0.15, 'pad': 12, 
                      'edgecolor': '#ff3366', 'linewidth': 2})
    
    plt.style.use('default')
    return fig

# =============================================================================
# 5. PREPROCESSING UTILITIES
# =============================================================================
def extract_mpr_and_slice(filename):
    mpr = re.search(r'mpr-(\d+)', filename, re.I)
    slc = re.search(r'_(\d+)\.(jpg|png)', filename, re.I)
    return int(mpr.group(1)) if mpr else 0, int(slc.group(1)) if slc else 0

def preprocess_volume_st(slices, fe):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    processed = torch.stack([transform(s) for s in slices]).to(DEVICE)

    with torch.no_grad():
        features = fe(processed).cpu()

    features = features.permute(1, 0, 2, 3)
    intensities = [features[:, i].abs().mean().item() for i in range(features.shape[1])]
    keep = np.argsort(intensities)[PREPROCESSING_REMOVE_DARK_SLICES:]
    keep.sort()

    features = features[:, keep]
    slices = slices[keep]

    fv = features.numpy()
    fv = (fv - fv.mean()) / (fv.std() + 1e-8)

    fv = torch.FloatTensor(fv).unsqueeze(0).permute(0, 1, 3, 4, 2)
    fv = F.interpolate(fv, size=(8, 8, TARGET_DEPTH), mode='trilinear')
    fv = fv.permute(0, 1, 4, 2, 3).squeeze(0)

    idx = np.linspace(0, len(slices)-1, TARGET_DEPTH).astype(int)
    return fv, slices[idx]

def overlay_heatmap(img, cam, alpha=0.5):
    img_rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    return cv2.addWeighted(img_rgb, 1-alpha, heatmap, alpha, 0)

# =============================================================================
# 6. SIDEBAR NAVIGATION
# =============================================================================
@st.cache_resource
def load_model_hf():
    """Load model once from Hugging Face and cache it"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Download from HF
    model_file = hf_hub_download(
        repo_id="RuthChg/medicalnet-resnet18",  # replace with your HF repo
        filename="medicalnet_resnet18_90acc.pth"
    )
    
    model = MedicalNetResNet18(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    return model, device

st.sidebar.title("Navigation Panel")

page = st.sidebar.radio("Navigation", ["💡Introduction", "🔍How it Works","📏Evaluation Results", "👩‍💻MRI Diagnosis & XAI"])

# st.sidebar.info("Hybrid 3D-CNN Model with Multimodal Explainability")

# =============================================================================
# 7. INTRODUCTION PAGE
# =============================================================================
if page == "💡Introduction":
    # st.title("🧠 Alzheimer’s Disease Diagnostic System")
    st.markdown(
    "<h1 style='margin-top:0; font-size:38px;'>🧠 Alzheimer’s Disease Diagnostic System</h1>",
    unsafe_allow_html=True
)
    st.subheader("Deep Learning-Based MRI Classification with Explainable AI (XAI)")

    # st.header("🔍 Understanding Alzheimer’s Disease")
    st.markdown("""
   Alzheimer’s Disease is a progressive **neurodegenerative disorder** that slowly destroys memory, thinking skills and eventually the ability to carry out the simplest tasks. It is the most common cause of dementia, accounting for **60-80%** of cases worldwide.
    """)

    st.subheader("🧬 Why Does AD Occur?")
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.image("assets/brain_position.png", width=650)

    with col2:
        st.markdown("""The disease is characterized by specific pathological changes in the brain that begin years before symptoms appear:""")
        st.markdown("""
        - **Amyloid Plaques** Abnormal clumps of a protein fragment called beta-amyloid that build up between nerve cells, **disrupting communication**.
        - **Tau Tangles** Twisted fibers of another protein called tau that build up inside cells, destroying the cell's internal transport system.
        - **Neuronal Death** As these proteins accumulate, neurons (nerve cells) stop functioning, lose connections with other neurons, and eventually die, leading to significant brain shrinkage (atrophy).
        """)
    
if page=="🔍How it Works":

    st.header("🤖 How the System Works")

    st.markdown("""
    Our system leverages **Deep Learning** techniques to analyze brain MRI scans
    and support Alzheimer's disease assessment through the following capabilities:
    """)

    st.markdown("""
    ### 🔍 Detect Patterns
    Identify **microscopic structural changes** in brain MRI scans that may be
    **imperceptible to the human eye**, enabling early and precise feature recognition.

    ### 🧠 Binary Classification
    Automatically categorize MRI scans into:
    - **AD** — Alzheimer’s Disease  
    - **CN** — Cognitively Normal  

    ### 🧪 Explainable AI (XAI)
    Generate **heatmaps and saliency maps** that highlight the specific brain regions
    most influential in the model’s decision-making process, promoting
    **clinical transparency, interpretability, and trust**.
    """)

    st.warning("""
    ⚠️ **Disclaimer**  
    This system is intended as a **diagnostic aid for research and clinical support only**.
    It should be used **in conjunction with professional medical evaluation** and
    not as a standalone diagnostic tool.
    """)

    st.warning("This system is a research and clinical decision-support tool only.")


if page == "📏Evaluation Results":

    # ===============================
    # PAGE TITLE
    # ===============================
    st.markdown("""
    <h1 style='
        text-align:center;
        font-size:38px;
        color:#E0E0E0;
        margin-bottom:5px;
    '>📊 Model Evaluation Dashboard</h1>

    <p style='
        text-align:center;
        font-size:18px;
        color:#9E9E9E;
        margin-bottom:25px;
    '>Alzheimer’s Disease MRI Classification Performance</p>
    <hr style='border:1px solid #333;'>
    """, unsafe_allow_html=True)

    # ===============================
    # METRIC CARDS (DARK MODE)
    # ===============================

    def dark_metric_card(title, value, accent):
        # This entire block is one HTML string
        st.markdown(f"""
        <div style='
            background: rgba(255,255,255,0.06);
            backdrop-filter: blur(10px);
            padding: 24px;
            border-radius: 18px;
            text-align: center;
            color: #E0E0E0;
            border: 1px solid {accent};
            box-shadow: 0 0 18px {accent}33;
            height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        '>
            <h3 style='margin: 0 0 10px 0; font-size: 1.1rem; line-height: 1.2;'>{title}</h3>
            <h1 style='margin: 0; color: {accent}; font-size: 2.4rem; font-weight: 700;'>{value}</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🚀 Overall Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        dark_metric_card("Accuracy", "90.0%", "#00E676")

    with col2:
        dark_metric_card("Macro F1-Score", "86.7%", "#40C4FF")

    with col3:
        dark_metric_card("Weighted F1-Score", "90.1%", "#E040FB")

    st.markdown("<br>", unsafe_allow_html=True)

    # ===============================
    # CONFUSION MATRIX (DARK CARD)
    # ===============================
    st.markdown("### 🔍 Confusion Matrix")

        # Updated Confusion Matrix Layout
    cm_col1, cm_col2 = st.columns([1, 1])

    with cm_col1:
        st.markdown("<p style='font-weight: 600; color: #E0E0E0;'>PREDICTION MATRIX</p>", unsafe_allow_html=True)
        # Using a dataframe for better styling control
        df_cm = pd.DataFrame(
            [[49, 4], [3, 14]], 
            index=["Actual CN", "Actual AD"], 
            columns=["Pred CN", "Pred AD"]
        )
        st.dataframe(df_cm, use_container_width=True)

    with cm_col2:
        st.markdown("""
        <div style='background: rgba(76, 175, 80, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #4CAF50;'>
            <p style='margin:0; color:#81C784; font-size:13px; font-weight:bold;'>STRENGTHS</p>
            <p style='margin:0; color:#E0E0E0; font-size:14px;'>High specificity (94%) in identifying healthy patients.</p>
        </div>
        <div style='background: rgba(255, 82, 82, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #FF5252; margin-top:10px;'>
            <p style='margin:0; color:#FF8A80; font-size:13px; font-weight:bold;'>CRITICAL METRIC</p>
            <p style='margin:0; color:#E0E0E0; font-size:14px;'>AD Recall at 82.3%. Focus on minimizing missed cases.</p>
        </div>
        """, unsafe_allow_html=True)

    # ===============================
    # FIXED BAR CHART (NO SCROLL JUMP)
    # ===============================
    st.markdown("### 📊 Metric Comparison by Class")

    # 1. Prepare data in "long" format for professional plotting
    bar_data = pd.DataFrame([
        {"Metric": "Precision", "Class": "Healthy (CN)", "Value": 0.94},
        {"Metric": "Precision", "Class": "Dementia (AD)", "Value": 0.77},
        {"Metric": "Recall", "Class": "Healthy (CN)", "Value": 0.92},
        {"Metric": "Recall", "Class": "Dementia (AD)", "Value": 0.82},
        {"Metric": "F1-Score", "Class": "Healthy (CN)", "Value": 0.93},
        {"Metric": "F1-Score", "Class": "Dementia (AD)", "Value": 0.80},
    ])

    # 2. Create Grouped Bar Chart
    chart = alt.Chart(bar_data).mark_bar().encode(
        # x-axis is grouped by Metric
        x=alt.X('Metric:N', title=None, axis=alt.Axis(labelAngle=0)),
        # y-axis shows the score
        y=alt.Y('Value:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
        # Color differentiates the Class
        color=alt.Color('Class:N', scale=alt.Scale(range=['#4CAF50', '#FF9800'])),
        # This creates the "grouped" side-by-side effect
        xOffset='Class:N'
    ).properties(height=350)

    st.altair_chart(chart, use_container_width=True)

    # ===============================
    # CLASSIFICATION REPORT
    # ===============================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Detailed Classification Report")

    # Use a native container with a border instead of custom HTML <div>
    with st.container(border=True):
        df_data = {
            "Class": ["Non-Demented (CN)", "Dementia (AD)", "Macro Avg", "Weighted Avg"],
            "Precision": [0.9423, 0.7778, 0.8600, 0.9024],
            "Recall": [0.9245, 0.8235, 0.8740, 0.9000],
            "F1-Score": [0.9333, 0.8000, 0.8667, 0.9010],
            "Support": [53, 17, 70, 70]
        }
        st.dataframe(df_data, use_container_width=True, hide_index=True)

    # Replace the orange linear-gradient card with this:
    st.markdown("""
    <div style='
        background: #1E1E1E;
        border: 1px solid #333;
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
    '>
        <h3 style='color: #FFB74D; margin-top: 0; display: flex; align-items: center;'>
            <span style='margin-right: 10px;'>📋</span> Clinical Insights Summary
        </h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
            <div style='color: #BDBDBD; font-size: 15px;'>
                <p><b>Diagnostic Reliability:</b> Excellent performance in Cognitively Normal (CN) cohorts, providing a stable baseline for screening.</p>
                <p><b>Early Detection:</b> Sensitivity for AD is high, supporting the identification of early-stage biomarkers.</p>
            </div>
            <div style='color: #BDBDBD; font-size: 15px;'>
                <p><b>Transparency:</b> Grad-CAM integration allows for voxel-wise verification of anatomical atrophy.</p>
                <p><b>Usage:</b> Intended as a <u>second-opinion</u> tool for neuro-radiological workflows.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# 8. DIAGNOSIS PAGE
# =============================================================================

if page == "👩‍💻MRI Diagnosis & XAI":
    
    st.subheader("Upload Patient Data")
    uploaded_file = st.file_uploader(
        "Upload ZIP file containing MRI slices",
        type=['zip'],
        help="ZIP should contain sequential brain MRI slices"
    )
    
    if uploaded_file:
        st.success(f"✓ File uploaded: {uploaded_file.name}")
    
    st.subheader("🔬 Analysis Controls")
    analyze_button = st.button("🚀 Analyze Patient Data", type="primary", use_container_width=True)
    model_file = hf_hub_download(
        repo_id="RuthChg/medicalnet-resnet18",  # your HF repo
        filename="medicalnet_resnet18_90acc.pth"
    )
    # Analysis section
    if analyze_button and uploaded_file:
        try:
            # Load model
            with st.spinner("Loading model..."):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = MedicalNetResNet18(num_classes=2).to(device)
                model.load_state_dict(torch.load(model_file, map_location=device))
                model.eval()

            # Process data
            with st.spinner("Processing MRI data..."):
                input_tensor = process_zip_file(uploaded_file, target_depth=32, target_size=224)
                input_tensor = input_tensor.to(device)
            
            # Inference
            with st.spinner("Running diagnosis..."):
                # Inference
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = F.softmax(output, dim=1)
                    
                    # CHANGE THIS: Use argmax to match Kaggle exactly
                    pred_idx = torch.argmax(output, dim=1).item() 
                    confidence = prob[0, pred_idx].item() * 100

                class_names = ['Non-Dementia', 'Dementia']
                prediction = class_names[pred_idx]
            
            if confidence < 50:
                conf_color = "#ff4b4b"  # Red
            elif 50 <= confidence < 70:
                conf_color = "#ffa500"  # Orange/Yellow
            else:
                conf_color = "#00cc66"  # Green

            # Display results
            st.markdown("---")
            st.subheader("📊 Diagnosis Results")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.markdown(f"""
                    <div style="line-height: 1.5;">
                        <p style="margin: 0; font-size: 14px; color: rgba(250, 250, 250, 0.6);">Prediction</p>
                        <p style="margin: 0; font-size: 32px; font-weight: 600;">
                            {prediction}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with result_col2:
                # We use HTML/CSS here to apply the custom color to the confidence value
                st.markdown(f"""
                    <div style="line-height: 1.5;">
                        <p style="margin: 0; font-size: 14px; color: rgba(250, 250, 250, 0.6);">Confidence</p>
                        <p style="margin: 0; font-size: 32px; font-weight: 600; color: {conf_color};">
                            {confidence:.2f}%
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            # 1. Prepare Data
            labels = [class_names[0], class_names[1]]
            values = [prob[0, 0].item() * 100, prob[0, 1].item() * 100]
            colors = ['#1f77b4', '#ff3366'] 

            # 2. Create Figure
            # Use a context manager to keep the dark style local to this plot only
            with plt.style.context('dark_background'):
                fig_prob, ax_prob = plt.subplots(figsize=(8, 3))
                bars = ax_prob.barh(labels, values, color=colors)
                
                # Increase x-limit slightly to accommodate the text labels
                ax_prob.set_xlim(0, 115) 
                ax_prob.set_xlabel('Probability (%)')

                # 3. Add text labels
                for bar in bars:
                    width = bar.get_width()
                    ax_prob.text(width + 2, bar.get_y() + bar.get_height()/2, 
                                f'{width:.2f}%', va='center', fontweight='bold', color='white')

                # 4. Display in Streamlit
                st.pyplot(fig_prob)
                plt.close(fig_prob) # Free up memory

            # XAI Visualization
            with st.spinner("Generating explainable AI visualizations..."):
                # Ensure this function handles its own plt.style if it needs a specific look
                fig_xai = create_xai_visualization(input_tensor, model, pred_idx, confidence, class_names)
                st.pyplot(fig_xai)
                plt.close(fig_xai) # Free up memory

            # Clinical notes
            st.markdown("---")
            st.markdown("### 📝 Clinical Notes")
            if pred_idx == 1:
                st.warning("""
                **Dementia Detected**
                - The model has identified patterns consistent with Alzheimer's disease
                - Review Grad-CAM heatmap for affected brain regions
                - Hippocampal atrophy may be present
                - Recommend clinical confirmation and further testing
                """)
            else:
                st.success("""
                **Normal Brain Structure**
                - No significant dementia markers detected
                - Brain structure appears consistent with healthy aging
                - Continue regular monitoring as recommended
                """)
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)
    
    elif analyze_button and not uploaded_file:
        st.warning("⚠️ Please upload a ZIP file first")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>⚕️ For research and educational purposes only. Not for clinical diagnosis.</p>
    <p>© 2025 Alzheimer's Diagnosis System </p>
    </div>
    """, unsafe_allow_html=True)
