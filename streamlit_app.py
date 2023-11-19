import streamlit as st
import torch
import io
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from visualize import visualize_results
from network_model import transform, Net # network_model.py から前処理とネットワークの定義を読み込み

# Streamlitインターフェースの設定
st.sidebar.title('血液の顕微鏡画像から細胞検出')

# ネットワークの準備
net = Net().cpu().eval()
# 学習済みモデルの重み（blood.pt）を読み込み
net.load_state_dict(torch.load('blood.pt', map_location=torch.device('cpu')))

# ユーザーが画像をアップロード
uploaded_file = st.sidebar.file_uploader("画像をアップロードしてください", type=["jpg"])

if uploaded_file is not None:

    # 画像の読み込み
    image = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')

    # 画像の前処理
    image = transform(image)
    image = image.unsqueeze(0) # 1次元増やす

    # 推論
    prediction = net(image)[0]

    # 結果の抽出
    boxes = prediction['boxes'].tolist()
    labels = prediction['labels'].tolist()
    scores = prediction['scores'].tolist()

    # 画像のTensorをPillowのImageに変換
    pil_image = to_pil_image(image.squeeze(0))

    st.image(visualize_results(pil_image, boxes, labels, scores), caption='推論結果画像', use_column_width="auto")
