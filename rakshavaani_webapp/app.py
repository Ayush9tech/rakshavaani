import os
import time
import torch
import torchaudio.functional as AF
import numpy as np
import soundfile as sf
import joblib
from flask import Flask, request, jsonify, render_template
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# --- Point this at the same project folder your notebook used ---
PROJECT_DIR = os.path.expanduser("~/Desktop/vaaniraksha")
MODEL_PATH = os.path.join(PROJECT_DIR, "rakshavaani_classifier_head.joblib")

app = Flask(__name__)

print("Checking device...")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

print("Loading XLS-R backbone (uses local HuggingFace cache from your notebook run)...")
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
backbone = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False

print("Loading trained classifier head from:", MODEL_PATH)
clf = joblib.load(MODEL_PATH)
print("Ready. Starting server...")


def simulate_phone_call(waveform, orig_sr, noise_level=0.005):
    target_sr = 8000
    wav_8k = AF.resample(waveform, orig_sr, target_sr)
    wav_mu = AF.mu_law_encoding(wav_8k, quantization_channels=256)
    wav_decoded = AF.mu_law_decoding(wav_mu, quantization_channels=256)
    noise = torch.randn_like(wav_decoded) * noise_level
    wav_noisy = wav_decoded + noise
    wav_final = AF.resample(wav_noisy, target_sr, 16000)
    return wav_final


@torch.no_grad()
def get_embedding(filepath, apply_telephony_sim=True):
    audio, sr = sf.read(filepath, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    if sr != 16000:
        waveform = AF.resample(waveform, sr, 16000)
        sr = 16000
    if apply_telephony_sim:
        waveform = simulate_phone_call(waveform, sr)
    inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    outputs = backbone(input_values)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files["audio"]
    tmp_path = os.path.join(PROJECT_DIR, "_tmp_upload.wav")
    audio_file.save(tmp_path)

    try:
        t0 = time.time()
        emb = get_embedding(tmp_path, apply_telephony_sim=True)
        prob_spoof = float(clf.predict_proba(emb.reshape(1, -1))[0, 1])
        verdict = "spoof" if prob_spoof > 0.5 else "real"
        confidence = prob_spoof if prob_spoof > 0.5 else 1 - prob_spoof
        elapsed = time.time() - t0
        return jsonify({
            "verdict": verdict,
            "confidence": round(confidence * 100, 1),
            "inference_time": round(elapsed, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(debug=False, port=5050)
