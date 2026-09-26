# RakshaVaani

voice clone detector, built for the SIH hackathon (problem statement 6 - real-time detection of voice cloning/impersonation attacks).

## the problem

voice cloning scams are actually a huge deal in india right now. 47% of indian adults have either been scammed or know someone who has been (mcafee survey put this out), which is almost double the global average. haryana's cyber cell logged a 450% jump in these cases in just one quarter. and most of the voice-clone detectors that exist right now are trained on clean english studio audio, which is basically nothing like how these scams actually happen - real scam calls are compressed, noisy, and often not in english at all.

so that's the gap this is trying to close.

## what we did

used `facebook/wav2vec2-xls-r-300m` (pretrained multilingual speech model) as a frozen feature extractor - didn't fine-tune the whole thing since that's just not feasible to do fast on a laptop without a proper GPU. trained a small logistic regression classifier on top of the embeddings instead. this is a pretty standard transfer learning approach, nothing fancy but it works.

trained on ASVspoof2019 (LA), which has labeled bonafide/spoof speech samples.

the main trick is before we extract embeddings, we run the audio through a telephony simulation - downsample to 8khz, mu-law encode/decode (basically simulating the codec compression a real phone call uses), add some background noise. point is to make the model actually useful for real phone-call conditions instead of just clean lab recordings.

## files in here

- `RakshaVaani_Voice_Clone_Detector.ipynb` - the whole pipeline, dataset download through to a working inference function
- `rakshavaani_webapp/` - small flask app, lets you record your voice or upload a clip in browser and get a verdict

## results

trained on 800 samples (400 real, 400 spoof) since that's what fit in the time we had.

- accuracy: ~83.5%
- EER on phone-simulated audio: ~17%
- EER on clean audio: ~19.5%

these aren't final numbers, this is a fast prototype. next step would be training on actual indian-language spoof datasets (SEA-Spoof, IndicSynth exist for this) to get the real english-vs-hindi/tamil gap number, which is really the whole point of the project.

## running this

notebook:
```
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio transformers librosa soundfile scikit-learn kagglehub pandas numpy matplotlib jupyter
jupyter notebook
```
need a kaggle api token in `~/.kaggle/kaggle.json` for the dataset download to work.

webapp (run this after the notebook has saved `rakshavaani_classifier_head.joblib`):
```
pip install flask
cd rakshavaani_webapp
python3 app.py
```
then open `http://127.0.0.1:5050`

## where this could actually be used

- bank/UPI call centers, server side, since they already have the audio
- telecom carriers as a spam/risk score alongside existing call flagging
- cybercrime cells / 1930 helpline as a forensic check
- a whatsapp voice-note checker for regular people, since android/ios don't let third party apps touch live call audio directly

## team

A4 CODERS
