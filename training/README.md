# Model training

`anemia_nailbed_training.ipynb` trains the RaktSetu classifier on Google Colab.

1. Open [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → pick this file.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**. The Ghana and Yakimov datasets download automatically.
4. To add photos collected with the device, download the training-data zip from the NailScan report,
   put it in Google Drive and set `MY_DATA_PATH` and `METADATA_CSV` (`metadata.csv` inside the zip) in Step 2.

The notebook ends by downloading `anemia_pi_package.zip`. Copy only these three files from it into
`device/models/`, replacing the old ones, then commit:

- `anemia_model_float32.tflite`
- `anemia_model_int8.tflite`
- `model_config.json`

(Ignore the `predict_pi.py` inside that zip; `device/predict_pi.py` is the up-to-date version.)
