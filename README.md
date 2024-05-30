# Clasisfication Demo App
A streamlit app with which you can inteface with your PyTorch image classifiers using your camera.

To start the app, install the requirements and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or use the startup script
```bash
./run.sh
```

Include your model's weight dictionary in the local directory and update the `app.py` file to load the correct weights. Currently, the script loads from `resnet50_gtsrb.pth`.