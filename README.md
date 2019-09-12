# Google Cloud Starter Project Scripts
Scripts for preparing image datasets and uploading to Google Cloud Storage to train an AutoML model. Used in Samoyed captcha project

# Usage

`prepare_dataset.py`:

Structure your files so that images for each label are contained in their own folder:
```
├── alice
│   └── alice0.jpg
│   ├── alice1.jpg
│   └── alice2.jpg
│   └── ...
├── jamie
│   ├── jamie0.jpg
│   ├── jamie1.jpg
│   └── jamie2.jpg
│   └── ...
└── prepare_dataset.py
```

Then run the script in the directory containing the labeled folders:

```
python prepare_dataset.py jamie-alice-classifier-251416-vcm --labels jamie alice --crop_iterations=2 --rename=True
```

The script will crop all images, and rename files if rename option is set.
