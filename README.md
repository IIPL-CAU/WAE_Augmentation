# Text Style Transfer & Augmentation with WAE (Wasserstein Auto-Encoder)

### Pre-processing

```
python3 main.py --preprocessing --dataset=IMDB
```

**These arguments must be provided**
- dataset: Name of a directory below dataset_path, which contations train.csv and test.csv

**Additional arguments**


**After execution, These files will be generated**


### Augmentation Training

```
python3 main.py --augment_training --dataset=IMDB
```

**These arguments must be provided**
- dataset: Name of a directory below dataset_path, which contations train.csv and test.csv

**Additional arguments**


**After execution, These files will be generated**

### Augmentation

```
python3 main.py --augmentation --dataset=IMDB
```

**These arguments must be provided**
- dataset: Name of a directory below dataset_path, which contations train.csv and test.csv

**Additional arguments**


**After execution, These files will be generated**

### Classification Training

```
python3 main.py --training --dataset=IMDB
```

**These arguments must be provided**
- dataset: Name of a directory below dataset_path, which contations train.csv and test.csv

**Additional arguments**


**After execution, These files will be generated**