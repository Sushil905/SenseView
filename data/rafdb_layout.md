# RAF-DB input layout

RAF-DB is not redistributed with this repository. Obtain access from the
[dataset owners](https://www.whdeng.cn/RAF/model1.html) and put the aligned basic-expression images and
`list_patition_label.txt` under `data/raw/rafdb/`:

```
data/raw/rafdb/
  aligned/
    train_00001_aligned.jpg
    test_00001_aligned.jpg
  list_patition_label.txt
```

The annotation list is whitespace separated: image filename, then integer
label. Basic-expression labels map as follows: 1 surprise, 2 fear, 3 disgust,
4 happiness, 5 sadness, 6 anger, 7 neutral. RAF-DB contains still images, so it
trains the per-frame ViT model only; the temporal LSTM requires separate
synchronized clips with a `yaw_degrees` value per camera and timestep. Do not
manufacture those sequences from RAF-DB still images.
