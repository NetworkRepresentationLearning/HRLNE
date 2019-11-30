# HRLNE

This is a TensorFlow Implementation of HRLNE(Hierarchical Reinforcement Learning for Network Embedding)

Thers are main four stages for the training process:

First, we pre-train the STNE;

Second, we pre-train the HRSS;

Third, we jointly tune STNE and HRSS;

Finally, we train STNE with only refined samples given by HRSS.


For stage 1-3, we can run:

# Model Training
```
python train.py
```

# select more meaningful paths

```
python select_path_train.py
python delete_padding_path.py
```

# Train STNE with selected paths
