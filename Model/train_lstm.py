#!../Venv/bin/python

print(":::::::::::::::::::: TRAINING ::::::::::::::::::::")
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn as sk
import joblib
import random
import gc

SEQ_LEN = 20

if __name__ == "__main__":
  df = pd.read_hdf("./dataset.h5", key="df")

  # Create X and Y
  y = df["chord"]
  X = df.drop(columns="chord")

  # Creating label encoder
  encoder = sk.preprocessing.LabelEncoder()
  y_encoded = encoder.fit_transform(y)
  joblib.dump(encoder, "./encoder.xz")
  del df
  del y
  gc.collect()

  # Creating sequence
  X_seq, y_encoded_seq = None, None
  X_seq_list = []
  y_encoded_seq_list = []
  for i in range(len(X) - SEQ_LEN + 1):
      X_seq_list.append(X.values[i : i + SEQ_LEN, :])
      y_encoded_seq_list.append(y_encoded[i + SEQ_LEN - 1])
  
  X_seq, y_encoded_seq = np.array(X_seq_list), np.array(y_encoded_seq_list)
  
  print("X sequence shape: ", X_seq.shape)
  print("y sequence shape: ", y_encoded_seq.shape)
  
  del X_seq_list
  del y_encoded_seq_list
  gc.collect()

  # Removing duplicate sequences
  X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)
  _, unique_idx = np.unique(X_seq_flat, axis=0, return_index=True)
  unique_idx = np.sort(unique_idx)

  X_seq = X_seq[unique_idx]
  y_encoded_seq = y_encoded_seq[unique_idx]

  print("X deduplicate shape: ", X_seq.shape)
  print("y deuplicate shape:  ", y_encoded_seq.shape)

  del X_seq_flat
  gc.collect()

  # Creating train & test data
  X_seq_train, X_seq_test, y_seq_train, y_seq_test = sk.model_selection.train_test_split(
    X_seq,
    y_encoded_seq,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded_seq,
  )

  print(f"X_train: {len(X_seq_train)}")
  print(f"y_train: {len(y_seq_train)}")
  print(f"X_test:  {len(X_seq_test)}")
  print(f"y_test:  {len(y_seq_test)}")

  del X_seq
  del y_encoded_seq
  gc.collect()
  print("Total feature: ", X_seq_train.shape[2])
  print("Total class:   ", len(encoder.classes_))

  # Class Weight
  class_weights = sk.utils.class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_seq_train),
    y=y_seq_train,
  )
  class_weight_dict = dict(enumerate(class_weights))

  # Model building
  model_lstm = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X_seq_train.shape[2]))),
    tf.keras.layers.Dropout(0.25),
        
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(len(encoder.classes_), activation='softmax'),
  ])

  model_lstm.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
  )
#model_lstm.summary()
  
  # Training
  early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True,
  )

  history = model_lstm.fit(
    X_seq_train,
    y_seq_train,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[early_stopping],
    class_weight=class_weight_dict
  )

  # Model evalutaion (TODO: also add this to explore)
  model_lstm.evaluate(X_seq_test, y_seq_test)

  y_pred = np.argmax(model_lstm.predict(X_seq_test), axis=1)
  print(
    sk.metrics.classification_report(y_seq_test, y_pred, target_names=encoder.classes_)
  )
  #print("Precision: ", sk.metrics.precision_score(y_seq_test, y_pred, average='macro'))
  #print("Recall   : ", sk.metrics.recall_score(y_seq_test, y_pred, average='macro'))
  #print("f1       : ", sk.metrics.f1_score(y_seq_test, y_pred, average='macro'))

  # Saving models
  model_lstm.save("model_lstm.keras")
  pd.DataFrame(history.history).to_csv("history.csv", index=False)
