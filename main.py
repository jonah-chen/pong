from ai import build_model, train
import numpy as np
from play import self_play, evaluate_against_chaser
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

model = build_model(2,128,64)
model.compile(optimizer=RMSprop(lr=1e-4), loss=categorical_crossentropy)
while 1:
    for i in range(10):
        print(f"Batch {i}:")
        evaluate_against_chaser(model)
        x, r = self_play(model)
        model.fit(x, r, batch_size=256, epochs=1, shuffle=True, use_multiprocessing=True)
        model.evaluate(x, r, batch_size=256, use_multiprocessing=True)
    model.save("kick_guerzhoys_ass.h5")

# for _ in range(10):
#     x1, r1 = self_play_2(model, games=100)
#  # tutorial trains on 100 game mini-batch

#     checkpoint = ModelCheckpoint('checkpoint', save_best_only=True, monitor='loss')
#     model.compile(optimizer=SGD(lr=1e-2, momentum=0.9), loss=categorical_crossentropy)
#     loss = model.train_on_batch(x, r)
#     print(f"Loss: {loss:.5f}")
    
# model.save("kick_guerzhoys_ass.h5")