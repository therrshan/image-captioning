import Preprocessing
import models

from keras.callbacks import ModelCheckpoint

epoch = 20
batch_size = 128
checkpoint_path = Preprocessing.current_working_directory+'/drive/MyDrive/weights-model3.{epoch:02d}-{loss:.2f}.h5'
# Create a ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor='accuracy',
    save_weights_only=True,
    save_best_only=False,
    period=10  # Save weights every 10 iterations
)

models.fin_model.fit_generator(Preprocessing.data_process(batch_size=batch_size), steps_per_epoch=Preprocessing.no_samples/batch_size, epochs=epoch, verbose=1,callbacks=[checkpoint_callback])
models.fin_model.save(Preprocessing.current_working_directory+"/drive/MyDrive/Weights_Bidirectional_LSTM-model3.h5")



