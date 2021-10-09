import tensorflow as tf

def dent_net_model_relu(input_shape,output_class):
  model=tf.keras.models.Sequential(
      [
       tf.keras.Input(shape=input_shape),
       tf.keras.layers.Conv2D(64,(3,3), padding='same',activation='relu'),
       tf.keras.layers.MaxPooling2D((2,2),2),
       tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'),
       tf.keras.layers.MaxPooling2D((2,2),2),
       tf.keras.layers.Conv2D(256,(3,3), padding='same',activation='relu'),
       tf.keras.layers.MaxPooling2D((2,2),2),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(1024,name="feature_map_1", activation='linear'),
       tf.keras.layers.Dropout(0.5),
       tf.keras.layers.Dense(512, name="feature_map_2", activation='linear'),
       tf.keras.layers.Dropout(0.5),
       tf.keras.layers.Dense(256, name="feature_map_3", activation='linear'),
       tf.keras.layers.Dropout(0.5),
       tf.keras.layers.Dense(output_class,activation='softmax'),
      ]
  )
  return model

model_1=dent_net_model_relu((128,128,1),302)
model_1.summary()

def dent_net_model_prelu(input_shape,output_class):
  model=tf.keras.models.Sequential(
      [
       tf.keras.Input(shape=input_shape),
       tf.keras.layers.Conv2D(64,(3,3), padding='same',activation='PReLU'),
       tf.keras.layers.MaxPooling2D((2,2),2),
       tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='PReLU'),
       tf.keras.layers.MaxPooling2D((2,2),2),
       tf.keras.layers.Conv2D(256,(3,3), padding='same',activation='PReLU'),
       tf.keras.layers.MaxPooling2D((2,2),2),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(1024,name="feature_map_1", activation='linear'),
       tf.keras.layers.Dropout(0.5),
       tf.keras.layers.Dense(512, name="feature_map_2", activation='linear'),
       tf.keras.layers.Dropout(0.5),
       tf.keras.layers.Dense(256, name="feature_map_3", activation='linear'),
       tf.keras.layers.Dropout(0.5),
       tf.keras.layers.Dense(output_class,activation='softmax'),
      ]
  )
  return model
  
model_2=dent_net_model_prelu((128,128,1),302)
model_2.summary()
