import tensorflow as tf

class LivenessNet:
    
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # 'channels last' and the channels dimension itself
        INPUT_SHAPE = (height, width, depth)
        chanDim = -1 # use for batch normalization along axis
        
        # if we are using "channels first", update the input shape
        # and channels dimension
        # note that: normally, by default, it's "channels last"
        if tf.keras.backend.image_data_format() == 'channels_first':
            INPUT_SHAPE = (depth, height, width)
            chanDim = 1
        
        # Our CNN exhibits VGGNet-esque qualities. It is very shallow with only a few learned filters. 
        # Ideally, we won’t need a deep network to distinguish between real and spoofed faces.
        model = tf.keras.Sequential([
                # first set CONV => BatchNorm CONV => BatchNorm => MaxPool => Dropout
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu' ,input_shape=INPUT_SHAPE),
                tf.keras.layers.BatchNormalization(axis=chanDim),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(axis=chanDim),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                # second set CONV => BatchNorm CONV => BatchNorm => MaxPool => Dropout
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(axis=chanDim),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(axis=chanDim),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                # FullyConnected => BatchNorm => Dropout
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                
                # output
                tf.keras.layers.Dense(classes, activation='softmax')
            ])
        
        return model