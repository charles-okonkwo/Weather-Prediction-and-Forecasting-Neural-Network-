from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Input Layer
input_layer = Input(shape=(8,))

# Encoder
hidden1 = Dense(
    6,
    activation='relu'
)(input_layer)

bottleneck = Dense(
    3,
    activation='linear'
)(hidden1)

# Decoder
hidden2 = Dense(
    6,
    activation='relu'
)(bottleneck)

output_layer = Dense(
    8,
    activation='sigmoid'
)(hidden2)

# Autoencoder Model
autoencoder = Model(
    inputs=input_layer,
    outputs=output_layer
)

autoencoder.compile(
    optimizer='adam',
    loss='mse'
)

# Show Structure
autoencoder.summary()