from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def build_model(num_classes, input_shape=(224, 224, 3), weights="imagenet"):
    """
    Build a CNN using EfficientNetB0 with ImageNet pretrained weights.
    """
    base = EfficientNetB0(include_top=False, input_shape=input_shape, weights=weights)
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=outputs)

    # Freeze base layers for transfer learning
    for layer in base.layers:
        layer.trainable = False

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
