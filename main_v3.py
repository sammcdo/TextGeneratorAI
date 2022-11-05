import tensorflow as tf

import numpy as np
import os
import time

one_step_reloaded = tf.saved_model.load('models/presDebate')

states = None
next_char = tf.constant(['BIDEN:'])
result = [next_char]

one_step_reloaded.temperature = 0.25

for n in range(1000):
  next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
  result.append(next_char)

print(tf.strings.join(result)[0].numpy().decode("utf-8"))