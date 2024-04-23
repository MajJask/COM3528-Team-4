import numpy as np

class SoundLocalizer:
    def __init__(self, mic_distance=0.1):  # Assume 10 cm between mics as an example
        self.mic_distance = mic_distance
        self.speed_of_sound = 343  # Speed of sound in m/s

    def gcc(self, mic1, mic2):
        pad = np.zeros(len(mic1))
        s1 = np.hstack([mic1, pad])
        s2 = np.hstack([pad, mic2])
        f_s1 = np.fft.fft(s1)
        f_s2 = np.fft.fft(s2)
        f_s2c = np.conj(f_s2)
        f_s = f_s1 * f_s2c
        denom = np.abs(f_s)
        denom[denom == 0] = 1e-10
        f_s /= denom
        correlation = np.fft.ifft(f_s)
        delay = np.argmax(np.abs(correlation)) - len(mic1)
        return delay

    def process_data(self, data_left, data_right, data_tail):
        delay_left_right = self.gcc(data_left, data_right)
        delay_left_tail = self.gcc(data_left, data_tail)
        delay_right_tail = self.gcc(data_right, data_tail)

        # Convert delays to angles using small angle approximation
        angle_left_right = (delay_left_right / self.speed_of_sound) * self.mic_distance
        angle_left_tail = (delay_left_tail / self.speed_of_sound) * self.mic_distance
        angle_right_tail = (delay_right_tail / self.speed_of_sound) * self.mic_distance

        # Simple average of angles as a naive triangulation approach
        estimated_direction = np.mean([angle_left_right, angle_left_tail, angle_right_tail])

        return estimated_direction

# Example of using the class
localizer = SoundLocalizer()
data_left = np.random.rand(1024)
data_right = np.random.rand(1024)
data_tail = np.random.rand(1024)
direction = localizer.process_data(data_left, data_right, data_tail)
print("Estimated direction of sound:", direction)