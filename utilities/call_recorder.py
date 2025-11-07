import threading
import time
import wave

import numpy as np
import pyaudiowpatch as pyaudio

# Config
RATE = 48000          # common for loopback + mic; downsample later for Whisper
CHUNK = 1024          # frames per read
FORMAT = pyaudio.paInt16
BYTES_PER_SAMPLE = 2  # int16

def pick_wasapi_loopback(p: pyaudio.PyAudio):
    """Pick default speakers' loopback device."""
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    def_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])

    # if default is already loopback, use it
    if def_out.get("isLoopbackDevice"):
        return def_out

    # otherwise find matching loopback by name
    for info in p.get_loopback_device_info_generator():
        if def_out["name"] in info["name"]:
            return info

    # fallback: first loopback
    for info in p.get_loopback_device_info_generator():
        return info

    raise RuntimeError("No WASAPI loopback device found.")

def pick_default_mic(p: pyaudio.PyAudio):
    """Pick a non-loopback input device as mic."""
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        mic = p.get_device_info_by_index(wasapi["defaultInputDevice"])
    except OSError:
        mic = p.get_default_input_device_info()

    if not mic or mic.get("maxInputChannels", 0) <= 0:
        # fallback: first non-loopback input
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0 and not info.get("isLoopbackDevice", False):
                return info
        raise RuntimeError("No suitable microphone device found.")

    if mic.get("isLoopbackDevice", False):
        # choose another device if this one is loopback
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0 and not info.get("isLoopbackDevice", False):
                return info
        raise RuntimeError("Only loopback devices found; no real mic?")
    return mic

def record_call_stereo(filename: str, stop_event: threading.Event):
    """
    Record loopback (far side) + mic (near side) into a 2-channel WAV:
      ch0 = loopback (what you hear)
      ch1 = mic      (what you say)
    Stop when stop_event is set (Ctrl+C handler, GUI button, etc.).
    """
    p = pyaudio.PyAudio()
    try:
        spk = pick_wasapi_loopback(p)
        mic = pick_default_mic(p)

        rate = RATE  # we ask both for the same; WASAPI usually handles it
        spk_ch = min(2, spk["maxInputChannels"])
        mic_ch = 1   # we’ll just take mono from mic

        print(f"[loopback] ({spk['index']}) {spk['name']} @ {rate} Hz, ch={spk_ch}")
        print(f"[mic]      ({mic['index']}) {mic['name']} @ {rate} Hz, ch={mic_ch}")

        spk_stream = p.open(format=FORMAT,
                            channels=spk_ch,
                            rate=rate,
                            input=True,
                            input_device_index=spk["index"],
                            frames_per_buffer=CHUNK)

        mic_stream = p.open(format=FORMAT,
                            channels=mic_ch,
                            rate=rate,
                            input=True,
                            input_device_index=mic["index"],
                            frames_per_buffer=CHUNK)

        wf = wave.open(filename, "wb")
        wf.setnchannels(2)                  # stereo: [loopback, mic]
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(rate)

        print(f"▶ Recording to {filename}")
        print("   ch0 = system audio, ch1 = your mic")
        print("   Press Ctrl+C to stop.")
        try:
            while not stop_event.is_set():
                # read raw bytes
                spk_data = spk_stream.read(CHUNK, exception_on_overflow=False)
                mic_data = mic_stream.read(CHUNK, exception_on_overflow=False)

                # to int16 arrays
                spk_arr = np.frombuffer(spk_data, dtype=np.int16)
                mic_arr = np.frombuffer(mic_data, dtype=np.int16)

                # downmix loopback to mono if needed
                if spk_ch > 1:
                    spk_arr = spk_arr.reshape(-1, spk_ch).mean(axis=1).astype(np.int16)

                # ensure same length
                n = min(len(spk_arr), len(mic_arr))
                if n <= 0:
                    continue
                spk_arr = spk_arr[:n]
                mic_arr = mic_arr[:n]

                # stack into [N, 2] -> interleaved stereo
                stereo = np.column_stack((spk_arr, mic_arr)).astype(np.int16)
                wf.writeframes(stereo.tobytes())
        finally:
            wf.close()
            spk_stream.stop_stream(); spk_stream.close()
            mic_stream.stop_stream(); mic_stream.close()
            print("\n⏹ Stopped.")
    finally:
        p.terminate()

if __name__ == "__main__":
    stop = threading.Event()
    try:
        record_call_stereo("call_recording.wav", stop)
    except KeyboardInterrupt:
        stop.set()
