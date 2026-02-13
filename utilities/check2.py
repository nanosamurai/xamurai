import os, sys, ctypes, torchcodec
print("torchcodec:", torchcodec.__version__)
dlls = [f for f in os.listdir(os.path.join(sys.prefix, "Lib","site-packages","torchcodec")) if f.startswith("libtorchcodec_core")]
print("found:", dlls)
core = [d for d in dlls if "7" in d] or dlls  # prefer ffmpeg7 core
ctypes.CDLL(os.path.join(sys.prefix, "Lib","site-packages","torchcodec", core[0]))
print("CDLL load OK")