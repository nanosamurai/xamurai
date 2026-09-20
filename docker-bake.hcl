variable "TAG" { default = "local" }

target "nemo-speech-native" {
  dockerfile = "nemo_speech_native/Dockerfile"
}

target "rtservice" {
  dockerfile = "rtservice/Dockerfile"
  tags = ["xamurai-rtservice:${TAG}"]
}

target "whisperx-worker" {
  dockerfile = "whisperx_worker/Dockerfile.refinement"
  tags = ["xamurai-whisperx-worker:${TAG}"]
}

target "finalizer-worker" {
  dockerfile = "whisperx_worker/Dockerfile.finalizer"
  contexts = { whisperx-worker = "target:whisperx-worker" }
  tags = ["xamurai-finalizer-worker:${TAG}"]
}

target "recorder-worker" {
  dockerfile = "recorder_worker/Dockerfile"
  tags = ["xamurai-recorder-worker:${TAG}"]
}

target "qwen-rtservice" {
  dockerfile = "qwen_rtservice/Dockerfile"
  tags = ["xamurai-qwen-rtservice:${TAG}"]
}

target "qwen-finalizer" {
  dockerfile = "qwen_worker/Dockerfile.finalizer"
  contexts = { qwen-rtservice = "target:qwen-rtservice" }
  tags = ["xamurai-qwen-finalizer:${TAG}"]
}

target "qwen-refinement" {
  dockerfile = "qwen_worker/Dockerfile.refinement"
  contexts = { qwen-finalizer = "target:qwen-finalizer" }
  tags = ["xamurai-qwen-refinement:${TAG}"]
}

target "nemotron-rtservice" {
  dockerfile = "nemotron_rtservice/Dockerfile"
  contexts = { nemo-speech-native = "target:nemo-speech-native" }
  tags = ["xamurai-nemotron-rtservice:${TAG}"]
}

target "parakeet-finalizer" {
  dockerfile = "parakeet_worker/Dockerfile.finalizer"
  contexts = { nemo-speech-native = "target:nemo-speech-native" }
  tags = ["xamurai-parakeet-finalizer:${TAG}"]
}

target "parakeet-refinement" {
  dockerfile = "parakeet_worker/Dockerfile.refinement"
  contexts = { parakeet-finalizer = "target:parakeet-finalizer" }
  tags = ["xamurai-parakeet-refinement:${TAG}"]
}
