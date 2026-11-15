class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunkSize = 4096;
    this.buffer = new Float32Array(this.chunkSize);
    this.offset = 0;
    this.port.onmessage = (event) => {
      if (event.data?.type !== "flush") return;
      this.sendPendingSamples();
      this.port.postMessage({ type: "flushed" });
    };
  }

  sendPendingSamples() {
    if (!this.offset) return;
    const samples = this.buffer.slice(0, this.offset);
    this.offset = 0;
    this.port.postMessage({ type: "audio", samples }, [samples.buffer]);
  }

  process(inputs, outputs) {
    const input = inputs[0]?.[0];
    if (input) {
      let sourceOffset = 0;
      while (sourceOffset < input.length) {
        const count = Math.min(input.length - sourceOffset, this.chunkSize - this.offset);
        this.buffer.set(input.subarray(sourceOffset, sourceOffset + count), this.offset);
        this.offset += count;
        sourceOffset += count;
        if (this.offset === this.chunkSize) this.sendPendingSamples();
      }
    }

    for (const output of outputs) {
      for (const channel of output) channel.fill(0);
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
