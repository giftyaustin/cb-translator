import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, GLib
import threading

Gst.init(None)


class FullDuplexAudio:
    def __init__(self, recv_port=5002, send_host='127.0.0.1', send_port=5004):
        self.recv_port = recv_port
        self.send_host = send_host
        self.send_port = send_port

        self.pipeline = Gst.parse_launch(f"""
            rtpbin name=rtpbin latency=100
            udpsrc port={self.recv_port} caps="application/x-rtp, media=audio, clock-rate=48000, encoding-name=OPUS, payload=96" !
                rtpbin.recv_rtp_sink_0
            rtpbin. ! rtpopusdepay ! opusdec ! audioconvert ! audioresample ! appsink name=mysink emit-signals=true max-buffers=1 drop=true

            appsrc name=mysource format=time is-live=true block=true caps=audio/x-raw,rate=48000,channels=1 !
                audioconvert ! audioresample ! opusenc ! rtpopuspay !
                udpsink host={self.send_host} port={self.send_port}
        """)

        self.appsink = self.pipeline.get_by_name("mysink")
        self.appsrc = self.pipeline.get_by_name("mysource")

        self.appsink.connect("new-sample", self.on_new_sample)

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            # Get raw audio bytes from buffer
            success, mapinfo = buffer.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR

            audio_data = mapinfo.data  # Raw PCM data (16-bit)
            buffer.unmap(mapinfo)

            # 🧠 Do your processing here (AI, effects, logging, etc.)
            print(f"🔊 Received {len(audio_data)} bytes of audio")

            # Push it back out
            self.push_audio(audio_data)

        return Gst.FlowReturn.OK

    def push_audio(self, data):
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        self.appsrc.emit("push-buffer", buf)

    def start(self):
        self.pipeline.set_state(Gst.State.PLAYING)

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)


def run_mainloop():
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Main loop stopped.")


if __name__ == '__main__':
    audio = FullDuplexAudio(recv_port=5002, send_port=5004)
    audio.start()

    # Start GLib main loop in a thread
    threading.Thread(target=run_mainloop).start()
