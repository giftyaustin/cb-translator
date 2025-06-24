import * as mediasoupClient from 'mediasoup-client';
import { socket } from './socket';

let device: mediasoupClient.Device;
let sendTransport: mediasoupClient.types.Transport;
let recvTransport: mediasoupClient.types.Transport;
const consumedProducers = new Set<string>();
let pc: RTCPeerConnection | null = null;

export async function startMediasoup(
  roomCode: string,
  onNewConsumerStream: (stream: MediaStream, kind: string) => void
) {
  return new Promise<void>((resolve) => {
    socket.emit('get-rtp-capabilities');

    socket.once('rtp-capabilities', async (rtpCapabilities) => {
      device = new mediasoupClient.Device();
      await device.load({ routerRtpCapabilities: rtpCapabilities });

      // Create Send Transport
      socket.emit('create-transport', { direction: 'send' });
      socket.once('transport-created-send', async (params) => {
        sendTransport = device.createSendTransport(params);

        sendTransport.on('connect', ({ dtlsParameters }, callback) => {
          socket.emit('connect-transport-send', { dtlsParameters });
          socket.once('transport-connected-send', callback);
        });

        sendTransport.on('produce', (params, callback) => {
          socket.emit('produce', { ...params, roomCode });
          socket.once('produced', ({ id }) => {
            callback({ id });
          });
        });

        // Create Recv Transport
        socket.emit('create-transport', { direction: 'recv' });
        socket.once('transport-created-recv', async (recvParams) => {
          recvTransport = device.createRecvTransport(recvParams);

          recvTransport.on('connect', ({ dtlsParameters }, callback) => {
            socket.emit('connect-transport-recv', { dtlsParameters });
            socket.once('transport-connected-recv', callback);
          });

          socket.on('new-producer', async ({ producerId }) => {
            await consume(producerId, onNewConsumerStream);
          });

          resolve();
        });
      });
    });
  });
}

export async function startStreaming(stream: MediaStream, roomCode: string) {
  for (const track of stream.getTracks()) {
    await sendTransport.produce({ track });
  }
}

async function consume(
  producerId: string,
  onNewConsumerStream: (stream: MediaStream, kind: string) => void
) {
  if (consumedProducers.has(producerId)) return;
  consumedProducers.add(producerId);

  socket.emit('consume', {
    producerId,
    rtpCapabilities: device.rtpCapabilities,
  });

  socket.once('consumed', async ({ id, producerId, kind, rtpParameters }) => {
    const consumer = await recvTransport.consume({
      id,
      producerId,
      kind,
      rtpParameters,
    });

    const stream = new MediaStream([consumer.track]);
    onNewConsumerStream(stream, kind);

    if (kind === 'audio') {
      const track = stream.getAudioTracks()[0];
      if (track) {
        await sendAudioToPython(track);
      }
    }
    if (kind === 'video') {
      const track = stream.getVideoTracks()[0];
      if (track) {
        await sendVideoToPython(track);
      }
    }
  });
}

export async function setupPeerConnection(audioTrack?: MediaStreamTrack) {
  if (!pc) {
    pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });

    pc.onicecandidate = (event) => {
      if (event.candidate) {
        console.log('🧊 ICE candidate:', event.candidate);
      }
    };

    pc.ontrack = (event) => {
      console.log('🎤 Received track from server:', event.track.kind);

      const stream = event.streams[0] || new MediaStream([event.track]);

      if (event.track.kind === 'audio') {
        const audioElement = document.createElement("audio");
        audioElement.autoplay = true;
        audioElement.controls = true;
        audioElement.srcObject = stream;
        document.body.appendChild(audioElement);

        audioElement.play().then(() => {
          console.log('▶️ Playing audio from server');
        }).catch(err => {
          console.error('❌ Error playing audio:', err);
        });
      }
    };
  }

  if (audioTrack) {
    pc.addTransceiver(audioTrack, { direction: "sendrecv" });
  }

  await negotiate();
}

async function negotiate() {
  try {
    const offer = await pc!.createOffer();
    await pc!.setLocalDescription(offer);

    await new Promise((resolve) => {
      if (pc!.iceGatheringState === 'complete') {
        resolve(null);
      } else {
        pc!.onicegatheringstatechange = () => {
          if (pc!.iceGatheringState === 'complete') {
            resolve(null);
          }
        };
      }
    });

    const response = await fetch('https://localhost:8000/offer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(pc!.localDescription)
    });

    const answer = await response.json();
    await pc!.setRemoteDescription(answer);

    console.log('✅ Peer connection established / renegotiated');
  } catch (err) {
    console.error('❌ Negotiation error:', err);
  }
}

export async function sendAudioToPython(audioTrack: MediaStreamTrack) {
  if (!pc) {
    console.log('📡 Setting up peer connection with audio track...');
    await setupPeerConnection(audioTrack);
  } else {
    console.log('📡 Adding track and renegotiating...');
    pc.addTransceiver(audioTrack, { direction: "sendrecv" });
    await negotiate();
  }
}


export async function sendVideoToPython(videoTrack: MediaStreamTrack) {
  if (!pc) {
    console.log('📹 Setting up peer connection with video track...');
    await setupPeerConnection();
  } else {
    console.log('📹 Adding video track and renegotiating...');
    pc.addTransceiver(videoTrack, { direction: "sendrecv" });
    await negotiate();
  }
}