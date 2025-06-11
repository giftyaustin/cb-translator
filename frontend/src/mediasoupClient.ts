import * as mediasoupClient from 'mediasoup-client';
import { socket } from './socket';

let device: mediasoupClient.Device;
let sendTransport: mediasoupClient.types.Transport;
let recvTransport: mediasoupClient.types.Transport;
const consumedProducers = new Set<string>();

export async function startMediasoup(
  roomCode: string,
  onNewConsumerStream: (stream: MediaStream, kind: string) => void
) {
  return new Promise<void>((resolve, reject) => {
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
          socket.once('transport-connected-send', () => {
            console.log('transport connected');
            callback();

          });
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

    // Pass both stream and kind to App
    onNewConsumerStream(stream, kind);
    if (kind === 'audio') {
      const track = stream.getAudioTracks()[0];
      if (track) {
        sendAudioToPython(track);
      }
    }
  });
}



export async function sendAudioToPython(audioTrack: MediaStreamTrack) {
  const pc = new RTCPeerConnection({
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
  });

  pc.addTransceiver(audioTrack, {direction: "sendrecv"});

pc.ontrack = (event) => {
  console.log('🎤 Received track from server:', event.track.kind);

  // If multiple tracks (e.g., audio + video), use event.streams[0]
  let stream;
  if (event.streams && event.streams[0]) {
    stream = event.streams[0];
  } else {
    stream = new MediaStream([event.track]);
  }

  // Create audio element only for audio tracks
  if (event.track.kind === 'audio') {
    const audioElement = document.createElement("audio");
    audioElement.autoplay = true;
    audioElement.controls = true; // Optional
    audioElement.srcObject = stream;
    console.log(stream);
    

    document.body.appendChild(audioElement);

    audioElement.play().then(() => {
      console.log('▶️ Playing audio from server');
    }).catch(err => {
      console.error('❌ Error playing audio:', err);
    });
  }
};


  pc.onicecandidate = (event) => {
    if (event.candidate) {
      console.log('🧊 ICE candidate:', event.candidate);
    }
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  await new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') {
      resolve(null);
    } else {
      pc.onicegatheringstatechange = () => {
        if (pc.iceGatheringState === 'complete') {
          resolve(null);
        }
      };
    }
  });

  try {
    const response = await fetch('http://localhost:8000/offer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(pc.localDescription)
    });
    const answer = await response.json();
    await pc.setRemoteDescription(answer);
  } catch (err) {
    console.error('❌ Error in WebRTC setup:', err);
  }
}



