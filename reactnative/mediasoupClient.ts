import * as mediasoupClient from 'mediasoup-client';
import { MediaStream, MediaStreamTrack, RTCPeerConnection, registerGlobals } from 'react-native-webrtc';
import { socket } from './socket';

let device: mediasoupClient.Device;
let sendTransport: mediasoupClient.types.Transport;
let recvTransport: mediasoupClient.types.Transport;
const consumedProducers = new Set<string>();
let pc: RTCPeerConnection | null = null;

export async function startMediasoup(
  roomCode: string,
  onNewConsumerStream: (stream: MediaStream, kind: string) => void
): Promise<void> {
  return new Promise<void>((resolve) => {
    socket.emit('get-rtp-capabilities');

    socket.once('rtp-capabilities', async (rtpCapabilities) => {
      registerGlobals();
      device = new mediasoupClient.Device();
      await device.load({ routerRtpCapabilities: rtpCapabilities });

      // Send Transport
      socket.emit('create-transport', { direction: 'send' });
      socket.once('transport-created-send', async (params) => {
        sendTransport = device.createSendTransport(params);

        sendTransport.on('connect', async ({ dtlsParameters }, callback, errback) => {
          try {
            socket.emit('connect-transport-send', { dtlsParameters });
            socket.once('transport-connected-send', () => {
              callback();
            });
          } catch (error) {
            console.error('Error connecting transport:', error);
            errback(error as Error);
          }
        });

        sendTransport.on('produce', (params, callback) => {
          socket.emit('produce', { ...params, roomCode });
          socket.once('produced', ({ id }) => {
            callback({ id });
          });
        });

        // Recv Transport
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
    try {
      await sendTransport.produce({ track });
    } catch (error) {
      console.error('Error creating producer:', error);
    }
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
    
    // OPTIONAL: You can send to Python server if needed
    // if (kind === 'audio') await sendAudioToPython(consumer.track);
    // if (kind === 'video') await sendVideoToPython(consumer.track);
  });
}

// ---------------------------
// WebRTC PeerConnection helpers (if needed)
// ---------------------------
export async function setupPeerConnection(audioTrack?: MediaStreamTrack) {
  if (!pc) {
    pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
    });

    pc.onicecandidate = (event) => {
      if (event.candidate) {
        console.log('🧊 ICE candidate:', event.candidate);
      }
    };

    pc.ontrack = (event) => {
      console.log(`🎤 Received ${event.track.kind} track`);
      // In React Native, audio will just play if attached to a stream and bound to RTCView (video)
    };
  }

  if (audioTrack) {
    try {
      pc.addTransceiver(audioTrack, { direction: 'sendrecv' });
    } catch (error) {
      console.error('Error adding transceiver:', error);
    }
  }

  await negotiate();
}

async function negotiate() {
  try {
    const offer = await pc!.createOffer();
    await pc!.setLocalDescription(offer);

    await new Promise<void>((resolve) => {
      if (pc!.iceGatheringState === 'complete') {
        resolve();
      } else {
        pc!.onicegatheringstatechange = () => {
          if (pc!.iceGatheringState === 'complete') {
            resolve();
          }
        };
      }
    });

    const response = await fetch('https://10.0.2.2:8000/offer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(pc!.localDescription),
    });

    const answer = await response.json();
    await pc!.setRemoteDescription(answer);

    console.log('✅ Peer connection negotiated');
  } catch (err) {
    console.error('❌ Negotiation failed:', err);
  }
}

export async function sendAudioToPython(audioTrack: MediaStreamTrack) {
  if (!pc) {
    console.log('📡 Initializing peer connection with local audio');
    await setupPeerConnection(audioTrack);
  } else {
    try {
      pc.addTransceiver(audioTrack, { direction: 'sendrecv' });
      await negotiate();
    } catch (error) {
      console.error('Error adding transceiver:', error);
    }
  }
}

export async function sendVideoToPython(videoTrack: MediaStreamTrack) {
  if (!pc) {
    console.log('📹 Initializing peer connection with video');
    await setupPeerConnection();
  } else {
    pc.addTransceiver(videoTrack, { direction: 'sendrecv' });
    await negotiate();
  }
}
