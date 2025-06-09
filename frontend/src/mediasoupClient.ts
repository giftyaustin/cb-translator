import * as mediasoupClient from 'mediasoup-client';
import { socket } from './socket';

let device: mediasoupClient.Device;
let sendTransport: mediasoupClient.types.Transport;
let recvTransport: mediasoupClient.types.Transport;
const consumedProducers = new Set<string>();

export async function startMediasoup(
  roomCode: string,
  onNewConsumerStream: (stream: MediaStream) => void
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
          socket.once('transport-connected-send', ()=>{
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
  onNewConsumerStream: (stream: MediaStream) => void
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

    // Send stream back to React
    onNewConsumerStream(stream);
  });
}
