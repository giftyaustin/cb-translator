import {
    SubscribeMessage,
    WebSocketGateway,
    OnGatewayInit,
} from '@nestjs/websockets';
import { Socket, Server } from 'socket.io';
import { MediasoupService } from './Mediasoup.service';
import {
    Producer,
    RtpCapabilities,
    WebRtcTransport,
    MediaKind,
    AppData
} from 'mediasoup/node/lib/types';
import { randomInt } from 'crypto';
import { getPort } from './port';
 
const rooms = new Map<
    string,
    {
        producers: Map<string, Producer>;
    }
>();
 
const userTransports = new Map<string, WebRtcTransport>();
 
let io: Server;
 
@WebSocketGateway({
    cors: {
        origin: '*',
    },
})
export class SignalingGateway implements OnGatewayInit {
    constructor(private readonly mediasoupService: MediasoupService) { }
 
    afterInit(server: Server) {
        io = server;
        this.mediasoupService.initMediasoup();
        console.log('🚀 Socket.IO Gateway ready');
    }
 
    @SubscribeMessage('get-rtp-capabilities')
    handleGetRtp(socket: Socket) {
        const rtpCapabilities = this.mediasoupService.getRtpCapabilities();
        socket.emit('rtp-capabilities', rtpCapabilities);
    }
 
    @SubscribeMessage('create-transport')
    async handleCreateTransport(
        socket: Socket,
        data: { direction: 'send' | 'recv' },
    ) {
        const { direction } = data;
        const { transport, params } =
            await this.mediasoupService.createWebRtcTransport();
 
        userTransports.set(`${socket.id}-${direction}`, transport);
        socket.emit(`transport-created-${direction}`, params);
 
        if (direction === 'send') {
            socket.on('connect-transport-send', async ({ dtlsParameters }) => {
                await transport.connect({ dtlsParameters });
                socket.emit('transport-connected-send');
            });
 
            socket.on('produce', async ({ kind, rtpParameters, roomCode }) => {
                if (kind !== 'audio' && kind !== 'video') {
                    socket.emit('produce-error', `Invalid media kind: ${kind}`);
                    return;
                }
 
                const producer = await transport.produce({
                    kind: kind as MediaKind,
                    rtpParameters,
                });
 
                let ffmpegProducer: Producer<AppData> | null = null;
 
                if (kind === 'audio') {
                    const rtpPort = getPort();
                    // const rtpPort = 25000;
                    // [Mediasoup -> FFmpeg]
                    const audioPlainTransport = await this.mediasoupService.createPlainTransport("send");
                    await audioPlainTransport.connect({
                        ip: '127.0.0.1',
                        port: rtpPort,
                    });
 
                    const consumer = await audioPlainTransport.consume({
                        producerId: producer.id,
                        rtpCapabilities: this.mediasoupService.getRtpCapabilities(),
                    });
 
                    // [FFmpeg -> Mediasoup]
                    const recvTransport = await this.mediasoupService.createPlainTransport("recv");
                    await recvTransport.connect({
                        ip: '127.0.0.1',    // FFmpeg sends audio to this IP
                        port: recvTransport.tuple.localPort,   // FFmpeg sends audio to this port
                    });
 
                    const codec = consumer.rtpParameters.codecs[0];
                    const payloadType = codec.payloadType;
                    const codecName = codec.mimeType.split('/')[1];
                    const clockRate = codec.clockRate;
                    const channels = codec.channels || 2;
                    const ssrc = randomInt(1, 0x7FFFFFFF);
                    // Emit translation initiation to the Python server
                    // io.emit("translation:initiate", {
                    //     producerId: producer.id,
                    //     rtpPort: rtpPort, // Send the unique RTP port to the server
                    //     ip: audioPlainTransport.tuple.localIp,
                    //     codec: codecName,
                    //     clockRate,
                    //     channels,
                    //     payloadType,
                    //     ssrc,
                    //     outputPort:recvTransport.tuple.localPort
                    // });
                    const payload = {
                        producerId: producer.id,
                        rtpPort: rtpPort,
                        ip: audioPlainTransport.tuple.localIp,
                        codec: codecName,
                        clockRate,
                        channels,
                        payloadType,
                        ssrc,
                        outputPort: recvTransport.tuple.localPort
                    };
 
                    fetch("http://0.0.0.0:2002/translation/initiate", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify(payload)
                    })
                        .then(response => response.json())
                        .then(data => {
                            console.log("✅ Translation pipeline initiated:", data);
                        })
                        .catch(error => {
                            console.error("❌ Error initiating translation pipeline:", error);
                        });
 
                    // Consume the audio data from FFmpeg
                    ffmpegProducer = await recvTransport.produce({
                        kind: 'audio',
                        rtpParameters: {
                            codecs: [
                                {
                                    mimeType: 'audio/opus',
                                    payloadType,
                                    clockRate,
                                    channels
                                },
                            ],
                            encodings: [{ ssrc }]
                        },
                    });
                }
 
                socket.join(roomCode);
 
                if (!rooms.has(roomCode)) {
                    rooms.set(roomCode, { producers: new Map() });
                }
 
                if (kind === 'audio') {
                    if (ffmpegProducer != null) {
                        rooms.get(roomCode)!.producers.set(`${socket.id}-ffmpeg`, ffmpegProducer);
                        socket.to(roomCode).emit('new-producer', {
                            producerId: ffmpegProducer.id,
                            socketId: socket.id,
                            kind,
                        });
                    }
                } else {
                    rooms.get(roomCode)!.producers.set(socket.id, producer);
                    socket.to(roomCode).emit('new-producer', {
                        producerId: producer.id,
                        socketId: socket.id,
                        kind,
                    });
                }


                // =======
                if (kind === 'video') {
                    const rtpPort = getPort();  // Allocate a dynamic RTP port
 
                    const videoPlainTransport = await this.mediasoupService.createPlainTransport("send");
                    await videoPlainTransport.connect({
                        ip: '127.0.0.1',
                        port: rtpPort,
                    });
 
                    const consumer = await videoPlainTransport.consume({
                        producerId: producer.id,
                        rtpCapabilities: this.mediasoupService.getRtpCapabilities(),
                    });
 
                    const codec = consumer.rtpParameters.codecs[0];
                    const payloadType = codec.payloadType;
                    const codecName = codec.mimeType.split('/')[1]; // Should be "H264"
                    const clockRate = codec.clockRate;
                    const ssrc = randomInt(1, 0x7FFFFFFF);
 
                    const payload = {
                        rtpPort: rtpPort,
                        ip: videoPlainTransport.tuple.localIp,
                        codec: codecName,
                        clockRate,
                        payloadType,
                    };
 
                    fetch("http://0.0.0.0:2002/video/initiate", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload),
                    })
                        .then(res => res.json())
                        .then(data => console.log("✅ Video capture pipeline initiated:", data))
                        .catch(err => console.error("❌ Error initiating video capture pipeline:", err));
                }
                // ====
 
                socket.emit('produced', { id: producer.id });
            });
        }
 
        if (direction === 'recv') {
            socket.on('connect-transport-recv', async ({ dtlsParameters }) => {
                await transport.connect({ dtlsParameters });
                socket.emit('transport-connected-recv');
            });
        }
    }
 
    @SubscribeMessage('consume')
    async handleConsume(
        socket: Socket,
        {
            producerId,
            rtpCapabilities,
        }: {
            producerId: string;
            rtpCapabilities: RtpCapabilities;
        },
    ) {
        const router = this.mediasoupService.getRouter();
 
        if (!router.canConsume({ producerId, rtpCapabilities })) {
            socket.emit('consume-error', 'Cannot consume this stream');
            return;
        }
 
        const transport = userTransports.get(`${socket.id}-recv`);
        if (!transport) {
            socket.emit('consume-error', 'No transport found');
            return;
        }
 
        const consumer = await transport.consume({
            producerId,
            rtpCapabilities,
            paused: false,
        });
 
        socket.emit('consumed', {
            id: consumer.id,
            kind: consumer.kind,
            rtpParameters: consumer.rtpParameters,
            producerId,
        });
 
        await consumer.resume();
    }
}
 