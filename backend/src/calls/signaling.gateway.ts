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
    PlainTransport,
} from 'mediasoup/node/lib/types';
import { spawn } from 'child_process';
import { getPort } from 'src/lib/port';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { msConfig } from './constants';

const rooms = new Map<
    string,
    {
        producers: Map<string, Producer>;
    }
>();

const userTransports = new Map<string, WebRtcTransport>();
const plainTransports = new Map<string, PlainTransport>();


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

                if (kind === 'audio') {

                    const rtpPort = 25000;
                    const rtcpPort = getPort();

                    const audioPlainTransport =
                        await this.mediasoupService.createPlainTransport("send");
                    // const recvTransport = await this.mediasoupService.createPlainTransport("recv");
                    // const ffmpegProducer = await recvTransport.produce({
                    //     kind: "audio",
                    //     rtpParameters: rtpParameters,
                    // })

                    // const listenRtpIp = recvTransport.tuple.localIp;
                    // const listenRtpPort = recvTransport.tuple.localPort;
                    // const listenRtcpPort = recvTransport.rtcpTuple?.localPort;
                    await audioPlainTransport.connect({
                        ip: '127.0.0.1',
                        port: rtpPort,
                        rtcpPort,
                    });

                    const consumer = await audioPlainTransport.consume({
                        producerId: producer.id,
                        rtpCapabilities: this.mediasoupService.getRtpCapabilities(),
                    });

                    const codec = consumer.rtpParameters.codecs[0];
                    const payloadType = codec.payloadType;
                    const codecName = codec.mimeType.split('/')[1];
                    const clockRate = codec.clockRate;
                    const channels = codec.channels || 2;


                    io.emit("translation:initiate", {
                        producerId: producer.id,
                        rtpPort: rtpPort,
                        rtcpPort: rtcpPort,
                        ip: audioPlainTransport.tuple.localIp,
                        codec: codecName,
                        clockRate,
                        channels,
                        payloadType,
                        listenRtpId: 20000
                    });
                }

                socket.join(roomCode);

                if (!rooms.has(roomCode)) {
                    rooms.set(roomCode, { producers: new Map() });
                }

                rooms.get(roomCode)!.producers.set(socket.id, producer);

                socket.to(roomCode).emit('new-producer', {
                    producerId: producer.id,
                    socketId: socket.id,
                    kind,
                });

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
