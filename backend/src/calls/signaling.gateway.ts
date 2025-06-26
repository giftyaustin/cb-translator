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

// Map that keeps track of producers for each room
const rooms = new Map<
    string,
    {
        producers: Map<string, Producer>;
    }
>();

// Map that keeps track of the transport for each user
const userTransports = new Map<string, WebRtcTransport>();

let io: Server;

@WebSocketGateway({
    cors: {
        origin: '*',
    },
})

// class that defines the gateway methods
export class SignalingGateway implements OnGatewayInit {
    constructor(private readonly mediasoupService: MediasoupService) {}

    // when server is started, saves it in the io variable
    afterInit(server: Server) {
        io = server;
        this.mediasoupService.initMediasoup();
        console.log('🚀 Socket.IO Gateway ready');
    }

    // returns the configured capabilities (audio and video formats and framerates) to the clients
    @SubscribeMessage('get-rtp-capabilities')
    handleGetRtp(socket: Socket) {
        const rtpCapabilities = this.mediasoupService.getRtpCapabilities();
        socket.emit('rtp-capabilities', rtpCapabilities);
    }

    // method called by clients to tell the mediasoup server they will be consuming a certain socket
    @SubscribeMessage('create-transport')
    async handleCreateTransport(
        socket: Socket,
        data: { direction: 'send' | 'recv' },
    ) {
        const { direction } = data;

        // transport is created and saved
        const { transport, params } = await this.mediasoupService.createWebRtcTransport(); 
        userTransports.set(`${socket.id}-${direction}`, transport);
        socket.emit(`transport-created-${direction}`, params);  // lets the client know the transport was created, and the parameters
        
        // if the client wants to create a socket to send us data
        if (direction === 'send') {

            // method called when client wants to link the transport to their dtls parameters
            socket.on('connect-transport-send', async ({ dtlsParameters }) => {
                await transport.connect({ dtlsParameters });    // configures transport to receive data using the peovided parameters
                socket.emit('transport-connected-send');        // letting the client know the connection was successfull
            });

            // method called when client is setting up their stream (to send data)
            // letting us know the stream's rtpParameters (and room code)
            socket.on('produce', async ({ kind, rtpParameters, roomCode }) => {

                // this mediasoup server only accepts audio and video
                if (kind !== 'audio' && kind !== 'video') {
                    socket.emit('produce-error', `Invalid media kind: ${kind}`);
                    return;
                }

                // creates the producer for the provided rtp parameters
                const producer = await transport.produce({
                    kind: kind as MediaKind,
                    rtpParameters,
                });
                
                // also creates the variable that will store the audio translation producer
                // (if the track is an audio track and the connection to the python server is successfull)
                let translationProducer: Producer<AppData> | null = null;

                // For audio tracks, we try to set up a translator track
                if (kind === 'audio') { try {
                    const rtpPort = getPort();

                    ///////////////////////////////// [Mediasoup -> Translator] /////////////////////////////////

                    // creates transport that sends audio to the translator
                    const translationSendTransport = await this.mediasoupService.createPlainTransport("send");
                    await translationSendTransport.connect({
                        ip: '127.0.0.1',    // ip of the translator service
                        port: rtpPort,      // port of the translator service
                    });
                    
                    // creates a consumer in the tansport. It will take the audio from the client's producer and send it to the translator
                    const consumer = await translationSendTransport.consume({
                        producerId: producer.id,
                        rtpCapabilities: this.mediasoupService.getRtpCapabilities(),
                    });

                    ///////////////////////////////// [Translator -> Mediasoup ] /////////////////////////////////

                    // creates a transport that receives audio from the translator
                    const translationReceiveTransport = await this.mediasoupService.createPlainTransport("recv");

                    await translationReceiveTransport.connect({
                        ip: '127.0.0.1',    // Translator will send audio to this IP
                        port: translationReceiveTransport.tuple.localPort,   // Translator will send audio to this port
                    });

                    // lets the python translation server know all the parameters for the connection
                    const codec = consumer.rtpParameters.codecs[0];
                    const payloadType = codec.payloadType;
                    const codecName = codec.mimeType.split('/')[1];
                    const clockRate = codec.clockRate;
                    const channels = codec.channels || 2;
                    const ssrc = randomInt(1, 0x7FFFFFFF);
                    io.emit("translation:initiate", {   // Emit translation initiation to the Python server
                        producerId: producer.id,
                        rtpPort: rtpPort,               // Send the unique RTP port to the server
                        ip: translationSendTransport.tuple.localIp,
                        codec: codecName,
                        clockRate,
                        channels,
                        payloadType,
                        ssrc,
                        outputPort:translationReceiveTransport.tuple.localPort
                    });

                    // Then, a producer is created that will read from the translationReceiveTransport and send it to the client
                    translationProducer = await translationReceiveTransport.produce({
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
                } catch (error){
                    console.log(`⚠️⚠️⚠️ Error when setting up the translation track for ${socket} - ${producer}`);
                    translationProducer = null;
                }}

                socket.join(roomCode);  // adds this socket to the room (mediasoup creates the room if it didnt exist)

                // adds this room to our internal map of rooms, without any producers
                if (!rooms.has(roomCode)) {
                    rooms.set(roomCode, { producers: new Map() });
                }

                // If the client is setting up its audio track, and a translator was properly created,
                // adds only the translated audio track to the room
                if (kind === 'audio' && translationProducer != null) {
                    rooms.get(roomCode)!.producers.set(`${socket.id}-translated`, translationProducer);
                    socket.to(roomCode).emit('new-producer', {
                        producerId: translationProducer.id,
                        socketId: socket.id,
                        kind,
                    });
                }
                
                // if the client is setting up its video stream, or if something went wrong when creating the translator,
                // adds the received track (audio or video) to the room
                else {
                    rooms.get(roomCode)!.producers.set(socket.id, producer);
                    socket.to(roomCode).emit('new-producer', {
                        producerId: producer.id,
                        socketId: socket.id,
                        kind,
                    });
                }

                socket.emit('produced', { id: producer.id });   // lets the client know the opperation was successfull
            });
        }

        // if the cliend wants to create a socket to receive data
        if (direction === 'recv') {
            socket.on('connect-transport-recv', async ({ dtlsParameters }) => {
                await transport.connect({ dtlsParameters });
                socket.emit('transport-connected-recv');
            });
        }
    }

    // method called when a client wants to consume from a certain producer
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
        // gets the router from mediasoup
        const router = this.mediasoupService.getRouter();

        // checks if the router is able to return the requested data (for the requested producer)
        if (!router.canConsume({ producerId, rtpCapabilities })) {
            socket.emit('consume-error', 'Cannot consume this stream');
            return;
        }

        // retrieves the "recieve" transport for the client
        const transport = userTransports.get(`${socket.id}-recv`);

        // checks if it exists
        if (!transport) {
            socket.emit('consume-error', 'No transport found');
            return;
        }

        // creates consumer, in the client's "receive" transport, reading from the requested producer
        const consumer = await transport.consume({
            producerId,
            rtpCapabilities,
            paused: false,
        });

        // lets the client know the consumer was successfully created
        socket.emit('consumed', {
            id: consumer.id,
            kind: consumer.kind,
            rtpParameters: consumer.rtpParameters,
            producerId,
        });

        // makes sure the consumer is running
        await consumer.resume();
    }
}