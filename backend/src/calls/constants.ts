export const msConfig = {
    announcedIp: '127.0.0.1', // using loopback (localhost) for development
    worker: {
        rtcMinPort: 10000,
        rtcMaxPort: 10100,
    },
    router: {
        mediaCodecs: [
            {
                kind: 'audio',
                mimeType: 'audio/opus',
                clockRate: 48000,
                channels: 2,
            },
            {
                kind: 'video',
                mimeType: 'video/VP8',
                clockRate: 90000,
            },
        ],
    },
    transport: {},
    ffmpeg: {
        port: 5004,
        rtcpPort: 5005
    },


}