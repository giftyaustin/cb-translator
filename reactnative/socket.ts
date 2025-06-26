import { io } from 'socket.io-client';

export const socket = io('http://10.10.0.82:3000', {
  transports: ['websocket'],
  forceNew: true,
});
