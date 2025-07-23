import { io } from 'socket.io-client';

const socketUrl = 'http://10.10.0.203:3000';
export const socket = io(socketUrl, {
  transports: ['websocket'],
});
